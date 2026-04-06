"""Trainer for full FISR architecture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from training.core.config_fisr import FISRTrainingConfig
from training.managers import CheckpointManager
from training.steps import MetricTracker
from training.train_utils import create_fisr_x2, create_fisr_x4


class FISRTrainer:
    def __init__(self, config: FISRTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)

        if config.variant == "x4":
            self.model = create_fisr_x4(
                inp_channels=config.inp_channels,
                out_channels=config.out_channels,
                dim=config.dim,
                num_blocks=list(config.num_blocks),
                num_refinement_blocks=config.num_refinement_blocks,
                heads=list(config.heads),
                ffn_expansion_factor=config.ffn_expansion_factor,
                bias=config.bias,
                layer_norm_type=config.layer_norm_type,
                decoder=config.decoder,
                use_loss=config.use_loss,
                use_attention=config.use_attention,
            ).to(self.device)
        else:
            self.model = create_fisr_x2(
                inp_channels=config.inp_channels,
                out_channels=config.out_channels,
                dim=config.dim,
                num_blocks=list(config.num_blocks),
                num_refinement_blocks=config.num_refinement_blocks,
                heads=list(config.heads),
                ffn_expansion_factor=config.ffn_expansion_factor,
                bias=config.bias,
                layer_norm_type=config.layer_norm_type,
                decoder=config.decoder,
                use_loss=config.use_loss,
                use_attention=config.use_attention,
            ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)

        self.checkpoint_manager = CheckpointManager(config.save_dir)
        self.metrics = MetricTracker()
        self.global_step = 0

    @staticmethod
    def _to_star_flux_map(flux_map: torch.Tensor) -> torch.Tensor:
        # STAR FISR expects (B,H,W), while this workspace dataset commonly yields (B,1,H,W).
        if flux_map.ndim == 4 and flux_map.shape[1] == 1:
            return flux_map.squeeze(1)
        return flux_map

    @staticmethod
    def _align_to_prediction(tensor: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-2:] != pred.shape[-2:]:
            return torch.nn.functional.interpolate(tensor, size=pred.shape[-2:], mode="nearest")
        return tensor

    def _build_targets(self, batch: dict[str, torch.Tensor], pred: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        hr = batch["hr_image"].to(self.device)
        mask = batch["hr_mask"].to(self.device)
        flux_map = self._to_star_flux_map(batch["lr_mask"].to(self.device))

        if pred is not None:
            hr = self._align_to_prediction(hr, pred)
            mask = self._align_to_prediction(mask, pred)

        targets = {
            "hr": hr,
            "mask": mask,
            "flux_map": flux_map,
            # Keep attention target available if use_attention is enabled.
            "attn_map": mask,
        }
        return targets

    def _compute_eval_loss(self, pred: torch.Tensor, targets: dict[str, torch.Tensor]) -> dict[str, float]:
        mask = targets["mask"]
        hr = targets["hr"]
        mask_sum = float(mask.sum().item() + 1e-3)

        if self.config.use_loss == "L2":
            base_loss = float((((pred - hr) ** 2) * mask).sum().item() / mask_sum)
            loss_name = "l2_loss"
        else:
            base_loss = float((torch.abs(pred - hr) * mask).sum().item() / mask_sum)
            loss_name = "l1_loss"

        losses = {
            "loss_total": base_loss,
            loss_name: base_loss,
        }

        if self.config.use_attention:
            attn_map = torch.nan_to_num(targets["attn_map"], nan=0.0)
            flux_loss = float((torch.abs(pred - hr) * attn_map).sum().item() / float(attn_map.sum().item() + 1e-3))
            losses["flux_loss"] = 0.01 * flux_loss
            losses["loss_total"] = base_loss + 0.01 * flux_loss

        return losses

    def train(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None):
        print("\n" + "=" * 70)
        print("Starting FISR training")
        print(f"Variant: {self.config.variant}")
        print(f"Device: {self.device}")
        print("=" * 70 + "\n")

        best_eval_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch [{epoch + 1}/{self.config.num_epochs}]")
            self.model.train()

            train_total = 0.0
            train_recon = 0.0
            train_flux = 0.0
            train_batches = 0

            for batch_index, batch in enumerate(train_loader):
                lr = batch["lr_image"].to(self.device)

                # Build targets at input resolution first to satisfy forward dependencies,
                # then align to prediction once prediction exists if needed.
                targets = self._build_targets(batch)

                self.optimizer.zero_grad()
                total_loss, losses = self.model(lr, targets)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_total += float(total_loss.item())
                train_recon += float(next(iter(losses.values())).item())
                if "flux_loss" in losses:
                    train_flux += float(losses["flux_loss"].item())
                train_batches += 1

                if (batch_index + 1) % self.config.log_interval == 0:
                    print(
                        f"  Batch {batch_index + 1}/{len(train_loader)} | "
                        f"loss={train_total / train_batches:.4f}"
                    )

            train_log = {
                "loss_total": train_total / max(train_batches, 1),
                "loss_recon": train_recon / max(train_batches, 1),
                "loss_flux": train_flux / max(train_batches, 1),
            }
            self.metrics.add_train_loss(train_log)
            self.global_step += len(train_loader)

            print(f"Train Loss: {train_log['loss_total']:.4f}")

            is_best = False
            if eval_loader is not None:
                self.model.eval()
                eval_total = 0.0
                eval_recon = 0.0
                eval_flux = 0.0
                eval_batches = 0

                with torch.no_grad():
                    for batch in eval_loader:
                        lr = batch["lr_image"].to(self.device)
                        targets = self._build_targets(batch)
                        outputs = self.model(lr, targets)
                        pred = outputs["pred_img"]
                        targets = self._build_targets(batch, pred=pred)
                        eval_losses = self._compute_eval_loss(pred, targets)

                        eval_total += eval_losses["loss_total"]
                        eval_recon += eval_losses.get("l1_loss", eval_losses.get("l2_loss", eval_losses["loss_total"]))
                        eval_flux += eval_losses.get("flux_loss", 0.0)
                        eval_batches += 1

                eval_log = {
                    "loss_total": eval_total / max(eval_batches, 1),
                    "loss_recon": eval_recon / max(eval_batches, 1),
                    "loss_flux": eval_flux / max(eval_batches, 1),
                }
                self.metrics.add_eval_loss(eval_log)
                print(f"Eval Loss:  {eval_log['loss_total']:.4f}")
                if eval_log["loss_total"] < best_eval_loss:
                    best_eval_loss = eval_log["loss_total"]
                    is_best = True

            self.metrics.add_learning_rate(self.optimizer.param_groups[0]["lr"])

            if (epoch + 1) % self.config.save_interval == 0:
                self.checkpoint_manager.save(
                    epoch=epoch + 1,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    config=self.config.to_dict(),
                    global_step=self.global_step,
                    is_best=is_best,
                )

            self.scheduler.step()

        self.checkpoint_manager.save(
            epoch=self.config.num_epochs,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config.to_dict(),
            global_step=self.global_step,
            is_best=False,
        )

        history_path = Path(self.config.save_dir) / "training_history_fisr.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

        print("\n" + "=" * 70)
        print("FISR training completed")
        print(f"Checkpoints: {self.config.save_dir}")
        print(f"History: {history_path}")
        print("=" * 70 + "\n")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        return self.checkpoint_manager.load(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device),
        )
