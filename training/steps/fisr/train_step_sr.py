"""
Super-Resolution Training Step
Trains with LR input, HR supervision, and advanced loss composition
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from typing import Dict, Callable, Optional

from training.train_utils.srcnn.losses_sr import (
    CharbonnierLoss,
    SSIMLoss,
    FFTLoss,
    MaskedReconstructionLoss,
    compute_multiresolution_losses
)
from training.core.config_sr import SRTrainingConfig


def train_one_epoch_sr(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: SRTrainingConfig,
    log_interval: int = 10,
    save_every_batches: int = 0,
    on_batch_checkpoint_callback_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Train for one epoch in Super-Resolution mode
    
    Input: LR images (B, C, H, W)
    Target: HR images (B, C, H*scale, W*scale)
    Losses: MSE + L1 + SSIM + FFT + Flux
    
    Args:
        model: Model to train (with sr_mode=True)
        train_loader: Training data loader with 'lr_image' and 'hr_image' batches
        optimizer: Optimizer
        device: Torch device
        config: SRTrainingConfig instance with all loss weights and parameters
        log_interval: Logging interval (batches)
        save_every_batches: Save checkpoint every N batches (0 to disable)
        on_batch_checkpoint_callback_fn: Callback function for batch checkpointing
    Returns:
        Dictionary with loss values (loss_total, loss_recon, loss_l1, loss_ssim, loss_fft, loss_flux)
    """
    model.train()
    
    # Initialize loss functions
    l1_loss_fn = CharbonnierLoss(epsilon=1e-3)
    ssim_loss_fn = SSIMLoss(window_size=11)
    fft_loss_fn = FFTLoss()
    masked_recon_loss_fn = MaskedReconstructionLoss(
        visible_weight=config.masked_visible_weight
    )
    
    # Loss accumulators
    total_loss = 0.0
    total_loss_recon = 0.0
    total_loss_l1 = 0.0
    total_loss_ssim = 0.0
    total_loss_fft = 0.0
    total_loss_flux = 0.0
    num_batches = 0
    
    progress_bar = tqdm.tqdm(train_loader, desc="Training (SR)")
    
    for batch_index, batch in enumerate(progress_bar):
        # Load batch: LR images as input, HR images as target
        lr_images = batch["lr_image"].to(device)
        hr_images = batch["hr_image"].to(device)
        
        # Verify that HR has correct size (LR * scale_factor)
        expected_hr_size = lr_images.shape[-1] * config.scale_factor
        assert hr_images.shape[-1] == expected_hr_size, \
            f"HR size {hr_images.shape[-1]} != expected {expected_hr_size}"
        
        # Forward pass in SR mode
        # Input: LR images, Output: HR reconstruction
        reconstructed, aux_outputs = model(lr_images, sr_mode=True)
        
        # Verify output shape
        assert reconstructed.shape == hr_images.shape, \
            f"Output shape {reconstructed.shape} != target shape {hr_images.shape}"
        
        # ============ COMPUTE LOSSES ============
        
        # 1. Masked Reconstruction Loss (MAE with masked patches only)
        loss_recon = masked_recon_loss_fn(
            pred=reconstructed,
            target=hr_images,
            mask=aux_outputs['mask'],
            patch_size=config.patch_size
        )
        
        # 2. L1 / Charbonnier Loss (encourages sharp edges)
        loss_l1 = l1_loss_fn(reconstructed, hr_images)
        
        # 3. SSIM Loss (perceptual quality)
        loss_ssim = ssim_loss_fn(reconstructed, hr_images, data_range=1.0)
        
        # 4. FFT Loss (frequency domain / global structure)
        loss_fft = fft_loss_fn(reconstructed, hr_images)
        
        # 5. Flux Loss (either sparsity or target consistency)
        if config.flux_loss_mode == "target":
            # True flux consistency (if flux_map_target is available)
            flux_map_target = aux_outputs.get('flux_map_target', None)
            loss_flux = model.get_flux_loss(
                aux_outputs=aux_outputs,
                flux_map_target=flux_map_target,
                mode="target"
            )
        else:
            # Flux sparsity regularization (default)
            loss_flux = model.get_flux_loss(
                aux_outputs=aux_outputs,
                flux_map_target=None,
                mode="sparsity"
            )
        
        # ============ TOTAL LOSS ============
        total_weighted_loss = (
            config.lambda_recon * loss_recon +
            config.lambda_l1 * loss_l1 +
            config.lambda_ssim * loss_ssim +
            config.lambda_fft * loss_fft +
            config.lambda_flux * loss_flux
        )
        
        # ============ OPTIONAL MULTI-SCALE SUPERVISION ============
        multiscale_losses = {}
        if config.enable_multiscale:
            multiscale_losses = compute_multiresolution_losses(
                pred=reconstructed,
                target=hr_images,
                scales=[1, 2, 4]  # Full res, 2x down, 4x down
            )
            
            # Add weighted multiscale losses
            for scale_idx, (loss_key, loss_val) in enumerate(multiscale_losses.items()):
                weight = config.multiscale_weights[scale_idx] if scale_idx < len(config.multiscale_weights) else 1.0
                total_weighted_loss += weight * loss_val
        
        # ============ BACKWARD PASS ============
        optimizer.zero_grad()
        total_weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # ============ CHECKPOINT SAVING ============
        if (save_every_batches > 0
                and on_batch_checkpoint_callback_fn is not None
                and (batch_index + 1) % save_every_batches == 0):
            on_batch_checkpoint_callback_fn(batch_index + 1)
        
        # ============ LOSS TRACKING ============
        total_loss += total_weighted_loss.item()
        total_loss_recon += loss_recon.item()
        total_loss_l1 += loss_l1.item()
        total_loss_ssim += loss_ssim.item()
        total_loss_fft += loss_fft.item()
        total_loss_flux += loss_flux.item()
        num_batches += 1
        
        # ============ PROGRESS LOGGING ============
        if (batch_index + 1) % log_interval == 0:
            avg_total = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_total:.4f}',
                'recon': f'{total_loss_recon / num_batches:.4f}',
                'l1': f'{total_loss_l1 / num_batches:.4f}',
                'ssim': f'{total_loss_ssim / num_batches:.4f}',
                'fft': f'{total_loss_fft / num_batches:.4f}',
                'flux': f'{total_loss_flux / num_batches:.4f}',
            })
    
    # Return averaged losses
    return {
        'loss_total': total_loss / num_batches,
        'loss_recon': total_loss_recon / num_batches,
        'loss_l1': total_loss_l1 / num_batches,
        'loss_ssim': total_loss_ssim / num_batches,
        'loss_fft': total_loss_fft / num_batches,
        'loss_flux': total_loss_flux / num_batches,
    }
