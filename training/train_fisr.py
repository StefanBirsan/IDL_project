"""Training entrypoint for FISR model."""

import argparse

from training.core.config_fisr import FISRTrainingConfig
from training.core.trainer_fisr import FISRTrainer
from utils import get_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FISR (x2/x4) model")

    parser.add_argument("--variant", type=str, default="x2", choices=["x2", "x4"])
    parser.add_argument("--data-dir", type=str, default="dataset/data/x2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=20)

    # learning rate for AdamW optimizer (training stability + speed)
    parser.add_argument("--lr", type=float, default=2e-4)

    # weight decay for AdamW optimizer, helps reduce overfitting
    #  which is when the model learns the data too well
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)

    # base channel width for the backbone model
    #  bigger values increase capacity but also 
    #  increase memory usage and training time
    parser.add_argument("--dim", type=int, default=48)

    # number of transformer blocks in the final refinement stage
    parser.add_argument("--num-refinement-blocks", type=int, default=4)

    # whether to enable the prompt-conditioned decoder path
    parser.add_argument("--decoder", action="store_true", default=True)

    # explicit disabling of the decoder (for ablation, 
    #  i.e. experimenting with enabling/disabling certain 
    #  model modules to see their impact)
    parser.add_argument("--no-decoder", action="store_false", dest="decoder")
    
    # the type of loss function to use for training, either L1 or L2
    parser.add_argument("--use-loss", type=str, default="L1", choices=["L1", "L2"])
    
    # enable extra flux attention modules in the backbone (for ablation)
    parser.add_argument("--use-attention", action="store_true", default=False)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints/fisr")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = FISRTrainingConfig(
        variant=args.variant,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        dim=args.dim,
        num_refinement_blocks=args.num_refinement_blocks,
        decoder=args.decoder,
        use_loss=args.use_loss,
        use_attention=args.use_attention,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )

    trainer = FISRTrainer(config)

    train_loader = get_dataloader(
        config.data_dir,
        split="train",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    eval_loader = get_dataloader(
        config.data_dir,
        split="eval",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    trainer.train(train_loader, eval_loader)


if __name__ == "__main__":
    main()
