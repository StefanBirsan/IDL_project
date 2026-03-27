"""
Training script for Physics-Informed Masked Vision Transformer
Modular training pipeline with clean separation of concerns
"""
import argparse
from training.core import TrainingConfig, Trainer
from utils import get_dataloader




def main():
    """Main entry point for training"""
    parser = argparse.ArgumentParser(
        description="Train Physics-Informed Masked Vision Transformer"
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='dataset/data',
                        help='Path to dataset directory')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--img-size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--embed-dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--encoder-depth', type=int, default=12,
                        help='Number of encoder blocks')
    parser.add_argument('--decoder-depth', type=int, default=8,
                        help='Number of decoder blocks')
    parser.add_argument('--num-heads', type=int, default=12,
                        help='Number of attention heads')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1.5e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--lambda-flux', type=float, default=0.01,
                        help='Flux loss weight')
    parser.add_argument('--mask-ratio', type=float, default=0.75,
                        help='Masking ratio (0.75-0.90)')
    
    # Hardware & logging arguments
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Checkpoint save interval')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval')
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = TrainingConfig(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lambda_flux=args.lambda_flux,
        mask_ratio=args.mask_ratio,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Create data loaders
    train_loader = get_dataloader(
        config.data_dir,
        split='train',
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        img_size=config.img_size,
    )
    
    eval_loader = get_dataloader(
        config.data_dir,
        split='eval',
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        img_size=config.img_size,
    )
    
    # Run training
    trainer.train(train_loader, eval_loader)


if __name__ == '__main__':
    main()
