"""
Training script for Physics-Informed Masked Vision Transformer
Supports both legacy mode (LR reconstruction) and SR mode (LR -> HR upscaling)
DEFAULT IS SR MODE
"""
import argparse
from training.core import TrainingConfig, Trainer
from training.core.config_sr import SRTrainingConfig
from training.core.trainer_sr import SRTrainer
from utils import get_dataloader




def main():
    """Main entry point for training"""
    parser = argparse.ArgumentParser(
        description="Train Physics-Informed Masked Vision Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SR mode (DEFAULT): 2x upscaling with advanced losses
  python training/train.py
  
  # SR with custom scale
  python training/train.py --scale-factor 4
  
  # Legacy mode: LR reconstruction for explainability
  python training/train.py --mode legacy
  
  # SR with custom loss weights
  python training/train.py --lambda-l1 0.7 --lambda-ssim 0.5 --lambda-fft 0.2
        """
    )
    
    # ============ MODE SELECTION ============
    parser.add_argument('--mode', type=str, default='sr', choices=['sr', 'legacy'],
                        help='Training mode: sr (DEFAULT, true upscaling) or legacy (LR reconstruction)')
    
    # ============ DATA ARGUMENTS ============
    parser.add_argument('--data-dir', type=str, default='dataset/data',
                        help='Path to dataset directory')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # ============ MODEL ARGUMENTS ============
    parser.add_argument('--img-size', type=int, default=128,
                        help='LR image size (default: 128)')
    parser.add_argument('--embed-dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--encoder-depth', type=int, default=12,
                        help='Number of encoder blocks')
    parser.add_argument('--decoder-depth', type=int, default=8,
                        help='Number of decoder blocks')
    parser.add_argument('--num-heads', type=int, default=12,
                        help='Number of attention heads')
    
    # ============ SR-SPECIFIC ARGUMENTS ============
    parser.add_argument('--scale-factor', type=int, default=2,
                        help='SR upscaling factor (2, 4, 8). Ignored in legacy mode.')
    
    # ============ LOSS WEIGHT ARGUMENTS (SR MODE) ============
    parser.add_argument('--lambda-recon', type=float, default=1.0,
                        help='Weight for reconstruction (MSE on masked patches) loss (SR mode)')
    parser.add_argument('--lambda-l1', type=float, default=0.5,
                        help='Weight for L1/Charbonnier loss (SR mode)')
    parser.add_argument('--lambda-ssim', type=float, default=0.3,
                        help='Weight for SSIM loss (SR mode)')
    parser.add_argument('--lambda-fft', type=float, default=0.1,
                        help='Weight for FFT loss (SR mode)')
    parser.add_argument('--lambda-flux', type=float, default=0.01,
                        help='Weight for flux loss (both modes)')
    
    # ============ ADVANCED SR ARGUMENTS ============
    parser.add_argument('--masked-visible-weight', type=float, default=0.0,
                        help='Weight for visible patches in MAE loss (0=masked-only, >0=include visible)')
    parser.add_argument('--flux-loss-mode', type=str, default='sparsity', choices=['sparsity', 'target'],
                        help='Flux loss mode: sparsity (regularize to sparse) or target (consistency)')
    parser.add_argument('--enable-multiscale', action='store_true',
                        help='Enable multi-scale supervision (1x, 2x, 4x)')
    parser.add_argument('--multiscale-weights', type=float, nargs='+', default=[1.0],
                        help='Weights for each scale (requires --enable-multiscale)')
    
    # ============ TRAINING ARGUMENTS ============
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1.5e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--mask-ratio', type=float, default=0.75,
                        help='Masking ratio (0.75-0.90)')
    
    # ============ HARDWARE & LOGGING ARGUMENTS ============
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Checkpoint save interval (epochs)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval (batches)')
    parser.add_argument('--save-every-batches', type=int, default=0,
                        help='Save checkpoint every N intra-epoch batches (0 to disable)')
    
    args = parser.parse_args()
    
    # ============ TRAINING MODE DISPATCH ============
    
    if args.mode == 'sr':
        # ===== SUPER-RESOLUTION MODE (DEFAULT) =====
        print(f"\n{'='*70}")
        print("SUPER-RESOLUTION MODE (DEFAULT)")
        print(f"Scale Factor: {args.scale_factor}x ({args.img_size} -> {args.img_size * args.scale_factor})")
        print(f"{'='*70}\n")
        
        # Create SR config
        sr_config = SRTrainingConfig(
            # SR params
            scale_factor=args.scale_factor,
            
            # Model params
            img_size=args.img_size,
            embed_dim=args.embed_dim,
            encoder_depth=args.encoder_depth,
            decoder_depth=args.decoder_depth,
            num_heads=args.num_heads,
            mask_ratio=args.mask_ratio,
            
            # Loss weights
            lambda_recon=args.lambda_recon,
            lambda_l1=args.lambda_l1,
            lambda_ssim=args.lambda_ssim,
            lambda_fft=args.lambda_fft,
            lambda_flux=args.lambda_flux,
            
            # Loss config
            masked_visible_weight=args.masked_visible_weight,
            flux_loss_mode=args.flux_loss_mode,
            enable_multiscale=args.enable_multiscale,
            multiscale_weights=args.multiscale_weights,
            
            # Training params
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            
            # Data params
            data_dir=args.data_dir,
            num_workers=args.num_workers,
            
            # Hardware & logging
            device=args.device,
            seed=args.seed,
            save_dir=args.save_dir,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
            save_every_batches=args.save_every_batches,
        )
        
        # Create SR trainer
        trainer = SRTrainer(sr_config)
        
        # Create data loaders (same for both modes)
        train_loader = get_dataloader(
            sr_config.data_dir,
            split='train',
            batch_size=sr_config.batch_size,
            num_workers=sr_config.num_workers,
            img_size=sr_config.img_size,
        )
        
        eval_loader = get_dataloader(
            sr_config.data_dir,
            split='eval',
            batch_size=sr_config.batch_size,
            num_workers=sr_config.num_workers,
            img_size=sr_config.img_size,
        )
        
        # Run SR training
        trainer.train(train_loader, eval_loader)
    
    else:
        # ===== LEGACY MODE (LR Reconstruction for backward compatibility) =====
        print(f"\n{'='*70}")
        print("LEGACY MODE (LR Reconstruction)")
        print("Note: Modern training should use --mode sr for better quality")
        print(f"{'='*70}\n")
        
        # Create legacy config
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
            save_every_batches=args.save_every_batches,
        )
        
        # Create legacy trainer
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
        
        # Run legacy training
        trainer.train(train_loader, eval_loader)


if __name__ == '__main__':
    main()
