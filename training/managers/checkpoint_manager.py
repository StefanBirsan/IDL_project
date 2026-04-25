"""
Checkpoint management for model training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


class CheckpointManager:
    """Manages saving and loading of model checkpoints"""
    
    def __init__(self, save_dir: str):
        """
        Initialize checkpoint manager
        
        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self,
             epoch: int,
             model: nn.Module,
             optimizer: optim.Optimizer,
             scheduler: Optional[Any],
             config: Dict,
             global_step: int,
             is_best: bool = False) -> Path:
        """
        Save checkpoint
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            config: Training configuration dictionary
            global_step: Global training step
            is_best: Whether this is the best checkpoint
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': config,
            'global_step': global_step,
        }
        
        # Regular checkpoint
        epoch_num = epoch + 1
        ckpt_path = self.save_dir / f'checkpoint_epoch_{epoch_num:04d}.pt'
        torch.save(checkpoint, ckpt_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.save_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")
        else:
            print(f"Saved checkpoint to {ckpt_path}")
        
        return ckpt_path
    
    def load(self,
             checkpoint_path: str,
             model: nn.Module,
             optimizer: optim.Optimizer,
             scheduler: Any,
             device: str) -> int:
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint on
        Returns:
            Epoch of loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        checkpoints = sorted(self.save_dir.glob('checkpoint_epoch_*.pt'))
        return checkpoints[-1] if checkpoints else None
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        best_path = self.save_dir / 'checkpoint_best.pt'
        return best_path if best_path.exists() else None
