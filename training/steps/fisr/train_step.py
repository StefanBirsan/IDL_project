"""
Single training step
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from typing import Dict

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_flux: float = 0.01,
    log_interval: int = 10,
    save_every_batches: int = 0,
    on_batch_checkpoint_callback_fn = None,
) -> Dict[str, float]:
    """
    Train for one epoch, using LR input and HR target
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        lambda_flux: Weight for flux loss term
        log_interval: Logging interval
        save_every_batches: Save checkpoint every N batches (0 to disable)
        on_batch_checkpoint_callback_fn: Callback function to call every N batches
    Returns:
        Dictionary with loss values
    """
    model.train()
    
    total_loss = 0.0
    total_loss_recon = 0.0
    total_loss_flux = 0.0
    num_batches = 0
    
    progress_bar = tqdm.tqdm(train_loader, desc="Training")
    
    for batch_index, batch in enumerate(progress_bar):
        lr_images = batch["lr_image"].to(device)
        hr_images = batch["hr_image"].to(device)
        
        # Forward pass on LR images
        reconstructed, aux_outputs = model(lr_images)

        hr_for_loss = F.interpolate(hr_images, size=lr_images.shape[-2:])
        
        # Compute loss against HR
        loss, loss_dict = model.get_loss(
            inputs=hr_for_loss,
            reconstructed=reconstructed,
            aux_outputs=aux_outputs,
            flux_map_target=None,
            lambda_flux=lambda_flux
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (save_every_batches > 0
                and on_batch_checkpoint_callback_fn is not None
                and (batch_index + 1) % save_every_batches == 0):
            on_batch_checkpoint_callback_fn(batch_index + 1)

        # Track loss
        total_loss += loss_dict['loss_total']
        total_loss_recon += loss_dict['loss_recon']
        total_loss_flux += loss_dict['loss_flux']
        num_batches += 1
        
        # Update progress bar
        if (batch_index + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'loss_recon': f'{total_loss_recon / num_batches:.4f}',
                'loss_flux': f'{total_loss_flux / num_batches:.4f}',
            })
    
    return {
        'loss_total': total_loss / num_batches,
        'loss_recon': total_loss_recon / num_batches,
        'loss_flux': total_loss_flux / num_batches,
    }
