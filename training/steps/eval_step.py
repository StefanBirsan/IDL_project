"""
Single evaluation step
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from typing import Dict


@torch.no_grad()
def eval_one_epoch(model: nn.Module,
                  eval_loader: DataLoader,
                  device: torch.device,
                  lambda_flux: float = 0.01) -> Dict[str, float]:
    """
    Evaluate for one epoch, using LR input and HR target
    
    Args:
        model: Model to evaluate
        eval_loader: Evaluation data loader
        device: Device to evaluate on
        lambda_flux: Weight for flux loss term
    Returns:
        Dictionary with loss values
    """
    model.eval()
    
    total_loss = 0.0
    total_loss_recon = 0.0
    total_loss_flux = 0.0
    num_batches = 0
    
    progress_bar = tqdm.tqdm(eval_loader, desc="Evaluating")
    
    for batch in progress_bar:
        lr_images = batch["lr_image"].to(device)
        hr_images = batch["hr_image"].to(device)
        
        # Forward pass on LR images
        reconstructed, aux_outputs = model(lr_images)
        
        # Compute loss
        loss, loss_dict = model.get_loss(
            inputs=hr_images,
            reconstructed=reconstructed,
            aux_outputs=aux_outputs,
            flux_map_target=None,
            lambda_flux=lambda_flux
        )
        
        total_loss += loss_dict['loss_total']
        total_loss_recon += loss_dict['loss_recon']
        total_loss_flux += loss_dict['loss_flux']
        num_batches += 1
    
    return {
        'loss_total': total_loss / num_batches,
        'loss_recon': total_loss_recon / num_batches,
        'loss_flux': total_loss_flux / num_batches,
    }
