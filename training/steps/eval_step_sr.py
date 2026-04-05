"""
Super-Resolution Evaluation Step
Evaluates model on validation set with same loss composition as training
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from typing import Dict

from training.train_utils.losses_sr import (
    CharbonnierLoss,
    SSIMLoss,
    FFTLoss,
    MaskedReconstructionLoss,
    compute_multiresolution_losses
)
from training.core.config_sr import SRTrainingConfig


@torch.no_grad()
def eval_one_epoch_sr(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    config: SRTrainingConfig
) -> Dict[str, float]:
    """
    Evaluate for one epoch in Super-Resolution mode
    
    Input: LR images (B, C, H, W)
    Target: HR images (B, C, H*scale, W*scale)
    
    Args:
        model: Model to evaluate (with sr_mode=True)
        eval_loader: Evaluation data loader with 'lr_image' and 'hr_image' batches
        device: Torch device
        config: SRTrainingConfig instance
    Returns:
        Dictionary with loss values
    """
    model.eval()
    
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
    
    progress_bar = tqdm.tqdm(eval_loader, desc="Evaluating (SR)")
    
    for batch in progress_bar:
        # Load batch
        lr_images = batch["lr_image"].to(device)
        hr_images = batch["hr_image"].to(device)
        
        # Forward pass in SR mode
        reconstructed, aux_outputs = model(lr_images, sr_mode=True)
        
        # ============ COMPUTE LOSSES ============
        
        # 1. Masked Reconstruction Loss
        loss_recon = masked_recon_loss_fn(
            pred=reconstructed,
            target=hr_images,
            mask=aux_outputs['mask'],
            patch_size=config.patch_size
        )
        
        # 2. L1 Loss
        loss_l1 = l1_loss_fn(reconstructed, hr_images)
        
        # 3. SSIM Loss
        loss_ssim = ssim_loss_fn(reconstructed, hr_images, data_range=1.0)
        
        # 4. FFT Loss
        loss_fft = fft_loss_fn(reconstructed, hr_images)
        
        # 5. Flux Loss
        if config.flux_loss_mode == "target":
            flux_map_target = aux_outputs.get('flux_map_target', None)
            loss_flux = model.get_flux_loss(
                aux_outputs=aux_outputs,
                flux_map_target=flux_map_target,
                mode="target"
            )
        else:
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
        if config.enable_multiscale:
            multiscale_losses = compute_multiresolution_losses(
                pred=reconstructed,
                target=hr_images,
                scales=[1, 2, 4]
            )
            
            for scale_idx, (loss_key, loss_val) in enumerate(multiscale_losses.items()):
                weight = config.multiscale_weights[scale_idx] if scale_idx < len(config.multiscale_weights) else 1.0
                total_weighted_loss += weight * loss_val
        
        # ============ LOSS TRACKING ============
        total_loss += total_weighted_loss.item()
        total_loss_recon += loss_recon.item()
        total_loss_l1 += loss_l1.item()
        total_loss_ssim += loss_ssim.item()
        total_loss_fft += loss_fft.item()
        total_loss_flux += loss_flux.item()
        num_batches += 1
    
    # Return averaged losses
    return {
        'loss_total': total_loss / num_batches,
        'loss_recon': total_loss_recon / num_batches,
        'loss_l1': total_loss_l1 / num_batches,
        'loss_ssim': total_loss_ssim / num_batches,
        'loss_fft': total_loss_fft / num_batches,
        'loss_flux': total_loss_flux / num_batches,
    }
