"""
Physics-Informed Masked Autoencoders (MAE) Vision Transformer
Complete architecture combining preprocessing, masking, flux guidance, and hybrid decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from training.train_utils.fisr.modules import (
    PhysicsInformedPreprocessing,
    MaskedPatchEmbedding,
    FluxGuidanceGeneration,
    RotaryPositionalEmbedding,
    TransformerBlock,
    FluxGuidanceController,
    PixelShuffleUpsample,
    CNNRefinementHead
)


class PhysicsInformedMAE(nn.Module):
    """
    Physics-Informed Masked Autoencoders (MAE) for Astronomical Image Reconstruction
    
    Architecture:
    1. Physics-Informed Preprocessing: Double differentiation + Tanh normalization
    2. Asymmetric Masking: 75-90% masking ratio, process visible tokens only
    3. Flux Guidance Generation: Pre-computed flux maps from object detection
    4. ViT Encoder: Transformer blocks with flux guidance controller
    5. Hybrid Decoder: Transformer + CNN refinement with PixelShuffle
    """
    
    def __init__(self,
                 img_size: int = 64,
                 patch_size: int = 4,
                 in_channels: int = 1,
                 embed_dim: int = 768,
                 encoder_depth: int = 12,
                 decoder_depth: int = 8,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 mask_ratio: float = 0.75,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.1,
                 scale_factor: int = 1,
                 sr_mode: bool = False):
        super().__init__()
        
        # Configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.scale_factor = scale_factor
        self.sr_mode = sr_mode
        
        # SR mode configuration
        if sr_mode and scale_factor > 1:
            import math
            self.num_upsamples = int(math.log2(scale_factor))
        else:
            self.num_upsamples = 0
        
        # ============ ENCODER COMPONENTS ============
        
        # Physics-informed preprocessing
        self.preprocessing = PhysicsInformedPreprocessing(in_channels)
        
        # Masked patch embedding
        self.patch_embed = MaskedPatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            mask_ratio=mask_ratio
        )
        
        # Learnable class token for encoder
        self.cls_token_encoder = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (for visible tokens only)
        self.pos_embed = RotaryPositionalEmbedding(embed_dim, max_seq_len=self.num_patches + 1)
        
        # Encoder blocks
        dpr = [drop_path * (i / encoder_depth) for i in range(encoder_depth)]
        self.encoder = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                drop_path=dpr[i]
            )
            for i in range(encoder_depth)
        ])
        
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # ============ FLUX GUIDANCE ============
        
        # Flux Guidance Generation module
        self.fgg = FluxGuidanceGeneration(
            img_size=img_size,
            patch_size=patch_size,
            num_scales=4
        )
        
        # Flux Guidance Controller (integrated into encoder)
        self.fgc = FluxGuidanceController(embed_dim, num_scales=4)
        
        # ============ DECODER COMPONENTS ============
        
        # Learnable mask tokens for decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Decoder positional embeddings
        self.decoder_pos_embed = RotaryPositionalEmbedding(embed_dim, max_seq_len=self.num_patches + 1)
        
        # Decoder blocks
        dpr_decoder = [drop_path * (i / decoder_depth) for i in range(decoder_depth)]
        self.decoder = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                drop_path=dpr_decoder[i]
            )
            for i in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(embed_dim)
        
        # Transformer decoder for global context
        self.transformer_decoder = nn.Sequential(*self.decoder)
        
        # CNN refinement head
        self.cnn_refine = CNNRefinementHead(
            in_channels=embed_dim,
            hidden_channels=256,
            out_channels=patch_size * patch_size
        )
        
        # Upsampling layers (num_upsamples determined by scale_factor in SR mode)
        self.upsample_layers = nn.ModuleList([
            PixelShuffleUpsample(embed_dim, embed_dim, upscale_factor=2)
            for _ in range(self.num_upsamples)
        ])
        
        # Output projection: from patch embedding to pixel values
        self.output_projection = nn.Linear(embed_dim, patch_size * patch_size)
        
        # ============ INITIALIZATION ============
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.cls_token_encoder, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
    
    def forward(self,
                x: torch.Tensor,
                flux_weights: Optional[torch.Tensor] = None,
                sr_mode: Optional[bool] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with optional Super-Resolution mode
        
        Args:
            x: Input image (B, C, H, W)
            flux_weights: Pre-calculated flux weights (B, num_patches) [optional]
            sr_mode: If True, perform SR upsampling (default: use self.sr_mode)
        Returns:
            reconstructed: Reconstructed image
                - If sr_mode=False: (B, C, H, W) same size as input
                - If sr_mode=True: (B, C, H*scale, W*scale) HR upsampled
            aux_outputs: Dict containing:
                - mask: Binary mask (B, num_patches)
                - edge_map: Edge map from preprocessing
                - flux_map: Generated flux guidance map
                - latent: Encoded latent (B, N, embed_dim)
                - patches_reconstructed: Patch representations
        """
        if sr_mode is None:
            sr_mode = self.sr_mode
        
        B, C, H, W = x.shape
        
        # ============ PREPROCESSING ============
        x_norm, edge_map = self.preprocessing(x)
        
        # ============ FLUX GUIDANCE GENERATION ============
        flux_map = self.fgg(edge_map, flux_weights)
        
        # ============ MASKING & PATCH EMBEDDING ============
        visible_patches, mask, ids_restore = self.patch_embed(x_norm)
        
        # Prepend class token
        cls_token = self.cls_token_encoder.expand(B, -1, -1)
        encoder_input = torch.cat([cls_token, visible_patches], dim=1)  # (B, N_visible+1, embed_dim)
        
        # Add positional embeddings
        encoder_input = self.pos_embed(encoder_input)
        
        # ============ ENCODER ============
        encoded = encoder_input
        for block in self.encoder:
            # Apply flux guidance control at each scale
            if block == self.encoder[0]:  # First block gets the control signal
                non_cls = encoded[:, 1:, :]  # Exclude CLS token
                non_cls = self.fgc(non_cls, flux_map)
                encoded = torch.cat([encoded[:, :1, :], non_cls], dim=1)
            
            encoded = block(encoded)
        
        encoded = self.encoder_norm(encoded)
        
        # ============ DECODER ============
        
        # Expand encoded features to all patches
        x_decode = encoded[:, 1:, :]  # Remove class token (B, N_visible, embed_dim)
        
        # Create mask tokens for masked patches
        num_masked = self.num_patches - x_decode.shape[1]
        mask_tokens = self.mask_token.expand(B, num_masked, -1)  # (B, N_masked, embed_dim)
        
        # Combine visible and mask tokens in original order
        x_decode = torch.cat([x_decode, mask_tokens], dim=1)  # (B, num_patches, embed_dim)
        x_decode = torch.gather(
            x_decode,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )  # Restore original order
        
        # Add decoder positional embeddings
        x_decode = self.decoder_pos_embed(x_decode)
        
        # Transformer decoder
        for block in self.decoder:
            x_decode = block(x_decode)
        
        x_decode = self.decoder_norm(x_decode)
        
        # ============ RECONSTRUCTION ============
        
        if not sr_mode:
            # ===== LEGACY MODE: LR Reconstruction =====
            # Project patches back to pixel values
            patches_reconstructed = self.output_projection(x_decode)  # (B, num_patches, patch_size^2)
            
            # Reshape to image (same size as input)
            reconstructed = patches_reconstructed.view(
                B,
                H // self.patch_size,
                W // self.patch_size,
                self.patch_size,
                self.patch_size
            )
            reconstructed = reconstructed.permute(0, 3, 4, 1, 2).contiguous()
            reconstructed = reconstructed.view(B, C, H, W)
        
        else:
            # ===== SR MODE: HR Upsampling Reconstruction =====
            # Use standard reconstruction output then upscale
            
            # Project patches back to pixel values
            patches_reconstructed = self.output_projection(x_decode)  # (B, num_patches, patch_size^2)
            
            # Reshape to image (LR size first)
            reconstructed = patches_reconstructed.view(
                B,
                H // self.patch_size,
                W // self.patch_size,
                self.patch_size,
                self.patch_size
            )
            reconstructed = reconstructed.permute(0, 3, 4, 1, 2).contiguous()
            reconstructed = reconstructed.view(B, C, H, W)
            
            # Upscale to HR using bilinear interpolation
            reconstructed = F.interpolate(
                reconstructed,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=False
            )
            # reconstructed is now (B, C, H*scale, W*scale)
        
        # ============ AUX OUTPUTS ============
        aux_outputs = {
            'mask': mask,
            'edge_map': edge_map,
            'flux_map': flux_map,
            'latent': encoded,
            'patches_reconstructed': patches_reconstructed,
        }
        
        return reconstructed, aux_outputs
    
    def get_flux_loss(self,
                      aux_outputs: Dict,
                      flux_map_target: Optional[torch.Tensor] = None,
                      mode: str = "sparsity") -> torch.Tensor:
        """
        Compute flux loss with configurable modes
        
        Args:
            aux_outputs: Auxiliary outputs from forward pass
            flux_map_target: Target flux map (for "target" mode)
            mode: Loss mode - "sparsity" (regularize to sparse) or "target" (consistency loss)
        Returns:
            loss_flux: Scalar loss tensor
        """
        flux_map = aux_outputs['flux_map']  # (B, 1, H_p, W_p)
        flux_weights = flux_map.squeeze(1).flatten(1)  # (B, num_patches)
        
        if mode == "target" and flux_map_target is not None:
            # Flux consistency loss against target
            loss_flux = torch.mean(
                flux_weights * torch.abs(flux_map - flux_map_target)
            )
        else:
            # Default: sparsity regularization (encourage concentration)
            loss_flux = torch.mean(
                torch.sum(flux_weights ** 2, dim=1)
            )
        
        return loss_flux
    
    def get_loss(self,
                 inputs: torch.Tensor,
                 reconstructed: torch.Tensor,
                 aux_outputs: Dict,
                 flux_map_target: Optional[torch.Tensor] = None,
                 lambda_flux: float = 0.01) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate total loss for legacy mode: L_recon + λ * L_flux
        
        Args:
            inputs: Original input images (B, C, H, W)
            reconstructed: Reconstructed images (B, C, H, W)
            aux_outputs: Auxiliary outputs from forward pass
            flux_map_target: Target flux maps for flux consistency [optional]
            lambda_flux: Weight for flux loss term
        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with individual loss components
        """
        # ============ RECONSTRUCTION LOSS ============
        # MSE computed only on masked patches (true MAE)
        
        mask = aux_outputs['mask']  # (B, num_patches)
        B, C, H, W = inputs.shape
        
        # Compute per-patch MSE
        input_patches = inputs.view(B, C, H // self.patch_size, self.patch_size,
                                    W // self.patch_size, self.patch_size)
        input_patches = input_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        input_patches = input_patches.view(B, -1, C * self.patch_size * self.patch_size)
        
        reconstructed_patches = aux_outputs['patches_reconstructed']  # (B, num_patches, patch_size^2)
        
        # L_recon: MSE only on masked regions (true MAE reconstruction)
        # Apply mask: 1 for masked patches, 0 for visible
        mask_expanded = mask.unsqueeze(-1).float()  # (B, num_patches, 1)
        masked_loss = ((reconstructed_patches - input_patches) ** 2) * mask_expanded
        loss_recon = masked_loss.sum() / (mask.sum().float() + 1e-8)  # Normalize by number of masked patches
        
        # ============ FLUX CONSISTENCY LOSS ============
        loss_flux = self.get_flux_loss(
            aux_outputs=aux_outputs,
            flux_map_target=flux_map_target,
            mode="sparsity" if flux_map_target is None else "target"
        )
        
        # ============ TOTAL LOSS ============
        total_loss = loss_recon + lambda_flux * loss_flux
        
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_recon': loss_recon.item(),
            'loss_flux': loss_flux.item(),
        }
        
        return total_loss, loss_dict


def create_physics_informed_mae(img_size: int = 64, **kwargs) -> PhysicsInformedMAE:
    """
    Factory function to create Physics-Informed MAE
    
    Supports both legacy and SR mode depending on kwargs
    """
    defaults = {
        "img_size": img_size,
        "patch_size": 4,
        "in_channels": 1,
        "embed_dim": 768,
        "encoder_depth": 12,
        "decoder_depth": 8,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "mask_ratio": 0.75,
        "scale_factor": 1,      # NEW: Default to no upsampling (legacy mode)
        "sr_mode": False,       # NEW: Default to legacy mode
    }
    defaults.update(kwargs)  # user-provided args override defaults
    return PhysicsInformedMAE(**defaults)
