"""
Documentation page for Enhanced SRCNN (ESRCNN)
Comprehensive technical documentation and training guide
"""
import streamlit as st


def render_documentation_page(config):
    """Render documentation page for ESRCNN"""
    
    st.markdown('<div class="section-header">Model Specifications</div>', unsafe_allow_html=True)
    
    # Training Configuration
    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Model Architecture**")
            st.code(f"""
Scale Factor: {config.scale_factor}×
Patch Size: {config.crop_size}×{config.crop_size}
Feature Channels: {config.num_features}
Residual Blocks: {config.num_residual_blocks}

Architecture Flow:
  1. Feature Extraction
     Conv(3→{config.num_features}, 3×3) + ReLU
  
  2. Residual Blocks (×{config.num_residual_blocks})
     Each block:
       Conv({config.num_features}→{config.num_features}, 3×3) + BN + ReLU
       Conv({config.num_features}→{config.num_features}, 3×3) + BN
       Skip Connection + ReLU
  
  3. Bottleneck
     Conv({config.num_features}→{config.num_features}, 3×3) + BN
     + Skip from feature extraction
  
  4. Upsampling ({config.scale_factor}×)
     Sub-Pixel Convolution
  
  5. Reconstruction
     Conv({config.num_features}→3, 3×3)
  
  6. Global Skip
     + Bicubic_Upsample(Input)

Total Parameters: ~930K
Receptive Field: ~50×50 pixels
            """)
        
        with col2:
            st.markdown("**Training Hyperparameters**")
            st.code(f"""
Optimizer: {config.optimizer.upper()}
  learning_rate: {config.learning_rate:.2e}
  weight_decay: {config.weight_decay:.2e}
  betas: (0.9, 0.999)
  
Batch Size: {config.batch_size}
  (Smaller due to larger model)

LR Scheduler: {config.lr_scheduler}
  milestones: [50, 100, 130]
  decay_factor: 0.5
  
Training:
  num_epochs: {config.num_epochs}
  validation_interval: {config.val_interval}
  log_interval: 10 batches
  
Advanced:
  mixed_precision: {config.mixed_precision}
  gradient_clipping: Optional
  
Convergence:
  Expected: 100-150 epochs
  Early stopping: Monitor val_loss
            """)
        
        with col3:
            st.markdown("**Loss Configuration**")
            st.code(f"""
Multi-Component Loss:

1. Pixel Loss ({config.pixel_loss_type.upper()})
   Weight: {config.loss_pixel_weight}
   {'L1 (Charbonnier)' if config.pixel_loss_type == 'l1' else 'MSE'}
   
2. Perceptual Loss (VGG19)
   Weight: {config.loss_perceptual_weight}
   Layers: [relu1_2, relu2_2, 
            relu3_4, relu4_4]
   Enabled: {config.use_perceptual_loss}
   
3. SSIM Loss
   Weight: {config.loss_ssim_weight}
   Window: 11×11
   
Total Loss:
L_total = {config.loss_pixel_weight} × L_pixel
        + {config.loss_perceptual_weight} × L_perceptual
        + {config.loss_ssim_weight} × L_SSIM

Why Multiple Losses?
✓ Pixel: Accuracy
✓ Perceptual: Realism
✓ SSIM: Structure
            """)
    
    st.markdown('<div class="section-header">Loss Functions</div>', unsafe_allow_html=True)
    
    # Pixel Loss
    with st.expander("📊 1. Pixel-wise Loss (L1/Charbonnier)", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Type:** {config.pixel_loss_type.upper()} Loss (Charbonnier variant)
            **Weight:** {config.loss_pixel_weight}
            
            **Mathematical Definition:**
            $$L_{{pixel}} = \\frac{{1}}{{N}} \\sum_{{i}} \\sqrt{{(\\hat{{y}}_i - y_i)^2 + \\epsilon^2}}$$
            
            **Where:**
            - $\\hat{{y}}$ = predicted HR image
            - $y$ = ground truth HR image
            - $\\epsilon$ = 1e-3 (for smoothness)
            - $N$ = number of pixels
            
            **Why L1 over MSE?**
            - Better gradient behavior
            - Less penalty for large errors
            - Preserves sharp edges
            - More robust to outliers
            
            **Charbonnier vs Standard L1:**
            - Smoothed near zero
            - Better convergence
            - Differentiable everywhere
            """)
        
        with col2:
            st.code(f"""
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(
            diff ** 2 + self.eps ** 2
        )
        return loss.mean()

# Usage
pixel_loss = CharbonnierLoss()
L_pixel = pixel_loss(pred, target)
            """, language="python")
    
    # Perceptual Loss
    with st.expander(f"🎨 2. Perceptual Loss (VGG19) - Weight: {config.loss_perceptual_weight}", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Type:** Feature-based perceptual loss using VGG19
            **Weight:** {config.loss_perceptual_weight}
            **Enabled:** {config.use_perceptual_loss}
            
            **Mathematical Definition:**
            $$L_{{perceptual}} = \\sum_{{l \\in L}} \\frac{{1}}{{N_l}} ||\\phi_l(\\hat{{y}}) - \\phi_l(y)||_1$$
            
            **Where:**
            - $\\phi_l$ = VGG19 features at layer $l$
            - $L$ = {{relu1_2, relu2_2, relu3_4, relu4_4}}
            - $N_l$ = number of features at layer $l$
            
            **Why Perceptual Loss is CRUCIAL for Faces:**
            
            🔴 **Without Perceptual Loss (MSE only):**
            - Model minimizes pixel-wise error
            - Averages out high-frequency details
            - Result: Blurry faces, soft edges
            - Good PSNR but poor visual quality
            
            🟢 **With Perceptual Loss:**
            - Matches high-level semantic features
            - Preserves facial structure (eyes, nose, mouth)
            - Sharper, more realistic outputs
            - Better perceptual quality
            
            **VGG19 Layers Used:**
            - `relu1_2`: Low-level edges, textures
            - `relu2_2`: Simple patterns
            - `relu3_4`: Mid-level features
            - `relu4_4`: High-level semantic features
            
            **Trade-off:**
            - Slightly lower PSNR
            - Much better perceptual quality
            - Essential for face super-resolution
            """)
        
        with col2:
            st.code(f"""
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True)
        
        # Extract feature layers
        self.layers = [3, 8, 17, 26]
        # relu1_2, relu2_2, 
        # relu3_4, relu4_4
        
        # Build extractors
        self.extractors = []
        for i in self.layers:
            self.extractors.append(
                nn.Sequential(
                    *list(vgg.features[:i+1])
                )
            )
        
        # Freeze VGG
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, pred, target):
        loss = 0.0
        for extractor in self.extractors:
            pred_feat = extractor(pred)
            target_feat = extractor(target)
            loss += F.l1_loss(
                pred_feat, 
                target_feat
            )
        return loss / len(self.extractors)
            """, language="python")
    
    # SSIM Loss
    with st.expander(f"📐 3. SSIM Loss (Structural Similarity) - Weight: {config.loss_ssim_weight}", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Type:** Structural Similarity Index Measure
            **Weight:** {config.loss_ssim_weight}
            
            **Mathematical Definition:**
            $$SSIM(x,y) = \\frac{{(2\\mu_x\\mu_y + c_1)(2\\sigma_{{xy}} + c_2)}}{{(\\mu_x^2 + \\mu_y^2 + c_1)(\\sigma_x^2 + \\sigma_y^2 + c_2)}}$$
            
            $$L_{{SSIM}} = 1 - SSIM(\\hat{{y}}, y)$$
            
            **Where:**
            - $\\mu_x, \\mu_y$ = local means
            - $\\sigma_x, \\sigma_y$ = local standard deviations
            - $\\sigma_{{xy}}$ = local covariance
            - $c_1, c_2$ = stability constants
            
            **Components:**
            1. **Luminance:** $l(x,y) = \\frac{{2\\mu_x\\mu_y + c_1}}{{\\mu_x^2 + \\mu_y^2 + c_1}}$
            2. **Contrast:** $c(x,y) = \\frac{{2\\sigma_x\\sigma_y + c_2}}{{\\sigma_x^2 + \\sigma_y^2 + c_2}}$
            3. **Structure:** $s(x,y) = \\frac{{\\sigma_{{xy}} + c_2/2}}{{\\sigma_x\\sigma_y + c_2/2}}$
            
            **Why SSIM?**
            - Measures **perceived** quality
            - More aligned with human vision
            - Captures structural information
            - Complements pixel-wise loss
            
            **Optional Usage:**
            - Weight = 0.0: Disabled (faster training)
            - Weight = 0.3-0.5: Enhanced structural preservation
            """)
        
        with col2:
            st.code(f"""
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        
        # Create Gaussian kernel
        sigma = 1.5
        kernel = self._gaussian_kernel(
            window_size, sigma
        )
        self.register_buffer(
            'kernel', kernel
        )
    
    def forward(self, pred, target):
        # Compute local means
        mu1 = F.conv2d(
            pred, self.kernel, 
            padding=self.window_size//2
        )
        mu2 = F.conv2d(
            target, self.kernel,
            padding=self.window_size//2
        )
        
        # ... compute SSIM
        
        ssim_map = (
            (2*mu1*mu2 + c1) * 
            (2*sigma12 + c2)
        ) / (
            (mu1**2 + mu2**2 + c1) * 
            (sigma1**2 + sigma2**2 + c2)
        )
        
        return 1 - ssim_map.mean()
            """, language="python")
    
    st.markdown('<div class="section-header">Training Guide</div>', unsafe_allow_html=True)
    
    # Quick Start
    with st.expander("🚀 Quick Start Training", expanded=True):
        st.markdown(f"""
        ### Training Command
        
        **Basic Training (Recommended):**
        ```bash
        uv run training/train_esrcnn.py \\
          --data-dir dataset/ffhq \\
          --scale-factor {config.scale_factor} \\
          --batch-size {config.batch_size} \\
          --num-epochs {config.num_epochs} \\
          --use-perceptual-loss \\
          --perceptual-weight {config.loss_perceptual_weight}
        ```
        
        **Quick Test (Faster, Lower Quality):**
        ```bash
        uv run training/train_esrcnn.py \\
          --data-dir dataset/ffhq \\
          --scale-factor {config.scale_factor} \\
          --num-residual-blocks 6 \\
          --batch-size 32 \\
          --num-epochs 50
        ```
        
        **High Quality (Slower, Best Quality):**
        ```bash
        uv run training/train_esrcnn.py \\
          --data-dir dataset/ffhq \\
          --scale-factor {config.scale_factor} \\
          --num-residual-blocks 16 \\
          --num-features {config.num_features} \\
          --batch-size 8 \\
          --num-epochs 200 \\
          --use-perceptual-loss \\
          --perceptual-weight {config.loss_perceptual_weight} \\
          --mixed-precision
        ```
        """)
    
    # Expected Results
    with st.expander("📈 Expected Results", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Classic SRCNN (Baseline)
            
            **Training:**
            - Time: 2-3 hours
            - GPU Memory: ~2GB
            - Convergence: 50-100 epochs
            
            **Results:**
            - PSNR: 24-26 dB
            - SSIM: 0.75-0.80
            - Quality: ⭐⭐ Blurry faces
            - Speed: Very fast inference
            
            **Use Case:**
            - Quick baseline
            - Speed-critical applications
            - Simple images (not faces)
            """)
        
        with col2:
            st.markdown(f"""
            ### Enhanced SRCNN (ESRCNN)
            
            **Training:**
            - Time: 8-12 hours
            - GPU Memory: ~6GB
            - Convergence: 100-150 epochs
            
            **Results:**
            - PSNR: 28-32 dB (+4-6 dB improvement!)
            - SSIM: 0.85-0.92
            - Quality: ⭐⭐⭐⭐⭐ Sharp, realistic faces
            - Speed: Fast inference (~10-20ms)
            
            **Use Case:**
            - Face super-resolution ⭐
            - Quality-critical applications
            - Production deployments
            """)
    
    # Hyperparameter Tuning
    with st.expander("🎛️ Hyperparameter Tuning Guide", expanded=True):
        st.markdown(f"""
        ### Key Hyperparameters to Tune
        
        | Parameter | Default | Range | Effect |
        |-----------|---------|-------|--------|
        | `num_residual_blocks` | {config.num_residual_blocks} | 6-16 | More blocks = better quality but slower |
        | `num_features` | {config.num_features} | 32-128 | More features = more capacity |
        | `learning_rate` | {config.learning_rate:.2e} | 5e-5 to 2e-4 | Lower = stable, Higher = faster |
        | `batch_size` | {config.batch_size} | 8-32 | Depends on GPU memory |
        | `perceptual_weight` | {config.loss_perceptual_weight} | 0.05-0.2 | Balance pixel accuracy vs perceptual |
        | `crop_size` | {config.crop_size} | 32-64 | Larger = more context |
        
        ### Tuning Strategies
        
        **For Better Quality:**
        - Increase `num_residual_blocks` to 12-16
        - Increase `num_features` to 128
        - Increase `perceptual_weight` to 0.15-0.2
        - Train longer (200+ epochs)
        
        **For Faster Training:**
        - Decrease `num_residual_blocks` to 6-8
        - Decrease `num_features` to 32
        - Increase `batch_size` to 32
        - Train for 50-80 epochs
        
        **For GPU Memory Constraints:**
        - Decrease `batch_size`
        - Decrease `crop_size`
        - Use `mixed_precision=True`
        - Gradient accumulation
        """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting", expanded=True):
        st.markdown("""
        ### Common Issues & Solutions
        
        **1. Training Loss Not Decreasing**
        - **Symptom:** Loss stays flat or increases
        - **Solutions:**
          - Lower learning rate (try 5e-5)
          - Check data normalization
          - Verify loss weights are reasonable
          - Ensure perceptual loss is initialized correctly
        
        **2. Blurry Outputs Despite Training**
        - **Symptom:** High PSNR but blurry faces
        - **Solutions:**
          - Enable perceptual loss
          - Increase perceptual_weight to 0.15-0.2
          - Add SSIM loss (weight=0.3)
          - Check that VGG19 is loaded correctly
        
        **3. Out of Memory (OOM)**
        - **Symptom:** CUDA OOM error
        - **Solutions:**
          - Reduce batch_size (try 8 or 4)
          - Reduce crop_size to 32
          - Enable mixed_precision
          - Reduce num_residual_blocks
        
        **4. Slow Training**
        - **Symptom:** Very slow epoch times
        - **Solutions:**
          - Enable mixed_precision (2× speedup)
          - Increase num_workers for data loading
          - Check GPU utilization
          - Consider reducing model size
        
        **5. Overfitting**
        - **Symptom:** Train loss << val loss
        - **Solutions:**
          - Increase weight_decay
          - Add data augmentation
          - Reduce model capacity
          - Use more training data
        
        **6. Perceptual Loss Error**
        - **Symptom:** VGG19 not loading
        - **Solutions:**
          - Install torchvision: `pip install torchvision`
          - Check internet connection (downloads pretrained weights)
          - Disable perceptual loss temporarily for testing
        """)
    
    st.success("""
    **💡 Pro Tip:** Start with the default configuration and tune based on your specific dataset and requirements. 
    Monitor both quantitative metrics (PSNR, SSIM) and visual quality on validation samples!
    """)
