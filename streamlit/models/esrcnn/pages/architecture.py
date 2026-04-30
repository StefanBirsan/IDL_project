"""
Architecture documentation page for Enhanced SRCNN (ESRCNN)
"""
import streamlit as st


def render_architecture_page(config):
    """Render architecture page for ESRCNN"""
    
    st.markdown('<div class="section-header">Enhanced SRCNN Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ### Deep Residual Super-Resolution Pipeline
        
        ESRCNN is a **significantly enhanced** version of classic SRCNN with:
        
        1. **Feature Extraction** → Deep feature learning (Conv + ReLU)
        2. **Residual Blocks (×10)** → Non-linear mapping with skip connections
        3. **Bottleneck Layer** → Feature refinement + global skip
        4. **Sub-Pixel Upsampling** → Efficient learned upscaling
        5. **Reconstruction** → Final RGB output
        6. **Global Skip Connection** → Enhanced gradient flow
        
        **Key Advantages:**
        - 🔥 **13× more parameters** than classic SRCNN
        - 🎯 **3× larger receptive field** (50×50 vs 17×17 pixels)
        - 💎 **Perceptual loss** for realistic facial features
        - ⚡ **Sub-pixel convolution** for efficient upsampling
        """)
    
    with col2:
        st.info(f"""
        **Model Configuration**
        - Scale Factor: {config.scale_factor}×
        - Feature Channels: {config.num_features}
        - Residual Blocks: {config.num_residual_blocks}
        - Patch Size: {config.crop_size}×{config.crop_size}
        - Parameters: ~900K (vs ~70K in SRCNN)
        - Receptive Field: ~50×50 pixels
        - Loss: {config.pixel_loss_type.upper()} + Perceptual + SSIM
        """)
    
    st.markdown('<div class="section-header">Architecture Components</div>', unsafe_allow_html=True)
    
    # Feature Extraction
    with st.expander("🎯 1. Feature Extraction Layer", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Extract low-level features from LR input
            
            **Configuration:**
            - **Kernel Size:** 3×3 (smaller, more efficient)
            - **Input Channels:** 3 (RGB)
            - **Output Channels:** {config.num_features}
            - **Padding:** 1 (maintains spatial size)
            - **Activation:** ReLU
            
            **Key Difference from SRCNN:**
            - No bicubic pre-upsampling required
            - Works directly on native LR image
            - More efficient pipeline
            """)
        
        with col2:
            st.code(f"""
import torch.nn as nn

# Feature Extraction
self.feature_extraction = nn.Sequential(
    nn.Conv2d(
        in_channels=3,
        out_channels={config.num_features},
        kernel_size=3,
        padding=1
    ),
    nn.ReLU(inplace=True)
)

# Input: (B, 3, H, W)
# Output: (B, {config.num_features}, H, W)
            """, language="python")
    
    # Residual Blocks
    with st.expander(f"🔁 2. Residual Blocks (×{config.num_residual_blocks})", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Deep non-linear feature mapping
            
            **Configuration:**
            - **Number of Blocks:** {config.num_residual_blocks}
            - **Channels:** {config.num_features}
            - **Kernel Size:** 3×3
            - **Normalization:** Batch Normalization
            - **Activation:** ReLU
            - **Skip Connection:** Identity mapping
            
            **Why Residual Blocks?**
            - ✅ Enables **deep networks** (10+ layers)
            - ✅ Better **gradient flow** during training
            - ✅ Learns **residual features** (easier optimization)
            - ✅ Prevents **vanishing gradients**
            
            **Formula:**
            ```
            y = F(x) + x
            ```
            Where F(x) is Conv-BN-ReLU-Conv-BN
            """)
        
        with col2:
            st.code(f"""
class ResidualBlock(nn.Module):
    def __init__(self, channels={config.num_features}):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x  # Save input
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity  # Skip connection
        out = self.relu(out)
        
        return out

# Stack {config.num_residual_blocks} blocks
self.residual_blocks = nn.Sequential(
    *[ResidualBlock({config.num_features}) 
      for _ in range({config.num_residual_blocks})]
)
            """, language="python")
    
    # Bottleneck
    with st.expander("🔧 3. Bottleneck Layer", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Feature refinement and aggregation
            
            **Configuration:**
            - **Conv:** {config.num_features}→{config.num_features}, 3×3
            - **Normalization:** Batch Normalization
            - **Skip Connection:** From feature extraction
            
            **Why Bottleneck?**
            - Aggregates features from all residual blocks
            - Additional skip connection improves gradient flow
            - Refines features before upsampling
            """)
        
        with col2:
            st.code(f"""
# Bottleneck
self.bottleneck = nn.Sequential(
    nn.Conv2d({config.num_features}, {config.num_features}, 3, padding=1),
    nn.BatchNorm2d({config.num_features})
)

# Forward with skip
features = self.feature_extraction(x)
residual_out = self.residual_blocks(features)
bottleneck_out = self.bottleneck(residual_out)

# Skip from feature extraction
out = bottleneck_out + features
            """, language="python")
    
    # Upsampling
    with st.expander(f"⬆️ 4. Sub-Pixel Upsampling ({config.scale_factor}×)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Efficient learned upsampling
            
            **Configuration:**
            - **Method:** Sub-pixel convolution (PixelShuffle)
            - **Scale Factor:** {config.scale_factor}×
            - **Advantage:** Learned vs interpolation-based
            
            **How Sub-Pixel Convolution Works:**
            1. Conv increases channels by scale²
            2. PixelShuffle rearranges to spatial dimensions
            3. More efficient than deconvolution
            
            **Example for 2× upsampling:**
            - Conv: 64 → 256 channels (4×)
            - PixelShuffle: Rearrange 256 to 2×2 spatial
            - Result: Height×2, Width×2, 64 channels
            
            **Formula:**
            ```
            H_out = H_in × scale
            W_out = W_in × scale
            C_out = C_in
            ```
            """)
        
        with col2:
            st.code(f"""
class UpsampleBlock(nn.Module):
    def __init__(self, channels={config.num_features}, scale=2):
        super().__init__()
        # Increase channels by scale²
        self.conv = nn.Conv2d(
            channels,
            channels * (scale ** 2),  # {config.num_features} → {config.num_features * 4}
            kernel_size=3,
            padding=1
        )
        # Rearrange to spatial dimensions
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: (B, {config.num_features}, H, W)
        x = self.conv(x)
        # x: (B, {config.num_features * 4}, H, W)
        x = self.pixel_shuffle(x)
        # x: (B, {config.num_features}, H×2, W×2)
        x = self.relu(x)
        return x

# For {config.scale_factor}× upsampling:
upsampling = UpsampleBlock({config.num_features}, scale={config.scale_factor})
            """, language="python")
    
    # Reconstruction
    with st.expander("🎨 5. Reconstruction Layer", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Generate final RGB output
            
            **Configuration:**
            - **Kernel Size:** 3×3
            - **Input Channels:** {config.num_features}
            - **Output Channels:** 3 (RGB)
            - **Activation:** None (linear)
            
            **Output Processing:**
            - No activation allows full dynamic range
            - Values can exceed [0, 1] before clipping
            - More flexibility for high-quality reconstruction
            """)
        
        with col2:
            st.code(f"""
# Reconstruction
self.reconstruction = nn.Conv2d(
    in_channels={config.num_features},
    out_channels=3,
    kernel_size=3,
    padding=1
)

# No activation - linear output
output = self.reconstruction(upsampled)

# Output: (B, 3, H×{config.scale_factor}, W×{config.scale_factor})
            """, language="python")
    
    # Global Skip Connection
    with st.expander("🔗 6. Global Skip Connection", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Enhance gradient flow and convergence
            
            **Configuration:**
            - **Type:** Bicubic upsampled input added to output
            - **Enabled:** {config.use_global_skip}
            
            **Why Global Skip?**
            - Makes optimization easier (learn residual)
            - Prevents complete failure (always has baseline)
            - Faster convergence during training
            - Better gradient propagation
            
            **Formula:**
            ```
            output_final = output + bicubic_upsample(input)
            ```
            
            The network learns to **add details** to the bicubic baseline
            rather than reconstructing from scratch.
            """)
        
        with col2:
            st.code(f"""
import torch.nn.functional as F

def forward(self, x):
    # Upsample input for skip connection
    if self.use_global_skip:
        x_upsampled = F.interpolate(
            x,
            scale_factor={config.scale_factor},
            mode='bicubic',
            align_corners=False
        )
    
    # Main path
    features = self.feature_extraction(x)
    residual_out = self.residual_blocks(features)
    bottleneck_out = self.bottleneck(residual_out)
    bottleneck_out = bottleneck_out + features
    upsampled = self.upsampling(bottleneck_out)
    output = self.reconstruction(upsampled)
    
    # Global skip connection
    if self.use_global_skip:
        output = output + x_upsampled
    
    return output
            """, language="python")
    
    st.markdown('<div class="section-header">SRCNN vs ESRCNN Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Classic SRCNN (3-layer)
        
        ❌ **Limitations:**
        - Only 3 convolutional layers
        - ~70K parameters
        - 17×17 receptive field
        - No skip connections
        - MSE loss only → blurry faces
        - Requires bicubic pre-upsampling
        
        ⏱️ **Performance:**
        - Training: 2-3 hours
        - PSNR: 24-26 dB
        - Quality: ⭐⭐ (Blurry)
        """)
    
    with col2:
        st.markdown(f"""
        ### Enhanced SRCNN (ESRCNN)
        
        ✅ **Advantages:**
        - {config.num_residual_blocks} residual blocks
        - ~900K parameters (13× more)
        - ~50×50 receptive field (3× larger)
        - Multiple skip connections
        - Perceptual + L1 + SSIM loss → sharp faces
        - Native LR input (no pre-upsampling)
        
        ⏱️ **Performance:**
        - Training: 8-12 hours
        - PSNR: 28-32 dB
        - Quality: ⭐⭐⭐⭐⭐ (Sharp)
        """)
    
    st.success("""
    **📌 Recommendation:** Use ESRCNN for face super-resolution projects. 
    The additional training time is worth the significant quality improvement!
    """)
