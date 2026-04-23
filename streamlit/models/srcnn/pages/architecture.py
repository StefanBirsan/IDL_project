"""
Architecture documentation page for SRCNN
"""
import streamlit as st


def render_architecture_page(config):
    """Render architecture page for SRCNN"""
    
    st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ### SRCNN Pipeline
        
        The model consists of three key convolutional layers:
        
        1. **Input Image (LR)** → Upsampling (Bicubic/Nearest Neighbor)
        2. **Feature Extraction** → ReLU Activation
        3. **Non-linear Mapping** → ReLU Activation
        4. **Reconstruction** → Linear Layer
        5. **Output Image (HR)** ← MSE Loss
        
        **Key Characteristic:** End-to-end trainable CNN with layer-specific learning rates
        """)
    
    with col2:
        st.info(f"""
        **Model Configuration**
        - Scale Factor: {config.scale_factor}×
        - Input Patch: {config.crop_size}×{config.crop_size}
        - Intermediate Channels: {config.intermediate_channels}
        - Architecture: 9-5-5 (kernel sizes)
        - Activation: ReLU
        - Loss: MSE
        """)
    
    st.markdown('<div class="section-header">Component Architecture</div>', unsafe_allow_html=True)
    
    # Upsampling Layer
    with st.expander("🔧 Upsampling Layer", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Increase resolution of low-resolution input
            
            **Methods:**
            - **Bicubic Interpolation**: Smooth, smooth upsampling
            - **Nearest Neighbor**: Fast, preserves edges
            
            **Dimensions:**
            - Input: (B, C, H, W) [Low-Resolution]
            - Output: (B, C, H×{config.scale_factor}, W×{config.scale_factor}) [Intermediate]
            - Upsampling factor: {config.scale_factor}×{config.scale_factor}
            
            **Note:** Upsampling is done before network input to allow fixed-size patches
            """)
        
        with col2:
            st.code(f"""
import torch.nn.functional as F

# Upsampling using interpolation
lr_image = input  # (B, C, H, W)

# Bicubic interpolation
upsampled = F.interpolate(
    lr_image,
    scale_factor={config.scale_factor},
    mode='bicubic',
    align_corners=False
)
# Output: (B, C, H×{config.scale_factor}, W×{config.scale_factor})

# Alternative: Nearest neighbor
upsampled_nn = F.interpolate(
    lr_image,
    scale_factor={config.scale_factor},
    mode='nearest'
)
            """, language="python")
    
    # Feature Extraction Layer
    with st.expander("🎯 Layer 1: Feature Extraction (9×9 Conv)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Extract low-level features from upsampled image
            
            **Configuration:**
            - **Kernel Size:** 9×9
            - **Input Channels:** 1 (or 3 for color)
            - **Output Channels:** {config.intermediate_channels}
            - **Padding:** 4 (to maintain spatial size)
            - **Activation:** ReLU
            
            **Operation:**
            - Large receptive field captures local context
            - Learns feature maps from upsampled input
            - ReLU introduces non-linearity
            
            **Output Shape:**
            - (B, {config.intermediate_channels}, H×{config.scale_factor}, W×{config.scale_factor})
            
            **Learning Rate:** {config.lr_early_layers:.2e} (early layer)
            """)
        
        with col2:
            st.code(f"""
import torch.nn as nn

class FeatureExtraction(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels={config.intermediate_channels},
            kernel_size=9,
            padding=4,
            bias=True
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: (B, 1, H×{config.scale_factor}, W×{config.scale_factor})
        x = self.conv(x)
        x = self.relu(x)
        # Output: (B, {config.intermediate_channels}, H×{config.scale_factor}, W×{config.scale_factor})
        return x
            """, language="python")
    
    # Non-linear Mapping Layer
    with st.expander("🧠 Layer 2: Non-linear Mapping (5×5 Conv)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Learn non-linear transformations of features
            
            **Configuration:**
            - **Kernel Size:** 5×5
            - **Input Channels:** {config.intermediate_channels}
            - **Output Channels:** {config.intermediate_channels}
            - **Padding:** 2 (to maintain spatial size)
            - **Activation:** ReLU
            
            **Operation:**
            - Smaller kernel for detailed feature mapping
            - Maintains channel depth for representational capacity
            - ReLU allows learning complex mappings
            
            **Output Shape:**
            - (B, {config.intermediate_channels}, H×{config.scale_factor}, W×{config.scale_factor})
            
            **Learning Rate:** {config.lr_early_layers:.2e} (early layer)
            """)
        
        with col2:
            st.code(f"""
class NonlinearMapping(nn.Module):
    def __init__(self, channels={config.intermediate_channels}):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            bias=True
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: (B, {config.intermediate_channels}, H×{config.scale_factor}, W×{config.scale_factor})
        x = self.conv(x)
        x = self.relu(x)
        # Output: (B, {config.intermediate_channels}, H×{config.scale_factor}, W×{config.scale_factor})
        return x
            """, language="python")
    
    # Reconstruction Layer
    with st.expander("🎨 Layer 3: Reconstruction (5×5 Conv)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Reconstruct high-resolution image
            
            **Configuration:**
            - **Kernel Size:** 5×5
            - **Input Channels:** {config.intermediate_channels}
            - **Output Channels:** 1 (or 3 for color)
            - **Padding:** 2 (to maintain spatial size)
            - **Activation:** None (linear output)
            
            **Operation:**
            - Final layer predicts residuals or direct pixel values
            - No activation for regression task
            - Generates final output image
            
            **Output Shape:**
            - (B, 1, H×{config.scale_factor}, W×{config.scale_factor})
            
            **Learning Rate:** {config.lr_reconstruction_layer:.2e} (lower for stability)
            """)
        
        with col2:
            st.code(f"""
class Reconstruction(nn.Module):
    def __init__(self, channels={config.intermediate_channels}, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=2,
            bias=True
        )
    
    def forward(self, x):
        # x: (B, {config.intermediate_channels}, H×{config.scale_factor}, W×{config.scale_factor})
        x = self.conv(x)
        # Output: (B, 1, H×{config.scale_factor}, W×{config.scale_factor})
        return x
            """, language="python")
    
    # Full Model
    with st.expander("🔗 Complete SRCNN Model", expanded=True):
        st.code(f"""
import torch.nn as nn

class SRCNN(nn.Module):
    '''Super-Resolution Convolutional Neural Network'''
    
    def __init__(
        self,
        scale_factor={config.scale_factor},
        intermediate_channels={config.intermediate_channels},
        in_channels=1,
        out_channels=1,
        upsampling_mode='bicubic'
    ):
        super(SRCNN, self).__init__()
        
        self.scale_factor = scale_factor
        self.upsampling_mode = upsampling_mode
        
        # Layer 1: Feature Extraction (9×9)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=9,
            padding=4,
            bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: Non-linear Mapping (5×5)
        self.conv2 = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=intermediate_channels,
            kernel_size=5,
            padding=2,
            bias=True
        )
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3: Reconstruction (5×5)
        self.conv3 = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=2,
            bias=True
        )
    
    def forward(self, x):
        '''
        Args:
            x: (B, C, H, W) low-resolution image
        Returns:
            output: (B, C, H×scale, W×scale) super-resolved image
        '''
        # Upsample input
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.upsampling_mode,
            align_corners=False if self.upsampling_mode in ['bilinear', 'bicubic'] else None
        )
        
        # Feature extraction
        x = self.conv1(x)
        x = self.relu1(x)
        
        # Non-linear mapping
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Reconstruction
        x = self.conv3(x)
        
        return x
        """, language="python")
