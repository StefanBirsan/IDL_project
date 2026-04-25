"""
Architecture documentation page for SRCNN
"""
import streamlit as st


def render_architecture_page(config):
    
    st.markdown('<div class="main-header">Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    The model consists of three layers:
    
    1. **Patch extraction and representation** layer
    2. **Non-linear mapping** layer
    3. **Reconstruction** layer
    """)
    
    st.markdown('<div class="section-header">Layer Breakdown</div>', unsafe_allow_html=True)
    
    with st.expander("Layer 1: Patch Extraction and Representation", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            This layer densely extracts overlapping patches from the input image, and represents each patch as a high-dimensional feature vector.
                    
            - Overlapping patches are esentially groups ("windows") of pixels from an image where neighboring windows share a significant number of the same pixels.
                    
            Mathematically, this is equivalent to a **convolution operation** with a set of filters. The convolution operation "slides" across the image pixel by pixel, naturally processing these overlapping areas to ensure every part of the image is analyzed in relation with its neighbors.
                    
            Each patch is convolved with $n_1$ different filters. The filters define the dimensionality of a vector, hence "high-dimensional" vector.
            
            - Filters are sets of weights that the network optimizes during training. A filter is designed to "slide" across the image and produce a "feature map" that highlights specific visual information.
            Kernels are the spatial dimensions of the filters. For example, a kernel size of 9x9 means the filter looks at a square of 81 pixels at a time. Larger kernels can capture more contextual information but are computationally more expensive.
            
            - Each patch is represented as a high-dimensional feature vector, which captures all of its specific characteristics.
            
            Using ReLU (Rectified Linear Unit) activation, the high-dimensional vector is mapped into another high-dimensional layer which conceptually represents the same patch but in a more abstract way. This allows the network to learn complex features that are not directly visible in the pixel space.
            """)
        
        with col2:
            st.code("""
self.layer1 = nn.Sequential(
    nn.Conv2d(in_channels=3, # RGB input
            out_channels=64, # 64 filters
            kernel_size=9, # 9x9 kernel to capture larger context
            padding=4,
            bias=True),
                    
    nn.ReLU(inplace=True)
)
""", language="python")
    
    # Non-linear Mapping Layer
    with st.expander("Layer 2: Non-linear Mapping", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            This layer transforms the $n_1$ dimensional feature vector from the first layer into another $n_2$ dimensional feature vector that represents high-resolution patches.
                        
            The layer takes the $n_1$ feature maps and applies a new set of convolution filters to them, often with:
            - a smaller kernel size (e.g. 5x5) to focus on local details
            - a different number of filters (e.g. 32), as the high-resolution representation is expected to be sparser (fewer features)
                        
            Similarly to the first layer, ReLU activation is applied to introduce non-linearity, allowing the network to learn more complex mappings between low-resolution and high-resolution features.
            """)
        
        with col2:
            st.code("""
self.layer2 = nn.Sequential(
    nn.Conv2d(in_channels=64, 
            out_channels=32, 
            kernel_size=5, 
            padding=2,  # padding to maintain spatial dimensions
            bias=True),
                    
    nn.ReLU(inplace=True)
)
""", language="python")
    
    # Reconstruction Layer
    with st.expander("Layer 3: Reconstruction", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
            This layer aggregates the high-resolution patch representations from the second layer to generate the final high-resolution image. Once again, this is implemented as a convolutional layer.
                        
            It applies $c$ filters of size $n_2 \times f_3 \times f_3$, where $c$ is the number of output channels (e.g. 1 for grayscale, 3 for RGB), and $f_3$ is the kernel size (e.g. 5x5).
                        
            This layer has a lower learning rate ($10^{-5}$) than the previous layers ($10^{-4}$) to allow for more stable training, as it directly affects the final output image quality.
                        
            Unlike the previous layers, no activation function is applied after this convolution, as we want the output to be able to take on a wide range of values (not limited by ReLU's non-linearity) to accurately reconstruct the high-resolution image.
            """)
        
        with col2:
            st.code("""
self.layer3 = nn.Conv2d(in_channels=32, 
                        out_channels=3,  # RGB output
                        kernel_size=5, 
                        padding=2,
                        bias=True)
""", language="python")
