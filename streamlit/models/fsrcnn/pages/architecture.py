import streamlit as st
from pathlib import Path

ASSETS_PATH = Path(__file__).parent.parent.parent.parent / "assets"

def render_architecture_page(config):

    st.markdown('<div class="section-header">FSRCNN Overview</div>', unsafe_allow_html=True)

    st.image(ASSETS_PATH / 'fsrcnn_architecture_diagram.png')

    st.markdown("""
        The model can be broken down into three conceptual stages

        1. **Feature extraction**
        2. **Non-linear mapping**
        3. **Deconvolution/Reconstruction**
        """)
    
    feature_extraction_col, non_linear_mapping_col, convolution_col = st.columns([1, 1, 1])
    
    with feature_extraction_col:
        st.markdown("""
        ## Feature Extraction
        
        This part consists of a single convolutional layer which maps the input image to `d = 56` feature 
        maps, using a `5x5` kernel.
        
        The number of feature maps controls how "rich" the representation of the image becomes.
        
        As the model is working with raw pixel values, the kernel size is quite large (`5x5`). This is 
        essential to extract valuable information from the image. Bigger kernel size, bigger area to work 
        on and to extract features from.
        
        A padding on the convolution operation is applied in order to keep the spatial size unchanged.
        
        A PReLU (Parametric Rectified Linear Unit) activation is applied to introduce non-linearity. 
        Compared to plain ReLU, PReLU makes non-linearity adaptable and evolving throughout training.
        """)

        st.code("""
self.first_part = nn.Sequential(  
    nn.Conv2d(num_channels,  
              d,  
              kernel_size=5,  
              padding=5//2),  
    nn.PReLU(d)  
)
        """, language="python")
    
    with non_linear_mapping_col:
        st.markdown(r"""
                ## Non-linear mapping

                This part consists of shrinking, mapping and expanding the feature maps.
                
                In the **shrinking step**, a `1x1` ocnvolution compresses the `56` feature maps down to `s = 
                12`. Purely a channel-mixing operation with no spacial context. Why do we shrink? Because 
                all the heavy computation happens here, and the less feature maps we have, the faster the 
                computation.
                
                In the **mapping step**, a series of `m = 4` convolutional layers of `3x3` kernel size 
                operate on the `s = 12` feature maps. Notice the smaller kernel size. Stacking multiple 
                convolutions with smaller kernel sizes gives a similar result to one convolutional layer 
                with a greater kernel size, but with way reduced computational cost.
                For example, **one** convolution operation on $12$ channels with a $5 \times 5$ kernel 
                costs $12 \times 5 \times 5 = 3600$ weights, whereas on the other hand, **two** stacked 
                convolutions with a kernel size of $3 \times 3$ cost $2 \times (3 \times 3 \times 12 \times 12) = 2592$ weights.
                
                In the **expanding step**, we expand back to `d = 56` channels before deconvolution, 
                to restore the full representational capacity needed to reconstruct high-quality pixel values.
                """)

        st.code("""
self.mid_part = [  
    nn.Conv2d(d, s, kernel_size=1),  
    nn.PReLU(s)  
]  
  
# For every mapping layer  
for _ in range(m):  
    # Add a convolution layer followed by a PReLU activation function  
    self.mid_part.extend([  
        nn.Conv2d(s,  
                  s,  
                  kernel_size=3,  
                  padding=3//2),  
        nn.PReLU(s)])  
  
# Add one more convolution layer followed by a PReLU activation function  
self.mid_part.extend([  
    nn.Conv2d(s,  
              d,  
              kernel_size=1),  
    nn.PReLU(d)])  
  
# Convert the list of layers into a sequential module  
self.mid_part = nn.Sequential(*self.mid_part)
                """, language="python")

    with convolution_col:
        st.markdown(r"""
                    ## Deconvolution
                    
                    Now with the `d = 56` feature maps back, it's time for upsampling.
                    
                    A `ConvTranspose2d` operation with a stride equivalent to the scale factor inserts 
                    zeroes between input elements, and **then** applies a `9x9` learned filter, literally 
                    learning how to "fill in" the missing pixels.
                    
                    No activation function is applied here, because the output is already in the desired 
                    form, and it must be left unconstrained.
                    """)

        st.code(r"""
self.last_part = nn.ConvTranspose2d(d,  
                                    num_channels,  
                                    kernel_size=9,  
                                    stride=scale_factor,  
                                    padding=9//2,  
                                    output_padding=scale_factor-1)
        """, language="python")
    
    st.markdown('<div class="section-header">The F in FSCRNN</div>', unsafe_allow_html=True)
    
    st.markdown("""
                FSRCNN is noted to be **faster** than SRCNN, because:
                - Unlike SRCNN, the input low-resolution image is not resized to the size of the desired 
                high-resolution image, but rather this happens in deconvolution.
                - All the expensive convolutions are ran where needed, in the beginning and at the end. In 
                the state of the image where it is rich of feature maps, only small kernel sizes (`1x1`, 
                `3x3`) are used.
                """)
