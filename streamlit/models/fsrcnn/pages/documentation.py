"""
Documentation page for Enhanced SRCNN (ESRCNN)
Comprehensive technical documentation and training guide
"""
import streamlit as st
from pathlib import Path

ASSETS_PATH = Path(__file__).parent.parent.parent.parent / "assets"

def render_documentation_page(config):
    st.markdown('<div class="section-header">Training Documentation</div>', unsafe_allow_html=True)
    
    with st.expander("Training Parameters"):

        st.markdown("The following constitute the training parameters used in training:")

        st.code("""
scale_factor = 4
num_channels = 3 # RGB
learning_rate = 1e-3
batch_size = 16
num_epochs = 20
        """, language="python")

        st.markdown("""
                    - Scale factor: The factor by which high-resolution images are downsampled to generate 
                    the low-resolution counterpart, and thus **the factor by which low-resolution images 
                    are expected to be upsampled the high-resolution prediction images**.
                    - Number of channels: The number of channels in the input image. Since we work with RGB images, we use 3.
                    - Learning rate: The rate at which the model learns. Used by the Adam optimizer as a 
                    base learning rate which adapts over training (see below).
                    - Batch size: The number of images in each batch. After each batch, backpropagation 
                    happens. Smaller batch sizes lead to noisy gradients that might be inaccurate. Larger 
                    batches can make use of GPU parallelism as well, speeding up training, but also taking 
                    up a lot of memory size.
                    - Number of epochs: Number of iterations (training and validation) over the dataset. 
                    Too many can lead to overfitting (the model knowing the training dataset too well and 
                    not being able to adapt to unknown inputs) but too little can lead to underfitting. We 
                    chose 20 as a good compromise.
                    """)

    with st.expander("Loss and Optimizer"):
        st.markdown("""
                    ## Loss Function: Mean Squared Error (MSE)
                    
                    MSE compares the model output with the high-resolution ground truth **pixel-by-pixel**.
                    
                    For every pixel, it subtracts from the true value the predicted value, squares it and averages it out across all pixels. The lower the better.
                    
                    ## Optimizer: Adam
                    
                    After each batch, the process of backpropagation is carried out to compute the 
                    gradients for each weight in the network. Gradients refer to how sensitive the loss is 
                    to each weight. The Adam (Adaptive Moment Estimation)  optimizer updates the learning rate for each weight, 
                    in an adaptive manner, according to how much each weight contributes to the loss. 
                    """)