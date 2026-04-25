from tkinter import Image

import streamlit as st
import numpy as np
from pathlib import Path


def render_examples_page(config):
    st.markdown('<div class="section-header">Our Results</div>', unsafe_allow_html=True)
    
    st.markdown("""
                For the dataset, we used the [FFHQ](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) dataset consisting of 52K+ high-quality human face images at 1024x1024 resolution.

                As described in the documentation page, the LR images were created by downscaling the HR images by the scaling factor, and upscaling back, both using bicubic interpolation.
                """)
    
    st.markdown("""
                ## First attempt: 2x super-resolution for 5 epochs
""")
    
    st.image(Path(__file__).parent.parent.parent.parent / 'assets' / 'srcnn_v1_output.png', caption="SRCNN 2x super-resolution results after 5 epochs of training")

    st.code("""
BICUBIC (basic resizing in image editing software)
PSNR: 36.79 dB
SSIM: 0.9540
SRCNN
PSNR: 36.79 dB
SSIM: 0.9540
            """)
    
    st.markdown("""
The results are not great, as the model visibly fails to reconstruct fine details and its reconstruction is equivalent to bicubic interpolation. The PSNR and SSIM metrics also confirm that the model is not improving over the bicubic baseline.
Also, note that because of 2x downscaling of the LR, the input looks similar to the output, but we assure that the input is in fact downscaled.
                """)
    
    st.markdown("""
                ## Second attempt: 4x super-resolution for 20 epochs
""")
    
    st.image(Path(__file__).parent.parent.parent.parent / 'assets' / 'srcnn_v2_output.png', caption="SRCNN 4x super-resolution results after 20 epochs of training")

    st.code("""
PSNR: 28.13 dB
SSIM: 0.8334
SRCNN
PSNR: 28.13 dB
SSIM: 0.8334
            """)
    
    st.markdown("""
Disappointing results once again... The model is still not improving over the bicubic baseline. We decided to take it up a notch and train more, with a larger scale factor.
                """)
    
    st.markdown("""
                ## Third attempt: 8x super-resolution for 100 epochs
                """)
    st.image(Path(__file__).parent.parent.parent.parent / 'assets' / 'srcnn_v3_output.png', caption="SRCNN 8x super-resolution results after 100 epochs of training")

    st.code("""
BICUBIC (basic resizing in image editing software)
PSNR: 29.55 dB
SSIM: 0.8010
SRCNN
PSNR: 29.55 dB
SSIM: 0.8010
            """)
    
    st.markdown("""
Even after training for 100 epochs, we still have the same results...
                
We decided that perhaps the architecture is too simple (the SRCNN paper itself is quite outdated), and that we need to look into more modern architectures.
                """)

def upload_and_infer():
    st.markdown('<div class="section-header">Try it yourself</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an image of a face to upscale", type=['png, jpg, jpeg'])
    
    if uploaded_file is not None:
        try:
            # Load the uploaded image file
            image = Image.open(uploaded_file)
            image = np.array(image)
            st.info(f"Loaded image with shape: {image.shape}")
            
            # Show upload options
            col1, col2 = st.columns(2)
            with col1:
                st.write("Image Statistics:")
                st.write(f"Min: {image.min():.4f}")
                st.write(f"Max: {image.max():.4f}")
                st.write(f"Mean: {image.mean():.4f}")
                st.write(f"Std: {image.std():.4f}")
            
            with col2:
                st.write("Image Preview:")
                st.image(image, clamp=True, use_column_width=True)
            
            # Inference button
            if st.button("Run Inference", key="inference_button"):
                st.info("Inference would run here with loaded checkpoint...")
                try:
                    import torch
                    st.success("PyTorch available for inference")
                except ImportError:
                    st.warning("PyTorch not installed - install with: pip install torch")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
