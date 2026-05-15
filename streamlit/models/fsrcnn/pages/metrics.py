"""
Metrics page for Enhanced SRCNN (ESRCNN)
Training metrics visualization and performance analysis
"""
import streamlit as st
from pathlib import Path

ASSETS_PATH = Path(__file__).parent.parent.parent.parent / "assets"

def render_metrics_page(config):

    st.markdown('<div class="section-header">Model Metrics</div>', unsafe_allow_html=True)

    st.markdown("## PSNR (Peak Signal-to-Noise Ratio)")
    st.markdown("""
                Measures the ratio (in decibels dB) between the maximum possible pixel value and the mean squared error (MSE) between the reconstructed image and the ground truth. Higher PSNR indicates better quality.
                """)

    with st.expander("How PSNR evolves throughout training"):
        st.markdown("""
                    When training FSRCNN for 20 epochs, this is how the PSNR evolved over each epoch:
                    """)

        st.image(ASSETS_PATH / "fsrcnn_psnr_graph.png",
                 caption="PSNR evolution over training, sourced through Weights and Biases platform.", width="content")




