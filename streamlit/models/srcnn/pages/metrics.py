import streamlit as st


def render_metrics_page(config):
    st.markdown('<div class="section-header">Metrics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### PSNR (Peak Signal-to-Noise Ratio)")
        st.markdown("""
        Measures the ratio (in decibels dB) between the maximum possible pixel value and the mean squared error (MSE) between the reconstructed image and the ground truth. Higher PSNR indicates better quality.
        """)
    
    with col2:
        st.markdown("### SSIM (Structural Similarity Index Measure)")
        st.markdown("""
        Evaluates the perceptual similarity between the reconstructed image and the ground truth by considering luminance, contrast, and structure. Values range from -1 to 1, where values closer to 1 indicate higher similarity, 0 indicates no similarity, and negative values indicate dissimilarity (anti-correlation).
        """)