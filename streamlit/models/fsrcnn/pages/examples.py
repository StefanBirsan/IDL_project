"""
Examples page for Enhanced SRCNN (ESRCNN)
Interactive inference and visualization
"""
import streamlit as st
from pathlib import Path

ASSETS_PATH = Path(__file__).parent.parent.parent.parent / "assets"

def render_examples_page(config):
    st.markdown('<div class="section-header">The Results</div>', unsafe_allow_html=True)

    st.markdown("""
                For the dataset, we used the [CelebA](
                https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) dataset consisting of 200k+ 
                of celebrity faces at a standard size of 178x218px.
                """)
    
    st.markdown("""## First attemppt: 4x super-resolution for 20 epochs""")

    st.image(ASSETS_PATH / "fsrcnn_example_1.png")
    st.image(ASSETS_PATH / "fsrcnn_example_2.png")
    st.image(ASSETS_PATH / "fsrcnn_example_3.png")
