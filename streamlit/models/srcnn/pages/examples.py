"""
Examples and inference page for SRCNN
"""
import streamlit as st
import numpy as np
from pathlib import Path


def render_examples_page(config):
    """Render examples page for SRCNN"""
    
    st.info("""
    💡 **Note:** Connect training checkpoints and dataset to run live inference.
    This section shows the inference framework and usage patterns.
    """)
    
    st.markdown('<div class="section-header">Example Showcase</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Input (LR)")
        st.markdown(f"""
        **Low-Resolution Input:**
        - Shape: 64×64 (example)
        - Upsampled to: 128×128 (for {config.scale_factor}×)
        - Bicubic interpolation
        
        **Quality:**
        - Blurry edges
        - Loss of detail
        - Typical LR artifacts
        """)
    
    with col2:
        st.markdown("### Network")
        st.markdown(f"""
        **SRCNN Processing:**
        - Input: Upsampled LR
        - 3 Conv layers
        - Feature extraction
        - Non-linear mapping
        - Reconstruction
        """)
    
    with col3:
        st.markdown("### Output (SR)")
        st.markdown(f"""
        **Super-Resolved Output:**
        - Shape: {128 * config.scale_factor // 2}×{128 * config.scale_factor // 2}
        - Enhanced details
        - Sharpened edges
        - Improved quality
        """)
    
    st.markdown('<div class="section-header">Inference Pipeline</div>', unsafe_allow_html=True)
    
    with st.expander("🚀 How to Run Inference", expanded=True):
        st.code(f"""
import torch
import torch.nn.functional as F
from training.train_utils.srcnn import SRCNN
from training.inference.srcnn_inference import infer_image

# ============ Setup ============
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============ Load Model ============
model = SRCNN(
    scale_factor={config.scale_factor},
    intermediate_channels={config.intermediate_channels},
    in_channels=1,
    out_channels=1,
    upsampling_mode='bicubic'
)

# Load checkpoint
checkpoint = torch.load('checkpoints/srcnn/srcnn_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# ============ Run Inference ============
# Load LR image
lr_image = np.load('path/to/lr_image.npy')  # Shape: (1, 1, H, W) or (H, W)

# Ensure correct shape
if lr_image.ndim == 2:
    lr_image = lr_image[np.newaxis, np.newaxis, ...]
elif lr_image.ndim == 3:
    lr_image = lr_image[np.newaxis, ...]

# Convert to tensor
lr_tensor = torch.from_numpy(lr_image).float().to(device)

# Inference
with torch.no_grad():
    sr_output = model(lr_tensor)

# Convert back to numpy
sr_image = sr_output.squeeze().cpu().numpy()

print(f"Input shape:  {{lr_tensor.shape}}")
print(f"Output shape: {{sr_output.shape}}")

# ============ Visualization ============
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Bicubic upsampled (for comparison)
bicubic = F.interpolate(
    lr_tensor, 
    scale_factor={config.scale_factor}, 
    mode='bicubic', 
    align_corners=False
).squeeze().cpu().numpy()

axes[0].imshow(lr_image.squeeze(), cmap='gray')
axes[0].set_title('Low-Resolution (LR)')
axes[0].axis('off')

axes[1].imshow(bicubic, cmap='gray')
axes[1].set_title('Bicubic Upsampling')
axes[1].axis('off')

axes[2].imshow(sr_image, cmap='gray')
axes[2].set_title('SRCNN Output (SR)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('srcnn_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
        """, language="python")
    
    st.markdown('<div class="section-header">Batch Inference</div>', unsafe_allow_html=True)
    
    st.code("""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from training.train_utils.srcnn import SRCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = SRCNN(scale_factor=2, intermediate_channels=32)
checkpoint = torch.load('checkpoints/srcnn/srcnn_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Process batch
batch_lr = next(iter(data_loader))  # Shape: (B, 1, H, W)
batch_lr = batch_lr.to(device)

with torch.no_grad():
    batch_sr = model(batch_lr)

# Compute metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

psnr_scores = []
ssim_scores = []

for i in range(len(batch_sr)):
    sr = batch_sr[i].squeeze().cpu().numpy()
    hr = batch_hr[i].squeeze().cpu().numpy()
    
    psnr = peak_signal_noise_ratio(hr, sr, data_range=1.0)
    ssim = structural_similarity(hr, sr, data_range=1.0)
    
    psnr_scores.append(psnr)
    ssim_scores.append(ssim)

print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    """, language="python")
    
    st.markdown('<div class="section-header">Visualization Techniques</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Side-by-Side Comparison
        - Input (LR)
        - Output (SR)
        - Reference (HR)
        
        **Shows:**
        - Overall reconstruction quality
        - Artifact presence
        - Detail preservation
        """)
    
    with col2:
        st.markdown("""
        ### Difference Maps
        - Compute: |SR - HR|
        - Visualize error distribution
        - Identify problem regions
        
        **Color Coding:**
        - Dark = low error
        - Bright = high error
        """)
    
    with col3:
        st.markdown("""
        ### Metrics Visualization
        - PSNR vs Scale Factor
        - SSIM over batches
        - Loss curves
        
        **Use for:**
        - Performance tracking
        - Hyperparameter tuning
        - Comparison with baselines
        """)


def upload_and_infer():
    """Section for uploading images and running inference"""
    st.subheader("Interactive Inference")
    
    uploaded_file = st.file_uploader("Upload a .npy image", type=['npy'])
    
    if uploaded_file is not None:
        try:
            # Load the uploaded numpy file
            image = np.load(uploaded_file)
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
