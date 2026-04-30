"""
Examples page for Enhanced SRCNN (ESRCNN)
Interactive inference and visualization
"""
import streamlit as st


def render_examples_page(config):
    """Render examples page for ESRCNN"""
    
    st.markdown('<div class="section-header">Face Super-Resolution Examples</div>', unsafe_allow_html=True)
    
    st.info("""
    **Enhanced SRCNN (ESRCNN)** provides high-quality face super-resolution with:
    - Deep residual architecture for complex feature learning
    - Perceptual loss for realistic facial details
    - Superior quality compared to classic SRCNN
    """)
    
    # Model Loading Section
    with st.expander("🔧 Model Setup", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Model Configuration:**
            - Scale Factor: {config.scale_factor}×
            - Residual Blocks: {config.num_residual_blocks}
            - Feature Channels: {config.num_features}
            - Parameters: ~930K
            
            **Expected Performance:**
            - PSNR: 28-32 dB
            - SSIM: 0.85-0.92
            - Quality: ⭐⭐⭐⭐⭐
            """)
        
        with col2:
            st.code(f"""
from training.train_utils import EnhancedSRCNN

# Create model
model = EnhancedSRCNN(
    scale_factor={config.scale_factor},
    num_residual_blocks={config.num_residual_blocks},
    num_features={config.num_features},
    use_global_skip={str(config.use_global_skip)}
)

# Load checkpoint
checkpoint = torch.load(
    '{config.checkpoint_name}',
    map_location='{config.device}'
)
model.load_state_dict(
    checkpoint['model_state_dict']
)
model.eval()
            """, language="python")
    
    # Inference Pipeline
    with st.expander("🚀 Inference Pipeline", expanded=True):
        st.markdown("""
        ### Step-by-Step Inference
        
        Unlike classic SRCNN, ESRCNN works directly on native LR images without bicubic pre-upsampling.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Input Processing:**
            1. Load LR image (e.g., 64×64)
            2. Normalize to [0, 1] range
            3. Convert to tensor (B, 3, H, W)
            4. Move to device (CPU/GPU)
            
            **No bicubic upsampling needed!**
            ESRCNN handles upsampling internally with learned sub-pixel convolution.
            """)
        
        with col2:
            st.code(f"""
import torch
from PIL import Image
import torchvision.transforms as T

# Load and preprocess LR image
lr_image = Image.open('face_lr.jpg')
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
lr_tensor = transform(lr_image).unsqueeze(0)

# Inference
with torch.no_grad():
    hr_output = model(lr_tensor)

# Post-process
hr_output = hr_output.squeeze(0)
hr_output = (hr_output + 1) / 2  # Denormalize
hr_image = T.ToPILImage()(hr_output)
hr_image.save('face_hr.jpg')
            """, language="python")
    
    # Comparison Visualization
    st.markdown('<div class="section-header">Quality Comparison</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔴 Bicubic Interpolation**
        - Blurry
        - Soft edges
        - No detail enhancement
        - PSNR: ~20 dB
        """)
    
    with col2:
        st.markdown("""
        **🟡 Classic SRCNN**
        - Slightly better than bicubic
        - Still blurry faces
        - Limited improvement
        - PSNR: 24-26 dB
        """)
    
    with col3:
        st.markdown("""
        **🟢 Enhanced SRCNN**
        - Sharp facial features
        - Clear skin texture
        - Realistic details
        - PSNR: 28-32 dB
        """)
    
    # Interactive Demo Section
    st.markdown('<div class="section-header">Interactive Demo</div>', unsafe_allow_html=True)
    
    st.warning("""
    **Note:** This is a placeholder for the interactive demo.
    To enable live inference, you need to:
    1. Train an ESRCNN model
    2. Save the checkpoint to the models directory
    3. Load the checkpoint in this page
    
    For now, see the code examples above for inference usage.
    """)
    
    # Example Code
    with st.expander("📝 Complete Inference Example", expanded=False):
        st.code(f"""
\"\"\"
Complete ESRCNN Inference Script
Super-resolve a face image from LR to HR
\"\"\"

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from training.train_utils.esrcnn import EnhancedSRCNN

def load_model(checkpoint_path, device='{config.device}'):
    \"\"\"Load trained ESRCNN model\"\"\"
    model = EnhancedSRCNN(
        scale_factor={config.scale_factor},
        num_residual_blocks={config.num_residual_blocks},
        num_features={config.num_features}
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def super_resolve_image(model, lr_image_path, output_path, device='{config.device}'):
    \"\"\"Super-resolve a single image\"\"\"
    # Load and preprocess
    lr_image = Image.open(lr_image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    lr_tensor = transform(lr_image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        hr_tensor = model(lr_tensor)
    
    # Post-process
    hr_tensor = torch.clamp(hr_tensor, 0, 1)
    hr_image = transforms.ToPILImage()(hr_tensor.squeeze(0).cpu())
    
    # Save
    hr_image.save(output_path)
    print(f"Saved HR image to {{output_path}}")
    
    return hr_image

def batch_super_resolve(model, input_dir, output_dir, device='{config.device}'):
    \"\"\"Super-resolve all images in a directory\"\"\"
    from pathlib import Path
    import os
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all images
    for img_path in input_path.glob('*.jpg'):
        output_file = output_path / f"{{img_path.stem}}_hr.jpg"
        super_resolve_image(model, str(img_path), str(output_file), device)
        print(f"Processed: {{img_path.name}}")

# Main usage
if __name__ == '__main__':
    # Load model
    model = load_model('checkpoints/esrcnn/best_model.pth', device='cuda')
    
    # Single image
    super_resolve_image(
        model,
        'examples/face_lr.jpg',
        'examples/face_hr_esrcnn.jpg'
    )
    
    # Batch processing
    batch_super_resolve(
        model,
        'dataset/test_lr',
        'results/test_hr'
    )
        """, language="python")
    
    # Performance Tips
    with st.expander("⚡ Performance Optimization Tips", expanded=False):
        st.markdown("""
        ### Inference Speed Optimization
        
        **1. Use Mixed Precision:**
        ```python
        with torch.cuda.amp.autocast():
            hr_output = model(lr_input)
        # 2× faster inference
        ```
        
        **2. Batch Processing:**
        ```python
        # Process multiple images at once
        lr_batch = torch.stack([img1, img2, img3, img4])
        with torch.no_grad():
            hr_batch = model(lr_batch)
        # More efficient GPU utilization
        ```
        
        **3. Export to ONNX:**
        ```python
        # Export for faster inference
        torch.onnx.export(
            model,
            dummy_input,
            'esrcnn_face.onnx',
            opset_version=13
        )
        # Can be optimized further with TensorRT
        ```
        
        **4. Use TorchScript:**
        ```python
        # JIT compile for faster execution
        scripted_model = torch.jit.script(model)
        scripted_model.save('esrcnn_scripted.pt')
        # 1.5-2× speedup
        ```
        
        **Expected Inference Times:**
        - CPU: ~100-200ms per image (128×128 → 256×256)
        - GPU (RTX 3060): ~10-20ms per image
        - GPU with mixed precision: ~5-10ms per image
        """)
    
    st.success("""
    **🎉 Ready to try ESRCNN?**
    
    1. Train the model using: `uv run training/train_esrcnn.py --data-dir dataset/ffhq`
    2. Use the inference code above to super-resolve your face images
    3. Compare with classic SRCNN to see the quality improvement!
    """)
