"""README - SRCNN (Super-Resolution Convolutional Neural Network) for Face Super-Resolution

================================================================================
OVERVIEW
================================================================================

This implementation provides a complete PyTorch-based SRCNN model for Face 
Super-Resolution (SR), based on the paper:

    "Image Super-Resolution Using Very Deep Convolutional Networks for 
     Photorealistic Results" by Dong et al. (TPAMI 2016)

The model uses a 9-5-5 convolutional architecture specifically optimized for 
face image enhancement and upscaling.


================================================================================
ARCHITECTURE (9-5-5 Configuration)
================================================================================

Input: Low-resolution face image (already upscaled to target size via bicubic)
Output: High-resolution super-resolved face image

Layer 1 - Patch Extraction:
  - Conv2d(3 → 64, 9×9 kernel) + ReLU
  - Extracts 9×9 patches and maps to 64-dimensional feature representations
  - 9×9 is large enough to capture context

Layer 2 - Non-linear Mapping:
  - Conv2d(64 → 32, 5×5 kernel) + ReLU  
  - Maps 64D feature space to 32D representation (optimized for structural info)
  - 5×5 kernel balances computation vs. expressiveness

Layer 3 - Reconstruction:
  - Conv2d(32 → 3, 5×5 kernel) [Linear activation]
  - Reconstructs the final RGB image
  - NO activation function - allows unrestricted output
  - Residual connection: Output = Input + Conv3(Conv2(Conv1(Input)))

Total Parameters: ~104,480


================================================================================
KEY FEATURES & OPTIMIZATIONS
================================================================================

1. LAYER-SPECIFIC LEARNING RATES:
   - Layers 1-2: lr = 1×10⁻⁴ (faster learning, broader feature adaptation)
   - Layer 3:    lr = 1×10⁻⁵ (slower learning, fine-grained refinement)
   - Ratio: 10:1 ensures stable convergence

2. INITIALIZATION:
   - Weights initialized from N(μ=0, σ=0.001)
   - Biases initialized to 0
   - Critical for SRCNN convergence

3. LOSS FUNCTION:
   - Mean Squared Error (MSE): minimizes Euclidean distance
   - Directly optimizes for higher PSNR
   - Simple but effective for face SR

4. OPTIMIZER:
   - SGD with momentum=0.9
   - Momentum helps escape local minima and accelerates convergence
   - No weight decay (L2 regularization)

5. PREPROCESSING:
   Input must be LR image upscaled to target size using bicubic interpolation
   Example:
     HR: 256×256 (original)
     → Downscale 2×: 128×128 (LR)
     → Upscale 2× bicubic: 256×256 (input to SRCNN)
     → SRCNN output: 256×256 (refined)

6. DATA AUGMENTATION (Training only):
   - Random horizontal flips
   - Random rotations (90°, 180°, 270°)
   - Random brightness adjustments
   - Prevents overfitting and improves generalization


================================================================================
TRAINING CONFIGURATION
================================================================================

Dataset:
  - FFHQ or similar high-quality face dataset
  - Training images cropped to 33×33 patches (prevents border effects)
  
Batch Size: 64
Epochs: 100
Learning Rates:
  - Early layers (1-2): 1e-4
  - Reconstruction layer (3): 1e-5
Optimizer: SGD(momentum=0.9, weight_decay=0.0)
Loss: MSELoss()

Scale Factors Supported: 2×, 4×, 8× super-resolution

Pre-processing:
  - Images normalized to [0, 1]
  - Cropped to 33×33 during training to avoid border artifacts
  - Bicubic upscaling applied offline

Schedule:
  - Validate every 5 epochs
  - Save checkpoint every 10 epochs
  - Save best model (lowest val loss)


================================================================================
FILE STRUCTURE
================================================================================

training/train_utils/srcnn.py
  - Core SRCNN model class
  - Implements 9-5-5 architecture with residual connection
  - Methods: forward(), get_layer_parameters(), summary()

training/core/config_srcnn.py
  - SRCNNTrainingConfig dataclass
  - All training hyperparameters and options
  - to_dict()/from_dict() for serialization

training/core/trainer_srcnn.py
  - SRCNNTrainer class - manages full training pipeline
  - Methods: train_epoch(), validate(), train(), _save_checkpoint()
  - Layer-specific optimizer setup
  - PSNR/SSIM metric calculation
  - Checkpoint and ONNX export

training/train_utils/face_sr_dataset.py
  - FaceSuperResolutionDataset class
  - Handles LR/HR image pair creation
  - Bicubic preprocessing and cropping
  - Augmentation pipeline
  - get_face_sr_dataloaders() convenience function

training/train_srcnn.py
  - Complete end-to-end training script
  - Command-line interface for full customization
  - Checkpoint loading and resuming

training/inference/srcnn_inference.py
  - SRCNNInference class for inference
  - Single image and batch processing
  - PSNR/SSIM evaluation metrics


================================================================================
QUICK START GUIDE
================================================================================

STEP 1: Prepare Dataset
━━━━━━━━━━━━━━━━━━━━━━

Option 1 - Single Directory (Recommended):
  dataset/images1024x1024/
  ├── face_001.jpg
  ├── face_002.jpg
  ├── face_003.jpg
  └── ... (more images)

The dataset loader will automatically split into 80% train / 20% val.

Option 2 - Pre-split Directories:
  dataset/ffhq/
  ├── train/
  │   ├── img_01.jpg
  │   ├── img_02.jpg
  │   └── ... (more images)
  └── val/
      ├── img_001.jpg
      ├── img_002.jpg
      └── ... (more images)

STEP 2: Train Model
━━━━━━━━━━━━━━━━━━

uv run training/train_srcnn.py --data-dir dataset/images1024x1024 --batch-size 64

STEP 3: Inference
━━━━━━━━━━━━━━━━

With Python:
  from training.inference.srcnn_inference import SRCNNInference
  sr = SRCNNInference('checkpoints/srcnn/best_model.pth')
  sr_image = sr.super_resolve('low_res_face.jpg', scale_factor=2)

Batch processing:
  sr.batch_super_resolve('test_images/', 'results/', scale_factor=2)

Alternatively via command line:
  uv run -c "from training.inference.srcnn_inference import SRCNNInference; sr = SRCNNInference('checkpoints/srcnn/best_model.pth'); sr.batch_super_resolve('test_images/', 'results/')"

STEP 4: ONNX Export
━━━━━━━━━━━━━━━━

Automatically exported after training to:
  checkpoints/srcnn/srcnn_face.onnx

Or manually:
  trainer.export_to_onnx()


================================================================================
COMMAND-LINE USAGE
================================================================================

Basic Training (with defaults):
  uv run training/train_srcnn.py --data-dir dataset/images1024x1024

4× Super-Resolution:
  uv run training/train_srcnn.py --data-dir dataset/images1024x1024 --scale-factor 4

Custom Batch & Learning Rates:
  uv run training/train_srcnn.py --data-dir dataset/ffhq \\
    --batch-size 128 --lr-early 5e-5 --lr-recon 5e-6

More Epochs & Custom Crop Size:
  uv run training/train_srcnn.py --data-dir dataset/images1024x1024 \\
    --num-epochs 200 --crop-size 48

Resume from Checkpoint:
  uv run training/train_srcnn.py --data-dir dataset/images1024x1024 \\
    --resume checkpoints/srcnn/checkpoint_epoch_0050.pth

All Available Options:
  uv run training/train_srcnn.py --help


================================================================================
THEORETICAL BACKGROUND
================================================================================

WHY SRCNN FOR FACE SR?
━━━━━━━━━━━━━━━━━━━

1. SIMPLICITY + EFFECTIVENESS
   - Fewer layers (3) than deeper networks
   - Fewer parameters (~104K vs millions)
   - Faster training and inference
   - Excellent PSNR/SSIM on faces

2. LAYER-WISE DESIGN
   - Layer 1: Learns patch representations (high-level features)
   - Layer 2: Non-linear mapping (learns manifold of natural images)
   - Layer 3: Reconstruction (precise pixel value prediction)

3. RESIDUAL CONNECTION
   - Network learns the HIGH-FREQUENCY details (residuals)
   - Easier optimization than learning full image
   - Equation: Output = LR_upscaled + HighFreqDetails

4. MSE LOSS
   - Proven to maximize PSNR for SR tasks
   - PSNR = 20 × log₁₀(MAX_VAL / √MSE)
   - Direct relationship: minimize MSE → maximize PSNR

SCALE FACTORS
━━━━━━━━━━━━

2×: Common for slight enhancement (moderate computation)
4×: Significant upscaling (2x slower than 2×)  
8×: Extreme upscaling (use cascaded approach for better results)

Example:
  256×256 → 2×: 512×512
  256×256 → 4×: 1024×1024
  256×256 → 8×: 2048×2048


================================================================================
METRICS & EVALUATION
================================================================================

PSNR (Peak Signal-to-Noise Ratio)
  - Range: 20-40 dB (higher is better)
  - Standard metric for image quality
  - Typical face SR: 28-35 dB (2× scale)

SSIM (Structural Similarity Index)
  - Range: -1.0 to 1.0 (higher is better)
  - Measures perceived similarity to human eye
  - Typical face SR: 0.80-0.95

Expected Performance (2× on FFHQ):
  - Bicubic baseline: ~32 dB PSNR, 0.82 SSIM
  - SRCNN (trained): ~34 dB PSNR, 0.88 SSIM
  - Improvement: ~2 dB PSNR, ~7% SSIM gain


================================================================================
TROUBLESHOOTING
================================================================================

1. Low PSNR After Training
   → Check: Learning rates might be too high (diverging)
   → Solution: Reduce lr-early to 5e-5, lr-recon to 5e-6

2. Training Loss Oscillating
   → Check: Batch size too small
   → Solution: Increase batch size to 128+ if GPU memory allows

3. CUDA Out of Memory
   → Solution 1: Reduce batch size (--batch-size 32)
   → Solution 2: Reduce image size (--crop-size 24)
   → Solution 3: Use CPU (--device cpu)

4. Poor Generalization (High val loss)
   → Check: Augmentation disabled or insufficient
   → Solution: Enable augmentation with proper dropout
   → Or: Use dataset with more variation

5. Slow Training
   → Using CPU instead of CUDA
   → Check: --device cuda
   → Verify: torch.cuda.is_available() returns True

6. ONNX Export Failing
   → Check: Model architecture is ONNX-compatible
   → Solution: Ensure torch >= 1.10, opset_version <= 17


================================================================================
ADVANCED TIPS
================================================================================

1. CASCADED SUPER-RESOLUTION (4× or 8× results)
   Instead of 8×SR directly, chain multiple 2× models:
   LR → 2×SR → 4×SR → 8×SR
   Better results than direct 8×SR

2. MULTI-SCALE LOSSES
   Modify trainer to include losses at multiple scales
   See loss functions in trainer_srcnn.py

3. TRANSFER LEARNING
   Train on general images (DIV2K) first, then fine-tune on faces
   Improves generalization

4. PERCEPTUAL LOSS
   Replace MSE with VGG-based perceptual loss for better visual quality
   Requires pre-trained VGG network

5. GAN-BASED ENHANCEMENT
   After SRCNN SR, apply lightweight GAN for texture enhancement
   ESRGAN is state-of-the-art in this space


================================================================================
REFERENCES
================================================================================

[1] Dong et al. (2016). "Image Super-Resolution Using Very Deep Convolutional Networks"
    TPAMI 2016. https://arxiv.org/abs/1501.04112

[2] Kim et al. (2016). "Deeply-Recursive Convolutional Network for Image Super-Resolution"
    CVPR 2016

[3] Karras et al. (2018). "Progressive Growing of GANs for Improved Quality, Stability, 
    and Variation"
    ICLR 2018 - FFHQ dataset paper

[4] PyTorch Documentation: https://pytorch.org/docs/


================================================================================
"""


"""
================================================================================
PYTHON API EXAMPLES
================================================================================
"""

# Example 1: Loading and using SRCNN in Python
# ─────────────────────────────────────────────

from training.train_utils.srcnn import SRCNN
import torch

# Create model
model = SRCNN(in_channels=3, intermediate_channels=32, scale_factor=2)
model = model.to('cuda')
model.eval()

# Forward pass
lr_image = torch.randn(1, 3, 256, 256).to('cuda')
with torch.no_grad():
    sr_image = model(lr_image)  # Output: (1, 3, 256, 256)


# Example 2: Training from scratch
# ──────────────────────────────────

from training.core.config_srcnn import SRCNNTrainingConfig
from training.core.trainer_srcnn import SRCNNTrainer
from training.train_utils.face_sr_dataset import get_face_sr_dataloaders

# Configuration
config = SRCNNTrainingConfig(
    scale_factor=2,
    batch_size=64,
    num_epochs=100,
    lr_early_layers=1e-4,
    lr_reconstruction_layer=1e-5,
    data_dir='dataset/images1024x1024'
)

# Data
train_loader, val_loader = get_face_sr_dataloaders(
    data_dir='dataset/images1024x1024',
    batch_size=config.batch_size,
    scale_factor=config.scale_factor,
    crop_size=33
)

# Train
trainer = SRCNNTrainer(config)
history = trainer.train(train_loader, val_loader)


# Example 3: Inference on single image
# ──────────────────────────────────────

from training.inference.srcnn_inference import SRCNNInference
from PIL import Image
import numpy as np

# Load inference model
sr = SRCNNInference('checkpoints/srcnn/best_model.pth', device='cuda')

# Super-resolve
sr_image = sr.super_resolve('low_res_face.jpg', scale_factor=2)

# Save result
output = (sr_image * 255).astype(np.uint8)
Image.fromarray(output).save('sr_face.png')


# Example 4: Batch processing
# ────────────────────────────

sr = SRCNNInference('checkpoints/srcnn/best_model.pth')
sr.batch_super_resolve(
    image_dir='test_images/',
    output_dir='results/',
    scale_factor=2
)


================================================================================
VERSION INFO
================================================================================

SRCNN Implementation: v1.0
PyTorch: >= 1.9.0
Python: >= 3.8
GPU: NVIDIA CUDA 11.0+ (optional, CPU mode supported)

License: MIT


================================================================================
"""
