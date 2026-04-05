# Physics-Informed Masked Vision Transformer

Complete PyTorch implementation of a custom deep learning architecture for astronomical image reconstruction and flux-consistent super-resolution.

> ⚠️ **IMPORTANT**: This codebase has been upgraded to support **true super-resolution (SR)** as the **default behavior**. The model now performs LR → HR upscaling with advanced reconstruction losses. See [`training/TRAINING_ARCHITECTURE_SR.md`](./TRAINING_ARCHITECTURE_SR.md) for detailed SR mode documentation. **Legacy mode is still available** via `--mode legacy` for backward compatibility.

## Architecture Overview

### 1. **Physics-Informed Preprocessing**
- **Double Differentiation**: Applies Sobel filters for edge detection (identifies sharp intensity changes in astronomical data)
- **Tanh Normalization**: Stabilizes high-dynamic-range (HDR) signals commonly found in astronomical images
- **Output**: Normalized image + edge map for downstream guidance

### 2. **Asymmetric Patch Partitioning & Masking (MAE-Style)**
- **Patch Size**: 4×4 non-overlapping patches
- **Masking Ratio**: Configurable 75-90% masking (default: 75%)
- **Asymmetric Processing**: Encoder processes only visible tokens (25% of patches) to reduce computation
- **Mask Tokens**: Learnable tokens added in decoder for masked patch reconstruction

### 3. **Flux Guidance Generation (FGG) Module**
- **Object Detection**: Celestial objects detected via edge maps
- **Photometry Calculation**: Total flux computed from detected objects
- **Flux Map**: Generated using rotatable Gaussian kernels modulated by object flux
- **Purpose**: Guides reconstruction toward physically meaningful solutions

### 4. **ViT Encoder Backbone**
- **Architecture**: 12 Transformer blocks with Multi-Head Self-Attention
- **Positional Embeddings**: Sine-cosine rotary embeddings for spatial awareness
- **Flux Guidance Controller (FGC)**: Injects flux map information into encoder features at each scale
  - Uses global average pooling + MLP to generate control signals
  - Applies multiplicative and additive modulation to features
- **Efficiency**: Processes only visible patches after masking

### 5. **Hybrid Decoder**
- **Architecture**: 8 Transformer blocks + CNN refinement head
- **Reconstruction Steps**:
  1. Append learnable mask tokens to encoded visible tokens
  2. Restore original patch order
  3. Transformer decoder processes full patch sequence for global context
  4. CNN refinement head (3 Conv layers with ReLU) refines output
- **Output**: Per-patch pixel values (patch_size² = 16 values per patch)

### 6. **Loss Functions**

#### Reconstruction Loss (L_recon)
- Mean Squared Error computed only on masked patches
- Encourages model to learn meaningful completion of occluded regions

#### Flux Consistency Loss (L_flux)
- Constrains photometry gap by weighting supervision with generated flux map
- Ensures high-flux regions are reconstructed accurately
- Regularizes flux map sparsity

#### Total Loss
```
L_total = L_recon + λ × L_flux
```
where λ = 0.01 (default)

### 7. **Optimization**
- **Optimizer**: AdamW with betas=(0.9, 0.95)
- **Base Learning Rate**: 1.5e-4
- **Weight Decay**: 0.05
- **Scheduler**: Cosine annealing

## Installation

### Using uv (Recommended)

```bash
# Navigate to project directory
cd IDL_project

# Install dependencies
uv sync

# Or add specific packages
uv pip install torch torchvision scipy scikit-image
```

### Using pip

```bash
pip install -r requirements.txt
```

## Project Structure

```
IDL_project/
├── models/
│   ├── __init__.py                      # Module exports
│   ├── modules.py                       # Core building blocks
│   └── physics_informed_mae.py          # Main architecture
├── dataset/
│   ├── __init__.py                      # Dataset exports
│   ├── numpy_dataset.py                 # Numpy file loader
│   └── data/
│       └── x2/
│           ├── train_hr_patch/          # Numpy files (.npy)
│           └── dataload_filename/       # Train/eval split files
├── training/
│   ├── train.py                         # Training script
│   └── inference.py                     # Inference & visualization
├── pyproject.toml                       # Project dependencies
└── utils/
    └── download_dataset.py              # Dataset download utility
```

## Usage

### 1. **Training the Model**

#### Super-Resolution Mode (DEFAULT - Recommended)

```bash
# Basic SR training (2x upscaling with advanced losses)
uv run training/train.py

# Custom scale factor (4x upscaling)
uv run training/train.py --scale-factor 4

# Custom loss weights
uv run training/train.py \
    --lambda-l1 0.7 \
    --lambda-ssim 0.5 \
    --lambda-fft 0.2 \
    --lambda-recon 1.0
```

See [`training/TRAINING_ARCHITECTURE_SR.md`](./TRAINING_ARCHITECTURE_SR.md) for comprehensive SR training guide.

#### Legacy Mode (LR Reconstruction - Backward Compatible)

```bash
# Use original behavior (LR -> LR reconstruction, no upsampling)
uv run training/train.py --mode legacy

# With custom legacy parameters
uv run training/train.py --mode legacy \
    --batch-size 64 \
    --num-epochs 200 \
    --lr 1.5e-4 \
    --weight-decay 0.05 \
    --lambda-flux 0.01 \
    --mask-ratio 0.75 \
    --img-size 64 \
    --device cuda
```

### 2. **Running Inference**

```bash
# Reconstruct a single sample
uv run training/inference.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --data-dir dataset/data \
    --sample-idx 0 \
    --output-dir results
```

### 3. **Using the Model Programmatically**

```python
import torch
from models import create_physics_informed_mae

# Create model
model = create_physics_informed_mae(img_size=64)

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
model.eval()
with torch.no_grad():
    image = torch.randn(1, 1, 64, 64)  # (B, C, H, W)
    reconstructed, aux_outputs = model(image)

# Access auxiliary outputs
print(aux_outputs.keys())
# dict_keys(['mask', 'edge_map', 'flux_map', 'latent', 'patches_reconstructed'])
```

### 4. **Loading Your Numpy Dataset**

```python
from dataset import get_dataset, get_dataloader

# Load single sample
dataset = get_dataset('dataset/data', split='train', img_size=64)
image, metadata = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Filename: {metadata['filename']}")

# Create data loader
train_loader = get_dataloader(
    'dataset/data',
    split='train',
    batch_size=32,
    num_workers=4,
    img_size=64
)

for batch_images, batch_metadata in train_loader:
    print(f"Batch shape: {batch_images.shape}")  # (32, 1, 64, 64)
    break
```

## Key Components Explained

### Physics-Informed Preprocessing
```
Input Image (B, C, H, W)
    ↓
Sobel Filtering (double differentiation)
    ↓
Edge Map (B, C, H, W)
    ↓
Tanh Normalization
    ↓
Output: Normalized Image + Edge Map
```

### Masking Process
```
Patches (B, N, embed_dim) where N = 256 for 64×64 with 4×4 patches
    ↓
Random Shuffle
    ↓
Select top 25% visible patches (75% masked)
    ↓
Visible Patches (B, 64, embed_dim) → Encoder
Mask Patches → Added as learnable tokens in decoder
```

### Flux Map Generation
```
Edge Map
    ↓
Max Pool to Patch Size
    ↓
Gaussian Kernel Modulation
    ↓
Flux Map (B, 1, 16, 16)
    ↓
Injected via FGC at encoder
```

### Decoder Reconstruction
```
Encoder Output (B, 64, embed_dim)
    ↓
Append Mask Tokens (B, 256, embed_dim)
    ↓
Restore Original Order
    ↓
Transformer Decoder (8 blocks)
    ↓
CNN Refinement Head
    ↓
Output Projection (B, 256, 16)
    ↓
Reshape to Image (B, 1, 64, 64)
```

## Configuration

### Training Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `img_size` | 64 | 32-256 | Input image size |
| `patch_size` | 4 | 4-8 | Patch size (smaller = more patches) |
| `embed_dim` | 768 | 256-1024 | Embedding dimension |
| `mask_ratio` | 0.75 | 0.6-0.9 | Fraction of patches to mask |
| `batch_size` | 32 | 8-128 | Batch size |
| `learning_rate` | 1.5e-4 | 1e-5-1e-3 | Learning rate |
| `weight_decay` | 0.05 | 0.0-0.1 | L2 regularization |
| `lambda_flux` | 0.01 | 0.0-1.0 | Flux loss weight |

### Model Architecture

| Component | Config | Description |
|-----------|--------|-------------|
| Encoder Blocks | 12 | Transformer blocks in encoder |
| Decoder Blocks | 8 | Transformer blocks in decoder |
| Attention Heads | 12 | Multi-head attention heads |
| MLP Ratio | 4.0 | Hidden dim / embed_dim in FFN |

## Output Format

### Training Outputs
- **Checkpoints**: Saved every N epochs in `checkpoints/`
  - `checkpoint_epoch_XXXX.pt`: Periodic checkpoints
  - `checkpoint_best.pt`: Best evaluation checkpoint
- **Logs**: `training_history.json` with loss curves

### Inference Outputs
- **Reconstructed Image**: Same shape as input (B, C, H, W)
- **Auxiliary Outputs**:
  - `mask`: Binary mask indicating which patches were masked
  - `edge_map`: Edge detection map from preprocessing
  - `flux_map`: Flux guidance map
  - `latent`: Encoded representation
  - `patches_reconstructed`: Per-patch reconstruction values

## Tips & Best Practices

1. **High Masking Ratio**: Use 85-90% for more challenging self-supervised learning
2. **Flux Map Quality**: Pre-computed flux maps yield better results than real-time detection
3. **Batch Size**: Use largest batch size your GPU allows (improves stability)
4. **Learning Rate Warmup**: Consider adding warmup for first 10 epochs
5. **Data Normalization**: Ensure numpy arrays are in [-1, 1] range
6. **Gradient Clipping**: Enabled by default (max_norm=1.0)

## Support

For issues or questions:
1. Check the training logs in `checkpoints/training_history.json`
2. Verify dataset loading: `uv run -c "from dataset import get_dataset; ds = get_dataset('dataset/data')"`
3. Test model forward pass: `uv run -c "from models import create_physics_informed_mae; m = create_physics_informed_mae()"`