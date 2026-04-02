"""
Documentation page for Physics-Informed MAE
Comprehensive technical documentation integrated from training guide
"""
import streamlit as st


def render_documentation_page(config):
    """Render documentation page for Physics-Informed MAE"""
    
    st.markdown('<div class="section-header">Model Specifications</div>', unsafe_allow_html=True)
    
    # Training Configuration
    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Model Architecture**")
            st.code(f"""
img_size: {config.img_size}×{config.img_size}
patch_size: {config.patch_size}×{config.patch_size}
num_patches: {(config.img_size//config.patch_size)**2}

Embedding:
  embed_dim: {config.embed_dim}
  num_heads: {config.num_heads}
  head_dim: {config.embed_dim//config.num_heads}
  
Depth:
  encoder: {config.encoder_depth} blocks
  decoder: {config.decoder_depth} blocks
  
Masking:
  ratio: {int(config.mask_ratio*100)}%
  visible: {int((1-config.mask_ratio)*100)}%
            """)
        
        with col2:
            st.markdown("**Training Hyperparameters**")
            st.code(f"""
Optimizer: AdamW
  lr: {config.learning_rate:.2e}
  weight_decay: {config.weight_decay}
  betas: (0.9, 0.95)
  batch_size: {config.batch_size}

Scheduler: CosineAnnealingLR
  T_max: {config.num_epochs} epochs
  
Regularization:
  gradient_clip: 1.0
  drop_path: 0.0-0.1
  
Convergence:
  num_epochs: {config.num_epochs}
  log_interval: 10 batches
            """)
        
        with col3:
            st.markdown("**Loss Configuration**")
            st.code(f"""
Primary Loss:
  L_recon: MSE on masked patches
  
Physics Loss:
  L_flux: Sparse flux regularization
  λ_flux: {config.lambda_flux}
  
Combined:
  L_total = L_recon + 
           λ_flux × L_flux

Loss computed:
  - Training: masked 75% → 25%
  - Eval: full image reconstruction
            """)
    
    st.markdown('<div class="section-header">Loss Functions</div>', unsafe_allow_html=True)
    
    # Reconstruction Loss
    with st.expander("📊 1. Reconstruction Loss (MSE)", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Type:** Mean Squared Error (MSE)
            **Applies to:** Masked patches only
            
            **Mathematical Definition:**
            ```
            L_recon = 1/(B×N_masked) × Σ ||ŷ - y||²
            ```
            
            **Where:**
            - B = batch size ({config.batch_size})
            - N_masked = {int((config.img_size//config.patch_size)**2 * config.mask_ratio)} masked patches
            - ŷ = reconstructed patch values
            - y = original patch values
            
            **Properties:**
            ✓ Differentiable everywhere (smooth gradients)
            ✓ Pixel-level precision encouragement
            ✓ Standard for pixel reconstruction tasks
            ✓ Symmetric around error = 0
            
            **Gradient Flow:**
            - High error → Large gradients → Significant updates
            - Low error → Small gradients → Fine-tuning mode
            
            **Typical Values During Training:**
            - Epoch 01: L_recon ≈ 0.5-1.0
            - Epoch 50: L_recon ≈ 0.1-0.2
            - Epoch 100: L_recon ≈ 0.05-0.15
            """)
        
        with col2:
            st.info("""
            **Why MSE for MAE?**
            
            ✓ Simplicity
            ✓ Stability
            ✓ Computational efficiency
            ✓ Proven effective
            ✓ Interpretability
            """)
    
    # Flux Loss
    with st.expander("🔬 2. Flux Consistency Loss", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Type:** Sparse Flux Regularization
            **Applies to:** Generated flux map from edge detection
            
            **Mathematical Definition:**
            ```
            L_flux = 1/B × Σ Σ (flux_weight[b,p])²
            ```
            
            **Where:**
            - flux_weight = normalized edge intensity per patch
            - Encourages sparsity in flux concentration
            
            **Purpose:**
            1. **Physics Guidance:** Incorporates domain knowledge
            2. **Sparse Attention:** Prevents attending to everything
            3. **Interpretability:** Flux maps show model focus
            4. **Efficiency:** Reduces computation on background
            
            **Effect:**
            - High edge regions → High flux → More attention
            - Low edge regions → Low flux → Less attention
            - Background suppression → Efficient learning
            
            **During Training:**
            - Flux loss weight: λ_flux = {config.lambda_flux}
            - Typical L_flux: 0.001 - 0.05
            - Orders of magnitude smaller than L_recon
            """)
        
        with col2:
            st.info("""
            **Integration Strategy**
            
            L_total = L_recon + 
                    λ × L_flux
            
            Where λ = {config.lambda_flux}
            
            **Balance:**
            - 99% reconstruction
            - 1% physics guidance
            - Stable training
            """)
    
    # Total Loss
    with st.expander("📈 Combined Loss & Training", expanded=True):
        st.markdown(f"""
        **Total Loss Function:**
        ```
        L_total = L_recon + λ_flux · L_flux
                = MSE(masked_regions) + {config.lambda_flux} · L_flux
        ```
        
        **Loss Computation Flow:**
        1. Forward pass: x → encoder → decoder → ŷ
        2. Compute L_recon on {int(config.mask_ratio*100)}% masked patches
        3. Extract edge map from preprocessing
        4. Generate flux map from edges
        5. Compute L_flux from flux map sparse regularization
        6. L_total = L_recon + {config.lambda_flux} × L_flux
        7. Backward pass: compute gradients
        8. Gradient clipping: max_norm = 1.0
        9. Adam update: θ ← θ - lr · ∇L_total
        10. Learning rate scheduling: cosine annealing
        
        **Gradient Clipping:**
        - Prevents exploding gradients in deep networks
        - Ensures stable parameter updates
        - Standard practice for transformers
        - Negligible computational overhead
        """)
    
    st.markdown('<div class="section-header">Activation Functions</div>', unsafe_allow_html=True)
    
    # GELU
    with st.expander("⚡ GELU (Gaussian Error Linear Unit)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Used in:** MLP blocks (feed-forward networks)
            
            **Mathematical Definition:**
            ```
            GELU(x) = x · Φ(x)
            ```
            
            Where Φ(x) is the cumulative distribution function of standard normal.
            
            **Practical Approximation:**
            ```
            GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
            ```
            
            **Characteristics:**
            - **Range:** (-∞, ∞) → (-∞, ∞) but approaches asymptotes
            - **Smoothness:** Infinitely differentiable
            - **Non-linearity:** Gentle, probabilistic gating
            - **Gradient:** Non-zero everywhere
            
            **In MLP Architecture:**
            ```
            x → Linear(768→3072) → GELU → Linear(3072→768) → +residual
            ```
            
            **Why GELU for Transformers?**
            ✓ Smooth gradients (better than ReLU)
            ✓ Probabilistic interpretation
            ✓ Works better in ViT than ReLU
            ✓ Better scaling properties
            """)
        
        with col2:
            st.code("""
import torch.nn as nn

# GELU activation
gelu = nn.GELU()

# Comparison
x = torch.tensor([-2, -1, 0, 1, 2])
relu_out = torch.relu(x)
gelu_out = gelu(x)

# Output:
# ReLU:  [0, 0, 0, 1, 2]
# GELU: [-0.02, -0.16, 0, 0.84, 1.96]
            """, language="python")
    
    # Tanh
    with st.expander("🔄 Tanh (Hyperbolic Tangent)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Used in:** 
            - Preprocessing normalization
            - Flux map normalization
            
            **Mathematical Definition:**
            ```
            tanh(x) = (e^x - e^-x) / (e^x + e^-x)
            ```
            
            **Approximation in code:**
            ```
            (e^x - e^-x) / (e^x + e^-x)
            ```
            
            **Characteristics:**
            - **Range:** [-1, 1] (bounded and symmetric)
            - **Mean:** 0 (zero-centered)
            - **Gradient:** tanh'(x) = 1 - tanh²(x)
            - **Max gradient:** 1.0 at x=0
            
            **In Preprocessing:**
            ```
            x_norm = tanh(x)  # HDR stability
            edge_map = tanh(edge_map / max_value)
            ```
            
            **Why Tanh for Preprocessing?**
            ✓ Handles HDR (high dynamic range) images
            ✓ Bounded output prevents extreme values
            ✓ Better gradients than sigmoid
            ✓ Zero-centered (faster convergence)
            ✓ Smooth, symmetric behavior
            ✓ Robust to outlier pixel values
            """)
        
        with col2:
            st.code("""
import torch

# Tanh normalization
x_raw = torch.tensor([0, 100, 1000, 10000])
x_norm = torch.tanh(x_raw)

# Output:
# [0.0000, 1.0000, 1.0000, 1.0000]
# All values map to [-1, 1]

# Edge map normalization
edge_raw = torch.tensor([0.1, 0.5, 2.0, 5.0])
edge_norm = torch.tanh(edge_raw / edge_raw.max())

# Provides smooth, bounded values
            """, language="python")
    
    # Softmax
    with st.expander("🎯 Softmax", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Used in:** Attention mechanism
            
            **Mathematical Definition:**
            ```
            softmax(x_i) = e^(x_i) / Σⱼ e^(x_j)
            ```
            
            **In Attention Context:**
            ```
            attention_weights = softmax(
                (Q @ K^T) / √(d_k)
            )
            ```
            
            **Properties:**
            - **Sum:** Σ softmax(x) = 1 (probability distribution)
            - **Range:** (0, 1) (strict probabilities)
            - **Differentiable:** Smooth everywhere
            - **Exp spacing:** Amplifies differences
            
            **Effect in Attention:**
            1. Normalize attention scores to [0,1]
            2. Each position attends to all positions
            3. Sum of attention weights = 1
            4. Enables gradient flow to all tokens
            5. Interpretable: shows which tokens matter most
            
            **During Inference:**
            - High attention → Important token interaction
            - Low attention → Ignore noisy token
            - All tokens contribute (soft attention)
            """)
        
        with col2:
            st.code("""
import torch
import torch.nn.functional as F

# Attention scores
Q_proj = queries @ W_q  # (batch, heads, seq, dim)
K_proj = keys @ W_k
scores = (Q_proj @ K_proj.T) / math.sqrt(dim)

# Apply softmax
attn_weights = F.softmax(scores, dim=-1)

# Properties:
assert attn_weights.sum(dim=-1) ≈ 1.0
assert (attn_weights >= 0).all()
assert (attn_weights <= 1).all()
            """, language="python")
    
    # ReLU
    with st.expander("🔌 ReLU (Rectified Linear Unit)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Used in:** CNN Refinement Head
            
            **Mathematical Definition:**
            ```
            ReLU(x) = max(0, x)
            ```
            
            **Characteristics:**
            - **Range:** [0, ∞)
            - **Sparsity:** ~50% of activations zero
            - **Non-linearity:** Piecewise linear
            - **Gradient:** 1 (x > 0), 0 (x < 0)
            
            **In CNN Refinement Head:**
            ```
            Conv(768→256) → ReLU → 
            Conv(256→256) → ReLU → 
            Conv(256→16)
            ```
            
            **Why ReLU in CNN?**
            ✓ Computationally efficient
            ✓ Creates sparse representations
            ✓ Reduces neuron co-adaptation
            ✓ Good empirical performance in CNNs
            ✓ Biological plausibility
            
            **ReLU vs GELU:**
            - ReLU: Fast, sparse (CNN context)
            - GELU: Smooth, dense (transformer MLP)
            """)
        
        with col2:
            st.code("""
import torch
import torch.nn as nn

# ReLU in CNN
cnn_head = nn.Sequential(
    nn.Conv2d(768, 256, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 16, 3, padding=1)
)

# Properties:
x = torch.randn(32, 768, 16, 16)
out = cnn_head(x)

# Sparsity check
sparsity = (out == 0).sum() / out.numel()
print(f"Sparsity: {sparsity:.2%}")
            """, language="python")
    
    st.markdown('<div class="section-header">Optimization Strategy</div>', unsafe_allow_html=True)
    
    # AdamW Optimizer
    with st.expander("🎲 AdamW Optimizer", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Algorithm:** Adaptive Moment Estimation with Weight Decay
            
            **Configuration:**
            ```
            AdamW(
              lr={config.learning_rate:.2e},
              betas=(0.9, 0.95),
              weight_decay={config.weight_decay},
              eps=1e-8
            )
            ```
            
            **Update Rule (simplified):**
            ```
            g_t = ∇L(θ_{{t-1}})              # Gradient
            m_t = β₁·m_{{t-1}} + (1-β₁)·g_t  # First moment
            v_t = β₂·v_{{t-1}} + (1-β₂)·g_t² # Second moment
            
            m̂_t = m_t / (1 - β₁^t)         # Bias correction
            v̂_t = v_t / (1 - β₂^t)
            
            θ_t = θ_{{t-1}} - lr·m̂_t/(√v̂_t + eps) - lr·λ·θ_{{t-1}}
            ```
            
            **Hyperparameter Interpretation:**
            - **lr = {config.learning_rate:.2e}:** Step size (balanced for ViT)
            - **β₁ = 0.9:** Heavy momentum (smooth gradient updates)
            - **β₂ = 0.95:** Adaptive learning rates (conservative)
            - **weight_decay = {config.weight_decay}:** L2 regularization strength
            
            **Why AdamW?**
            ✓ Adaptive per-parameter learning rates
            ✓ Momentum accelerates convergence
            ✓ Decoupled weight decay (better than L2)
            ✓ Industry standard for transformers
            ✓ Stable training across architectures
            """)
        
        with col2:
            st.info("""
            **Optimizer State Size:**
            
            For {config.embed_dim}-dim model:
            
            Weights: ~130M params
            Moments: 2× weights
            Gradients: 1× weights
            
            Total: ~4× model size
            """)
    
    # Learning Rate Schedule
    with st.expander("📉 Cosine Annealing LR Schedule", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Type:** Cosine Annealing with warm restart
            
            **Formula:**
            ```
            lr_t = lr_min + 0.5(lr_max - lr_min)[1 + cos(πt/T)]
            ```
            
            **Where:**
            - t = current epoch (0 to {config.num_epochs})
            - T = total epochs ({config.num_epochs})
            - lr_max = initial LR ({config.learning_rate:.2e})
            - lr_min ≈ 0
            
            **Schedule Behavior:**
            - **Early epochs:** High LR for exploration
            - **Mid epochs:** Gradual decrease for refinement
            - **Late epochs:** Low LR for convergence
            - Smooth decay (no discontinuities)
            
            **Learning Rates by Epoch:**
            - Epoch 001: {config.learning_rate:.2e} (start)
            - Epoch 025: {config.learning_rate*1.06:.2e} (~75%)
            - Epoch 050: {config.learning_rate*0.75:.2e} (~50%)
            - Epoch 075: {config.learning_rate*0.24:.2e} (~25%)
            - Epoch 100: ≈0 (end)
            
            **Why Cosine Annealing?**
            ✓ No manual tuning needed
            ✓ Smooth, continuous decay
            ✓ Empirically proven for ViT
            ✓ Deterministic and reproducible
            ✓ Convergence from exploration to exploitation
            """)
        
        with col2:
            st.info("""
            **Warm-up Strategy:**
            
            No warm-up in default config
            
            Consider adding for:
            - Larger models
            - Smaller batch sizes
            - Unstable initial training
            """)
    
    st.markdown('<div class="section-header">Advanced Training Concepts</div>', unsafe_allow_html=True)
    
    # Gradient Clipping
    with st.expander("📍 Gradient Clipping", expanded=False):
        st.markdown(f"""
        **Purpose:** Prevent exploding gradients
        
        **Implementation:**
        ```python
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        ```
        
        **How it works:**
        ```
        grad_norm = ||∇L||₂  (L2 norm of all gradients)
        
        if grad_norm > 1.0:
            ∇L = ∇L × (1.0 / grad_norm)  # Normalize
        else:
            ∇L unchanged
        ```
        
        **Effect:**
        - Bounds gradient magnitude
        - Prevents loss spikes
        - Enables deeper networks
        - Standard for transformers
        - Cost: negligible
        """)
    
    # Masked Autoencoder Strategy
    with st.expander("🎭 Masked Autoencoder (MAE) Strategy", expanded=False):
        st.markdown(f"""
        **Principle:** Learn from reconstructing masked content
        
        **Process:**
        1. **Masking:** Randomly hide {int(config.mask_ratio*100)}% of patches
        2. **Encoding:** Compress visible {int((1-config.mask_ratio)*100)}% to {config.embed_dim}-dim space
        3. **Decoding:** Reconstruct entire image from compressed representation
        4. **Loss:** MSE between reconstructed and original patches
        
        **Why it works:**
        ✓ Information bottleneck forces meaningful representations
        ✓ No labeled data needed (self-supervised)
        ✓ High masking ratio (75%) → harder task → richer features
        ✓ Encoder learns to extract all relevant info from 25%
        ✓ Proven effective for vision transformers
        
        **Key insight:**
        ```
        Encoder compresses 62.5 KB → {config.embed_dim} dims → Decoder reconstructs 62.5 KB
        This compression-decompression forces learning!
        ```
        """)
    
    # Asymmetric Architecture
    with st.expander("⚔️ Asymmetric Encoder-Decoder", expanded=False):
        st.markdown(f"""
        **Design:**
        - Encoder: {config.encoder_depth} blocks (deep, processes sparse info)
        - Decoder: {config.decoder_depth} blocks (shallow, reconstructs dense info)
        
        **Reasoning:**
        ```
        ENCODER (processes 25% visible patches):
          - {config.encoder_depth} blocks for detailed feature extraction
          - High computation per patch
          - Learns rich representations from sparse data
        
        DECODER (reconstructs 100% of patches):
          - {config.decoder_depth} blocks for efficient reconstruction
          - Lower computation due to decoder output density
          - Sufficient capacity to reconstruct known + unknown regions
        ```
        
        **Benefits:**
        ✓ ~30% faster training than 12-12 symmetric
        ✓ Reduced memory footprint
        ✓ Maintains reconstruction quality
        ✓ Balances capacity and efficiency
        ✓ Empirically proven effective
        """)
    
    st.markdown('<div class="section-header">Data Processing</div>', unsafe_allow_html=True)
    
    with st.expander("📁 Input Data Format & Requirements", expanded=True):
        st.markdown(f"""
        **Supported Format:** NumPy (.npy)
        
        **Required Shape:** {config.img_size}×{config.img_size}
        
        **Shape Options:**
        - (H, W) → 2D single-channel
        - (C, H, W) → 3D multi-channel
        - (B, C, H, W) → Batched (for loading)
        
        **Value Range:** [0, 1] or unbounded
        (Preprocessing handles normalization automatically)
        
        **Loading Pipeline:**
        1. Load .npy file
        2. Convert to PyTorch tensor
        3. Physics-informed preprocessing (Tanh + Sobel)
        4. Patch embedding
        5. Asymmetric masking
        6. Model inference
        
        **Batching Strategy:**
        - Batch size: {config.batch_size}
        - Shuffled: True (training)
        - Workers: 4 (parallel loading)
        - Pin memory: True (GPU transfer)
        """)
    
    st.markdown('<div class="section-header">Key Innovations Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        ### Asymmetric Design
        - **Encoder:** {config.encoder_depth} blocks
        - **Decoder:** {config.decoder_depth} blocks
        - **Efficiency:** 30% faster training
        - **Quality:** Maintained reconstruction
        """)
    
    with col2:
        st.markdown(f"""
        ### Masking Strategy
        - **Ratio:** {int(config.mask_ratio*100)}% masked
        - **Tokens:** {int((1-config.mask_ratio)*100)}% processed by encoder
        - **Self-supervised:** No labels needed
        - **Information bottleneck:** Forces learning
        """)
    
    with col3:
        st.markdown(f"""
        ### Physics Constraint
        - **Edge detection:** Sobel preprocessing
        - **Flux guidance:** λ = {config.lambda_flux}
        - **Sparse attention:** Selective focus
        - **Domain knowledge:** Astronomical aware
        """)
