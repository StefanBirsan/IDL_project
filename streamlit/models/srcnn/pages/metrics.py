"""
Metrics page for SRCNN
"""
import streamlit as st


def render_metrics_page(config):
    """Render metrics page for SRCNN"""
    
    st.info("""
    💡 **Note:** Connect training checkpoints and logs to visualize real metrics.
    This section shows the metrics framework that will be populated during training.
    """)
    
    st.markdown('<div class="section-header">Available Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Reconstruction Metrics")
        st.markdown("""
        - **MSE Loss**: Mean Squared Error
        - **PSNR**: Peak Signal-to-Noise Ratio (dB)
        - **SSIM**: Structural Similarity Index
        - **MAE**: Mean Absolute Error
        """)
    
    with col2:
        st.markdown("### Training Metrics")
        st.markdown(f"""
        - **Learning Rate**: {config.lr_early_layers:.2e} (early), {config.lr_reconstruction_layer:.2e} (recon)
        - **Gradient Norm**: Training stability
        - **Batch Time**: Training speed
        - **GPU Memory**: Resource utilization
        """)
    
    with col3:
        st.markdown("### Scale Factor Performance")
        st.markdown("""
        - **2× PSNR**: Standard performance
        - **4× PSNR**: More challenging
        - **8× PSNR**: Advanced difficulty
        - **Convergence Speed**: Epochs to convergence
        """)
    
    st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)
    
    # PSNR
    with st.expander("📈 1. Peak Signal-to-Noise Ratio (PSNR)", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Definition:** Logarithmic measure of image quality
            
            **Mathematical Formula:**
            $$PSNR = 10 \\log_{10}\\left(\\frac{L_{max}^2}{MSE}\\right)$$
            
            **Where:**
            - $L_{max}$ = maximum pixel value (typically 1.0 for normalized)
            - $MSE$ = mean squared error between SR and HR
            
            **Interpretation:**
            - **Higher is better** (lower error)
            - **Units:** Decibels (dB)
            - **Typical range:** 20-40 dB for super-resolution
            
            **Properties:**
            ✓ Directly related to MSE loss
            ✓ Easy to compute
            ✓ Widely used in literature
            ✗ Doesn't perfectly match human perception
            """)
        
        with col2:
            st.info("""
            **PSNR Guidelines**
            
            - 30+ dB: Good quality
            - 25-30 dB: Fair quality
            - 20-25 dB: Poor quality
            - <20 dB: Very poor
            """)
    
    # SSIM
    with st.expander("🎨 2. Structural Similarity Index (SSIM)", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Definition:** Perceptual quality metric based on structural similarity
            
            **Mathematical Formula:**
            $$SSIM = \\frac{(2\\mu_x\\mu_y + C_1)(2\\sigma_{xy} + C_2)}{(\\mu_x^2 + \\mu_y^2 + C_1)(\\sigma_x^2 + \\sigma_y^2 + C_2)}$$
            
            **Components:**
            - $\\mu_x, \\mu_y$ = mean of images
            - $\\sigma_x, \\sigma_y$ = variance of images
            - $\\sigma_{xy}$ = covariance
            - $C_1, C_2$ = stability constants
            
            **Interpretation:**
            - **Range:** -1 to 1
            - **Higher is better** (>0.9 is excellent)
            - **1.0:** Identical images
            
            **Advantages over PSNR:**
            ✓ Better correlation with human perception
            ✓ Considers luminance, contrast, structure
            ✗ Slightly more computationally expensive
            """)
        
        with col2:
            st.info("""
            **SSIM Guidelines**
            
            - 0.9-1.0: Excellent
            - 0.8-0.9: Very good
            - 0.7-0.8: Good
            - 0.6-0.7: Fair
            - <0.6: Poor
            """)
    
    # MAE
    with st.expander("📊 3. Mean Absolute Error (MAE)", expanded=True):
        st.markdown("""
        **Definition:** Average pixel-level absolute difference
        
        **Mathematical Formula:**
        $$MAE = \\frac{1}{N} \\sum_{i=1}^{N} |\\hat{y}_i - y_i|$$
        
        **Where:**
        - $N$ = total number of pixels
        - $\\hat{y}_i$ = predicted pixel value
        - $y_i$ = ground truth pixel value
        
        **Relationship to MSE:**
        - MSE penalizes large errors more heavily (squared)
        - MAE treats all errors uniformly (linear)
        - MSE typically results in lower error values
        
        **When to use:**
        - More robust to outliers than MSE
        - Better for medical or safety-critical applications
        - Often used alongside MSE
        """)
    
    st.markdown('<div class="section-header">Loss Functions</div>', unsafe_allow_html=True)
    
    with st.expander("📉 Training Loss Breakdown", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            ### MSE Loss During Training
            
            **Computation:**
            - Patches: {config.crop_size}×{config.crop_size}
            - Batch Size: {config.batch_size}
            - Loss per batch: MSE across all pixels
            
            **Expected Behavior:**
            - **First epochs:** Rapid decrease (high initial error)
            - **Middle epochs:** Gradual decrease
            - **Final epochs:** Plateau (convergence)
            
            **Red Flags:**
            ⚠️ Loss increases = LR too high
            ⚠️ Loss plateaus early = LR too low
            ⚠️ Loss oscillates = Instability
            """)
        
        with col2:
            st.code("""
# Example loss tracking
import matplotlib.pyplot as plt

epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, 'b-', label='Train Loss')
plt.plot(epochs, val_losses, 'r-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('SRCNN Training Progress')
plt.legend()
plt.yscale('log')  # Log scale often shows convergence better
plt.grid(True)
plt.show()
            """, language="python")
    
    st.markdown('<div class="section-header">Metrics Framework</div>', unsafe_allow_html=True)
    
    with st.expander("🔧 MetricTracker Implementation", expanded=True):
        st.code("""
class MetricTracker:
    '''Track training and evaluation metrics across epochs'''
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_psnr': [],
            'val_psnr': [],
            'val_ssim': [],
            'val_mae': [],
            'batch_time': [],
            'learning_rates': [],
        }
        self.best_psnr = 0.0
        self.best_epoch = 0
    
    def update(self, name: str, value: float):
        '''Update metric value'''
        if name in self.metrics:
            self.metrics[name].append(value)
    
    def get_summary(self) -> dict:
        '''Get latest metric summary'''
        return {k: v[-1] if v else 0 for k, v in self.metrics.items()}
    
    def is_best(self, psnr: float) -> bool:
        '''Check if current PSNR is best seen'''
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            self.best_epoch = len(self.metrics['val_psnr'])
            return True
        return False
    
    def save_plot(self, output_path: str):
        '''Save metric plots to file'''
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plot
        axes[0, 0].plot(self.metrics['train_loss'], label='Train')
        axes[0, 0].plot(self.metrics['val_loss'], label='Val')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].legend()
        
        # PSNR plot
        axes[0, 1].plot(self.metrics['train_psnr'], label='Train')
        axes[0, 1].plot(self.metrics['val_psnr'], label='Val')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].legend()
        
        # SSIM plot
        axes[1, 0].plot(self.metrics['val_ssim'])
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_ylim([0, 1])
        
        # Learning rate plot
        axes[1, 1].plot(self.metrics['learning_rates'])
        axes[1, 1].set_ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        """, language="python")
    
    st.markdown('<div class="section-header">Comparison Baseline</div>', unsafe_allow_html=True)
    
    with st.expander("📊 Typical Performance Benchmarks", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 2× Super-Resolution
            
            **Bicubic Baseline:**
            - PSNR: ~33 dB
            - SSIM: ~0.91
            
            **SRCNN:**
            - PSNR: ~35-36 dB
            - SSIM: ~0.93-0.94
            
            **Improvement:** ~2-3 dB
            """)
        
        with col2:
            st.markdown("""
            ### 4× Super-Resolution
            
            **Bicubic Baseline:**
            - PSNR: ~28 dB
            - SSIM: ~0.77
            
            **SRCNN:**
            - PSNR: ~30-31 dB
            - SSIM: ~0.84-0.85
            
            **Improvement:** ~2-3 dB
            """)
        
        with col3:
            st.markdown("""
            ### Dataset Specific
            
            **Varies by:**
            - Image content
            - Texture complexity
            - Noise levels
            - Compression artifacts
            
            **Always evaluate:**
            - On test set
            - Multiple runs
            - Compare with baselines
            """)
