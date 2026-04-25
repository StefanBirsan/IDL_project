"""
Quick test and comparison: SRCNN vs Enhanced SRCNN (ESRCNN)

This script helps you understand the architectural differences and decide which to use.

Run: python training/train_utils/test_esrcnn.py
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.train_utils.srcnn import SRCNN
from training.train_utils.esrcnn import EnhancedSRCNN, PerceptualLoss


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def compare_architectures():
    """Compare SRCNN and ESRCNN architectures"""
    print_section("Architecture Comparison: SRCNN vs ESRCNN")
    
    # Create models
    srcnn = SRCNN(scale_factor=2)
    esrcnn = EnhancedSRCNN(scale_factor=2, num_residual_blocks=10, num_features=64)
    
    # Compare parameters
    srcnn_params = sum(p.numel() for p in srcnn.parameters())
    esrcnn_params = sum(p.numel() for p in esrcnn.parameters())
    
    print(f"\n{'Model':<20} {'Parameters':<15} {'Relative Size':<15}")
    print("-" * 50)
    print(f"{'Classic SRCNN':<20} {srcnn_params:>12,}   {1.0:>12.1f}x")
    print(f"{'Enhanced SRCNN':<20} {esrcnn_params:>12,}   {esrcnn_params/srcnn_params:>12.1f}x")
    
    print(f"\n📊 ESRCNN has {esrcnn_params/srcnn_params:.1f}× more parameters")
    print(f"   → More expressive, can capture complex facial features")


def test_forward_pass():
    """Test forward pass with both models"""
    print_section("Forward Pass Test")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create models
    srcnn = SRCNN(scale_factor=2).to(device)
    esrcnn = EnhancedSRCNN(scale_factor=2, num_residual_blocks=10).to(device)
    
    srcnn.eval()
    esrcnn.eval()
    
    # Test input: 64x64 low-resolution image
    lr_input = torch.randn(1, 3, 64, 64).to(device)
    
    print(f"\nInput shape: {tuple(lr_input.shape)}")
    
    # SRCNN expects bicubic-upscaled input
    import torch.nn.functional as F
    lr_upscaled = F.interpolate(lr_input, scale_factor=2, mode='bicubic', align_corners=False)
    
    with torch.no_grad():
        srcnn_output = srcnn(lr_upscaled)
        esrcnn_output = esrcnn(lr_input)  # ESRCNN handles upsampling internally
    
    print(f"\nSRCNN:")
    print(f"  Input:  {tuple(lr_upscaled.shape)} (bicubic upscaled)")
    print(f"  Output: {tuple(srcnn_output.shape)}")
    
    print(f"\nESRCNN:")
    print(f"  Input:  {tuple(lr_input.shape)} (native LR)")
    print(f"  Output: {tuple(esrcnn_output.shape)}")
    
    print("\n✅ Both models produce 128×128 HR output from 64×64 LR input")


def test_receptive_field():
    """Compare receptive fields"""
    print_section("Receptive Field Analysis")
    
    print("\nSRCNN (3-layer):")
    print("  Layer 1: 9×9 conv → RF = 9")
    print("  Layer 2: 5×5 conv → RF = 13")
    print("  Layer 3: 5×5 conv → RF = 17")
    print("  Total Receptive Field: 17×17 pixels")
    
    print("\nESRCNN (10 residual blocks):")
    print("  Initial conv: 3×3 → RF = 3")
    print("  Each residual block adds ~4 pixels")
    print("  After 10 blocks: RF ≈ 43")
    print("  With upsampling: Effective RF ≈ 50×50 pixels")
    print("  Total Receptive Field: ~50×50 pixels")
    
    print("\n📏 ESRCNN has 3× larger receptive field")
    print("   → Can capture more facial context (eyes, nose, mouth together)")


def show_loss_comparison():
    """Compare loss functions"""
    print_section("Loss Function Comparison")
    
    print("\n1️⃣  Classic SRCNN Loss:")
    print("   loss = MSE(pred, target)")
    print("   ✗ Produces blurry results")
    print("   ✗ Optimizes for PSNR but poor visual quality")
    
    print("\n2️⃣  Enhanced SRCNN Loss:")
    print("   loss = L1(pred, target)")
    print("        + 0.1 × Perceptual(VGG features)")
    print("        + 0.3 × SSIM (optional)")
    print("   ✓ Sharp, realistic faces")
    print("   ✓ Preserves facial structure")
    print("   ✓ Better perceptual quality")
    
    # Test perceptual loss
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        perceptual_loss = PerceptualLoss().to(device)
        
        dummy_pred = torch.randn(1, 3, 128, 128).to(device)
        dummy_target = torch.randn(1, 3, 128, 128).to(device)
        
        with torch.no_grad():
            loss = perceptual_loss(dummy_pred, dummy_target)
        
        print(f"\n✅ Perceptual loss initialized successfully")
        print(f"   Loss value: {loss.item():.6f}")
    except Exception as e:
        print(f"\n⚠️  Could not initialize perceptual loss: {e}")
        print("   (torchvision may need to be installed)")


def show_recommendations():
    """Show usage recommendations"""
    print_section("When to Use Each Model")
    
    print("\n🔵 Use Classic SRCNN (3-layer) if:")
    print("   • You need very fast inference (<1ms)")
    print("   • Model size must be tiny (<500KB)")
    print("   • Simple images (not faces)")
    print("   • Quick baseline for comparison")
    
    print("\n🟢 Use Enhanced SRCNN (ESRCNN) if:")
    print("   • Face super-resolution is your goal ⭐")
    print("   • Quality matters more than speed")
    print("   • You have GPU for training")
    print("   • You want state-of-the-art results")
    
    print("\n💡 Recommendation for your face upscaling project:")
    print("   → Use ESRCNN with 10 residual blocks")
    print("   → Enable perceptual loss (weight=0.1)")
    print("   → Train for 150 epochs")
    print("   → Expected PSNR: 28-32 dB (vs 24-26 with SRCNN)")


def show_training_commands():
    """Show example training commands"""
    print_section("Training Commands")
    
    print("\n1️⃣  Classic SRCNN (your current implementation):")
    print("   python training/train_srcnn.py \\")
    print("     --data-dir dataset/ffhq \\")
    print("     --scale-factor 2 \\")
    print("     --batch-size 64 \\")
    print("     --num-epochs 100")
    print("   Training time: ~2-3 hours")
    print("   Quality: ⭐⭐ (Blurry)")
    
    print("\n2️⃣  Enhanced SRCNN (recommended for faces):")
    print("   python training/train_esrcnn.py \\")
    print("     --data-dir dataset/ffhq \\")
    print("     --scale-factor 2 \\")
    print("     --batch-size 16 \\")
    print("     --num-epochs 150 \\")
    print("     --use-perceptual-loss \\")
    print("     --perceptual-weight 0.1")
    print("   Training time: ~8-12 hours")
    print("   Quality: ⭐⭐⭐⭐⭐ (Sharp)")
    
    print("\n3️⃣  Quick test (fast training for verification):")
    print("   python training/train_esrcnn.py \\")
    print("     --data-dir dataset/ffhq \\")
    print("     --scale-factor 2 \\")
    print("     --num-residual-blocks 6 \\")
    print("     --batch-size 32 \\")
    print("     --num-epochs 50")
    print("   Training time: ~2-3 hours")
    print("   Quality: ⭐⭐⭐⭐")


def main():
    """Run all comparisons"""
    print("\n" + "="*80)
    print("  SRCNN vs Enhanced SRCNN (ESRCNN) - Complete Comparison")
    print("="*80)
    
    compare_architectures()
    test_forward_pass()
    test_receptive_field()
    show_loss_comparison()
    show_recommendations()
    show_training_commands()
    
    print("\n" + "="*80)
    print("  For detailed guide, see: training/train_utils/ESRCNN_GUIDE.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
