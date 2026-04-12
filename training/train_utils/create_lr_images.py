"""
Script to pre-generate low-resolution (LR) image pairs from high-resolution (HR) images.

This creates a dataset structure for training SRCNN on full-resolution images:
    dataset/
        hr_images/           # Original high-res images (1024x1024, etc)
            img1.jpg
            img2.jpg
            ...
        lr_images_2x/        # 2x downsampled + bicubic upsampled back (512x512 -> 1024x1024)
            img1.jpg
            img2.jpg
            ...
        lr_images_4x/        # 4x downsampled + bicubic upsampled back (256x256 -> 1024x1024)
            img1.jpg
            img2.jpg
            ...

Usage:
    # Generate 2x LR images
    python training/train_utils/create_lr_images.py \\
        --input-dir dataset/images1024x1024 \\
        --output-dir dataset \\
        --scale-factor 2 \\
        --quality 95
    
    # Generate 4x LR images
    python training/train_utils/create_lr_images.py \\
        --input-dir dataset/images1024x1024 \\
        --output-dir dataset \\
        --scale-factor 4 \\
        --quality 95
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torchvision.transforms.functional as TF
from PIL import Image


def setup_output_dirs(output_dir: Path, scale_factor: int):
    """Create output directory structure."""
    hr_dir = output_dir / "hr_images"
    lr_dir = output_dir / f"lr_images_{scale_factor}x"
    
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    
    return hr_dir, lr_dir


def find_images(input_dir: Path):
    """Find all image files in directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = []
    
    for ext in image_extensions:
        images.extend(input_dir.glob(f'*{ext}'))
    
    return sorted(images)


def create_lr_from_hr(
    hr_image_path: str,
    hr_output_path: str,
    lr_output_path: str,
    scale_factor: int = 2,
    quality: int = 95
) -> bool:
    """
    Create LR version of an HR image.
    
    Process:
        1. Load original image (HR)
        2. Save copy to HR directory
        3. Downscale by scale_factor using bicubic
        4. Upscale back to original size using bicubic (approximates SRCNN input)
        5. Save to LR directory
    
    Args:
        hr_image_path: Path to original HR image
        hr_output_path: Path to save HR copy
        lr_output_path: Path to save LR version
        scale_factor: Downsampling factor (2, 4, 8, etc)
        quality: JPEG quality for saving
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load HR image
        hr_image = Image.open(hr_image_path).convert('RGB')
        
        # Get dimensions
        w, h = hr_image.size
        
        # Make dimensions divisible by scale_factor
        w_new = (w // scale_factor) * scale_factor
        h_new = (h // scale_factor) * scale_factor
        
        # Crop to divisible size
        if w > w_new or h > h_new:
            hr_image = hr_image.crop((0, 0, w_new, h_new))
        
        # Save HR image
        if hr_output_path and not Path(hr_output_path).exists():
            hr_image.save(hr_output_path, 'JPEG', quality=quality)
        
        # Create LR version
        # Step 1: Downscale
        lr_w = w_new // scale_factor
        lr_h = h_new // scale_factor
        lr_image = hr_image.resize((lr_w, lr_h), Image.BICUBIC)
        
        # Step 2: Upscale back using bicubic (this is the SRCNN input)
        lr_upsampled = lr_image.resize((w_new, h_new), Image.BICUBIC)
        
        # Save LR image
        lr_upsampled.save(lr_output_path, 'JPEG', quality=quality)
        
        return True
        
    except Exception as e:
        print(f"Error processing {hr_image_path}: {e}")
        return False


def generate_lr_dataset(
    input_dir: str,
    output_dir: str,
    scale_factor: int = 2,
    quality: int = 95,
    max_images: int = None
):
    """
    Generate full dataset with LR images from HR images.
    
    Args:
        input_dir: Directory containing HR images
        output_dir: Root output directory
        scale_factor: Downsampling factor
        quality: JPEG quality
        max_images: Max images to process (None = all)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    # Setup output directories
    hr_dir, lr_dir = setup_output_dirs(output_path, scale_factor)
    
    # Find images
    images = find_images(input_path)
    
    if not images:
        raise ValueError(f"No images found in {input_dir}")
    
    if max_images:
        images = images[:max_images]
    
    print(f"Found {len(images)} images")
    print(f"Scale factor: {scale_factor}x")
    print(f"Output directories:")
    print(f"  HR: {hr_dir}")
    print(f"  LR: {lr_dir}")
    print()
    
    # Process images
    successful = 0
    failed = 0
    
    for img_path in tqdm(images, desc=f"Generating {scale_factor}x LR images"):
        img_name = img_path.name
        hr_output = hr_dir / img_name
        lr_output = lr_dir / img_name
        
        # Skip if LR already exists
        if lr_output.exists():
            successful += 1
            continue
        
        if create_lr_from_hr(
            str(img_path),
            str(hr_output),
            str(lr_output),
            scale_factor=scale_factor,
            quality=quality
        ):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessed: {successful} successful, {failed} failed")
    print(f"LR images saved to: {lr_dir}")
    print(f"HR images saved to: {hr_dir}")
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description='Generate low-resolution image pairs from HR images for SRCNN training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 2x LR images from dataset
  python training/train_utils/create_lr_images.py --input-dir dataset/images1024x1024 --output-dir dataset --scale-factor 2
  
  # Generate 4x LR images with custom quality
  python training/train_utils/create_lr_images.py --input-dir dataset/images1024x1024 --output-dir dataset --scale-factor 4 --quality 90
  
  # Process only first 100 images (testing)
  python training/train_utils/create_lr_images.py --input-dir dataset/images1024x1024 --output-dir dataset --scale-factor 2 --max-images 100
        """
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing HR images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory (will create hr_images/ and lr_images_Nx/ subdirs)')
    parser.add_argument('--scale-factor', type=int, default=2, choices=[2, 4, 8],
                        help='Downsampling scale factor (default: 2)')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for saving (1-100, default: 95)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')
    
    args = parser.parse_args()
    
    generate_lr_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scale_factor=args.scale_factor,
        quality=args.quality,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
