"""
Image processing utilities
"""
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, Tuple
import argparse
import sys


def downscale_image(image_path: Union[str, Path], scale_factor: int = 2) -> Image.Image:
    """
    Load an image and downscale it by a given factor.
    
    Args:
        image_path: Path to the image file
        scale_factor: Factor to downscale by (default 2x)
    
    Returns:
        PIL Image object downscaled by scale_factor
    
    Example:
        >>> downscaled = downscale_image('path/to/image.jpg', scale_factor=2)
        >>> downscaled.save('output.jpg')
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Get original dimensions
        original_width, original_height = image.size
        
        # Calculate new dimensions
        new_width = original_width // scale_factor
        new_height = original_height // scale_factor
        
        # Downscale using high-quality resampling
        downscaled = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
        
        return downscaled
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise Exception(f"Error downscaling image: {e}")


def downscale_image_to_numpy(image_path: Union[str, Path], scale_factor: int = 2) -> np.ndarray:
    """
    Load an image, downscale it, and return as numpy array.
    
    Args:
        image_path: Path to the image file
        scale_factor: Factor to downscale by (default 2x)
    
    Returns:
        Numpy array (H, W, C) for RGB/BGR images or (H, W) for grayscale
    
    Example:
        >>> img_array = downscale_image_to_numpy('path/to/image.jpg')
        >>> print(img_array.shape)
    """
    downscaled = downscale_image(image_path, scale_factor)
    return np.array(downscaled)


def get_downscaled_dimensions(image_path: Union[str, Path], scale_factor: int = 2) -> Tuple[int, int]:
    """
    Get the dimensions that an image would have after downscaling.
    
    Args:
        image_path: Path to the image file
        scale_factor: Factor to downscale by (default 2x)
    
    Returns:
        Tuple of (width, height) after downscaling
    
    Example:
        >>> width, height = get_downscaled_dimensions('path/to/image.jpg')
        >>> print(f"Downscaled size: {width}x{height}")
    """
    image = Image.open(image_path)
    original_width, original_height = image.size
    
    new_width = original_width // scale_factor
    new_height = original_height // scale_factor
    
    return new_width, new_height


__all__ = [
    'downscale_image',
    'downscale_image_to_numpy',
    'get_downscaled_dimensions',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Downscale an image by a given factor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python image_utils.py input.jpg                    # Downscale by 2x (default)
  python image_utils.py input.jpg -s 4               # Downscale by 4x
  python image_utils.py input.jpg -o downscaled.jpg # Save to specific output file
  python image_utils.py input.jpg -s 2 -o output.jpg
        '''
    )
    
    parser.add_argument(
        'input',
        help='Path to input image'
    )
    parser.add_argument(
        '-s', '--scale',
        type=int,
        default=2,
        help='Scale factor (default: 2)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Path to save downscaled image (default: input_2x.ext)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load and downscale
        print(f"Loading image: {args.input}")
        downscaled = downscale_image(args.input, scale_factor=args.scale)
        
        # Get output path
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input)
            output_path = input_path.parent / f"{input_path.stem}_{args.scale}x{input_path.suffix}"
        
        # Save
        downscaled.save(output_path)
        
        print(f"✅ Done!")
        print(f"Input size:  {downscaled.size[0] * args.scale} × {downscaled.size[1] * args.scale}")
        print(f"Output size: {downscaled.size[0]} × {downscaled.size[1]}")
        print(f"Saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
