"""
Dataset loader for SRCNN Face Super-Resolution training.
Handles bicubic upscaling, cropping, and augmentation.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict, Callable
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import sys


class FaceSuperResolutionDataset(Dataset):
    """
    Dataset for SRCNN training with face images.
    
    Supports two directory structures:
    
    Option 1 - Single directory (recommended):
        dataset/images1024x1024/
            *.jpg or *.png (all images)
    
    Option 2 - Split directories:
        dataset/
            train/
                *.jpg or *.png
            val/
                *.jpg or *.png
    
    For each image:
        1. Load original image (treated as HR)
        2. Downscale by scale_factor using bicubic
        3. Upscale back to original size using bicubic (this is input to SRCNN)
        4. Crop to crop_size x crop_size to avoid border effects
        5. Apply augmentations (rotation, flip, etc.)
    """
    # TODO: current dataset only has a training split
    #  so remove split param
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 scale_factor: int = 2,
                 crop_size: int = 33,
                 normalize: bool = True,
                 augmentation: Dict[str, bool] = None,
                 train_fraction: float = 0.8):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing images or root containing 'train'/'val' subdirectories
            split: 'train' or 'val' - used when splitting single directory
            scale_factor: Super-resolution upscaling factor (2, 4, 8, etc.)
            crop_size: Size of patches to crop from images (33x33 is standard)
            normalize: Whether to normalize images to [0, 1]
            augmentation: Dict of augmentation flags {'rotate': bool, 'flip': bool, 'brightness': bool}
            train_fraction: Fraction of images to use for training when splitting a single directory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.normalize = normalize
        self.train_fraction = train_fraction
        
        # Augmentation settings
        if augmentation is None:
            augmentation = {'rotate': True, 'flip': True, 'brightness': True}
        self.augmentation = augmentation
        
        # Find images - try to detect directory structure
        self.image_paths = self._find_images()

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found for split '{split}'")
        
        print(f"Loaded {len(self.image_paths)} images from {self.data_dir} ({split} split)")
    
    def _find_images(self):
        """
        Find images in directory structure.
        Supports both single directory and train/val split structures.
        
        Returns:
            List of image paths
        """
        images = sorted(set(self.data_dir.glob('*.png')))

        return images
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process an image pair.
        
        Returns:
            Dictionary with:
                - 'lr_image': Bicubic-upscaled LR image (input to SRCNN)
                - 'hr_image': Original HR image (ground truth)
        """
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            hr_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image on failure
            hr_image = Image.new('RGB', (self.crop_size * self.scale_factor,
                                         self.crop_size * self.scale_factor))
        
        # Convert to numpy array and float32
        hr_array = np.array(hr_image, dtype=np.float32)
        
        # Normalize to [0, 1]
        if self.normalize:
            hr_array = hr_array / 255.0
        
        # Create HR image tensor (C, H, W) format
        hr_tensor = torch.from_numpy(hr_array).permute(2, 0, 1)  # (3, H, W)
        
        # ========== Create LR image ==========
        # Step 1: Get the image size
        _, h, w = hr_tensor.shape
        
        # Make sure image size is divisible by scale factor
        h_lr = (h // self.scale_factor) * self.scale_factor
        w_lr = (w // self.scale_factor) * self.scale_factor
        
        # Crop to divisible size if needed
        if h > h_lr or w > w_lr:
            hr_tensor = hr_tensor[:, :h_lr, :w_lr]
        
        # Step 2: Downscale to get LR
        hr_pil = TF.to_pil_image(hr_tensor)
        lr_h, lr_w = h_lr // self.scale_factor, w_lr // self.scale_factor
        lr_pil = hr_pil.resize((lr_w, lr_h), Image.BICUBIC)
        
        # Step 3: Upscale back to original size using bicubic
        lr_upscaled_pil = lr_pil.resize((w_lr, h_lr), Image.BICUBIC)
        
        # Convert back to tensor
        lr_tensor = transforms.ToTensor()(lr_upscaled_pil)  # (3, H, W), normalized to [0, 1]
        
        # If original normalization was [0, 255], scale accordingly
        if not self.normalize:
            lr_tensor = lr_tensor * 255.0
        
        # ========== Crop to patch size ==========
        lr_tensor, hr_tensor = self._random_crop(lr_tensor, hr_tensor)
        
        # ========== Apply augmentations (training only) ==========
        if self.split == 'train':
            lr_tensor, hr_tensor = self._augment(lr_tensor, hr_tensor)
        
        return {
            'lr_image': lr_tensor,    # (3, crop_size, crop_size)
            'hr_image': hr_tensor,    # (3, crop_size, crop_size)
        }

    def get_random_sample(self, scale_factor: int) -> Dict[str, torch.Tensor]:
        """Get a random sample from the dataset"""

        # get random index
        idx = np.random.randint(0, len(self))
        chosen_path = self.image_paths[idx]

        # apply all downscaling steps except cropping 
        # 1) Load image
        try:
            hr_image = Image.open(chosen_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {chosen_path}: {e}")
            sys.exit(1)

        # 2) Convert to numpy array and float32
        hr_array = np.array(hr_image, dtype=np.float32)
        
        # 3) Normalize to [0, 1]
        if self.normalize:
            hr_array = hr_array / 255.0
        
        # 4) Create HR image tensor (C, H, W) format
        hr_tensor = torch.from_numpy(hr_array).permute(2, 0, 1)  # (3, H, W)
        
        # 5) Create LR image
        # 5.1) Get the image size
        _, h, w = hr_tensor.shape
        
        # 5.2) Make sure image size is divisible by scale factor
        h_lr = (h // scale_factor) * scale_factor
        w_lr = (w // scale_factor) * scale_factor
        
        # Crop to divisible size if needed
        if h > h_lr or w > w_lr:
            hr_tensor = hr_tensor[:, :h_lr, :w_lr]
        
        # 5.3) Downscale to get LR
        hr_pil = TF.to_pil_image(hr_tensor)
        lr_h, lr_w = h_lr // scale_factor, w_lr // scale_factor
        lr_pil = hr_pil.resize((lr_w, lr_h), Image.BICUBIC)
        
        # 5.4) Upscale back to original size using bicubic
        lr_upscaled_pil = lr_pil.resize((w_lr, h_lr), Image.BICUBIC)
        
        # Convert back to tensor
        lr_tensor = transforms.ToTensor()(lr_upscaled_pil)  # (3, H, W), normalized to [0, 1]

        return {
            'lr_image': lr_tensor,    # (3, H, W)
            'hr_image': hr_tensor,    # (3, H, W)
        }
    
    def _random_crop(self, lr: torch.Tensor, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly crop tensors to crop_size.
        Ensures valid crop region that doesn't go out of bounds.
        
        Args:
            lr: LR image tensor (3, H, W)
            hr: HR image tensor (3, H, W)
            
        Returns:
            Cropped (lr, hr) tensors
        """
        _, h, w = hr.shape
        
        # Crop only if image is larger than crop_size
        if h > self.crop_size or w > self.crop_size:
            if h > self.crop_size:
                top = np.random.randint(0, h - self.crop_size + 1)
            else:
                top = 0
            
            if w > self.crop_size:
                left = np.random.randint(0, w - self.crop_size + 1)
            else:
                left = 0
            
            lr = lr[:, top:top + self.crop_size, left:left + self.crop_size]
            hr = hr[:, top:top + self.crop_size, left:left + self.crop_size]
        else:
            # Pad if smaller than crop_size
            if h < self.crop_size:
                pad_h = self.crop_size - h
                lr = torch.nn.functional.pad(lr, (0, 0, pad_h // 2, pad_h - pad_h // 2), mode='reflect')
                hr = torch.nn.functional.pad(hr, (0, 0, pad_h // 2, pad_h - pad_h // 2), mode='reflect')
            
            if w < self.crop_size:
                pad_w = self.crop_size - w
                lr = torch.nn.functional.pad(lr, (pad_w // 2, pad_w - pad_w // 2, 0, 0), mode='reflect')
                hr = torch.nn.functional.pad(hr, (pad_w // 2, pad_w - pad_w // 2, 0, 0), mode='reflect')
        
        return lr, hr
    
    def _augment(self, lr: torch.Tensor, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply consistent augmentations to both LR and HR images.
        
        Args:
            lr: LR image tensor
            hr: HR image tensor
            
        Returns:
            Augmented (lr, hr) tensors
        """
        # Convert to PIL for augmentation
        lr_pil = TF.to_pil_image(lr)
        hr_pil = TF.to_pil_image(hr)
        
        # Random horizontal flip
        if self.augmentation.get('flip', True) and np.random.rand() > 0.5:
            lr_pil = TF.hflip(lr_pil)
            hr_pil = TF.hflip(hr_pil)
        
        # Random rotation
        if self.augmentation.get('rotate', True) and np.random.rand() > 0.7:
            angle = int(np.random.choice([90, 180, 270]))
            lr_pil = TF.rotate(lr_pil, angle)
            hr_pil = TF.rotate(hr_pil, angle)
        
        # Random brightness/contrast
        if self.augmentation.get('brightness', True) and np.random.rand() > 0.7:
            brightness_factor = float(np.random.uniform(0.8, 1.2))
            lr_pil = TF.adjust_brightness(lr_pil, brightness_factor)
            hr_pil = TF.adjust_brightness(hr_pil, brightness_factor)
        
        # Convert back to tensor
        lr = transforms.ToTensor()(lr_pil)
        hr = transforms.ToTensor()(hr_pil)
        
        return lr, hr


# ============ Convenience functions =============

def get_face_sr_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    scale_factor: int = 2,
    crop_size: int = 33,
    num_workers: int = 4,
    normalize: bool = True,
    train_fraction: float = 0.8,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders for SRCNN.
    
    Supports both single directory and split directory structures.
    
    Args:
        data_dir: Path to dataset directory (single folder with all images)
                 or root containing 'train'/'val' subdirectories
        batch_size: Batch size for training
        scale_factor: Super-resolution scale factor
        crop_size: Patch crop size
        num_workers: Number of data loading workers
        normalize: Whether to normalize images
        train_fraction: Fraction of images to use for training when using single directory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Training dataset
    train_dataset = FaceSuperResolutionDataset(
        data_dir=data_dir,
        split='train',
        scale_factor=scale_factor,
        crop_size=crop_size,
        normalize=normalize,
        augmentation={'rotate': True, 'flip': True, 'brightness': True},
        train_fraction=train_fraction
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation dataset - only create if train_fraction < 1.0 or val directory exists
    val_loader = None
    val_dir = Path(data_dir) / 'val'
    
    if val_dir.exists() or train_fraction < 1.0:
        val_dataset = FaceSuperResolutionDataset(
            data_dir=data_dir,
            split='val',
            scale_factor=scale_factor,
            crop_size=crop_size,
            normalize=normalize,
            augmentation={k: False for k in ['rotate', 'flip', 'brightness']},  # No augmentation for val
            train_fraction=train_fraction
        )
        
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
    
    return train_loader, val_loader
