"""
Dataset loader for SRCNN full-resolution image training.

Instead of cropping patches, this dataset loads complete LR and HR image pairs.
This is suitable for training SRCNN on whole images at their native resolution.

Directory structure:
    dataset/
        hr_images/
            img1.jpg
            img2.jpg
        lr_images_2x/  (or lr_images_4x, etc)
            img1.jpg
            img2.jpg
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class FullResolutionFaceSRDataset(Dataset):
    """
    Full-resolution SRCNN dataset.
    
    Loads complete LR/HR image pairs without cropping patches.
    
    Expected directory structure:
        data_root/
            hr_images/
                img1.jpg
                img2.jpg
                ...
            lr_images_2x/  (or lr_images_4x, etc for different scales)
                img1.jpg       (same filename as HR pair)
                img2.jpg
                ...
    """
    
    def __init__(self,
                 data_root: str,
                 scale_factor: int = 2,
                 split: str = 'train',
                 normalize: bool = True,
                 augmentation: Dict[str, bool] = None,
                 train_fraction: float = 0.8):
        """
        Initialize full-resolution dataset.
        
        Args:
            data_root: Root directory containing hr_images/ and lr_images_Nx/ subdirectories
            scale_factor: Super-resolution scale factor (2, 4, 8)
            split: 'train' or 'val'
            normalize: Normalize images to [0, 1]
            augmentation: Dictionary of augmentation flags
            train_fraction: Fraction of images for training split
        """
        self.data_root = Path(data_root)
        self.scale_factor = scale_factor
        self.split = split
        self.normalize = normalize
        self.train_fraction = train_fraction
        
        # Setup paths
        self.hr_dir = self.data_root / 'hr_images'
        self.lr_dir = self.data_root / f'lr_images_{scale_factor}x'
        
        # Validate directories exist
        if not self.hr_dir.exists():
            raise ValueError(f"HR directory not found: {self.hr_dir}")
        if not self.lr_dir.exists():
            raise ValueError(f"LR directory not found: {self.lr_dir}")
        
        # Augmentation settings
        if augmentation is None:
            augmentation = {'rotate': False, 'flip': True, 'brightness': False}
        self.augmentation = augmentation
        
        # Find images
        self.image_paths = self._find_paired_images()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No paired images found")
        
        # Split into train/val
        if split in ['train', 'val']:
            np.random.seed(42)
            all_images = sorted(self.image_paths)
            num_train = int(len(all_images) * train_fraction)
            
            if split == 'train':
                self.image_paths = all_images[:num_train]
            else:
                self.image_paths = all_images[num_train:]
        
        print(f"Loaded {len(self.image_paths)} image pairs ({split} split)")
        if len(self.image_paths) > 0:
            # Load first image to show dimensions
            try:
                sample_hr = Image.open(self.hr_dir / self.image_paths[0]).convert('RGB')
                print(f"Image resolution: {sample_hr.size[0]}x{sample_hr.size[1]} pixels")
            except:
                pass
    
    def _find_paired_images(self) -> List[str]:
        """
        Find images that exist in both HR and LR directories.
        Returns filename list (same names assumed for LR/HR pairs).
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        # Get all HR images
        hr_images = set()
        for ext in image_extensions:
            hr_images.update(f.name for f in self.hr_dir.glob(f'*{ext}'))
        
        # Get all LR images
        lr_images = set()
        for ext in image_extensions:
            lr_images.update(f.name for f in self.lr_dir.glob(f'*{ext}'))
        
        # Find intersection (paired images)
        paired = sorted(hr_images & lr_images)
        
        if not paired:
            print(f"Warning: No paired images found")
            print(f"  HR images in {self.hr_dir}: {len(hr_images)}")
            print(f"  LR images in {self.lr_dir}: {len(lr_images)}")
        
        return paired
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load LR and HR image pair.
        
        Returns:
            Dictionary with:
                - 'lr_image': LR image (3, H, W)
                - 'hr_image': HR image (3, H, W)
                - 'filename': Image filename
        """
        filename = self.image_paths[idx]
        
        try:
            # Load images
            hr_image = Image.open(self.hr_dir / filename).convert('RGB')
            lr_image = Image.open(self.lr_dir / filename).convert('RGB')
            
            # Ensure same dimensions
            if hr_image.size != lr_image.size:
                print(f"Warning: Size mismatch for {filename}")
                print(f"  HR: {hr_image.size}, LR: {lr_image.size}")
                # Resize LR to match HR
                lr_image = lr_image.resize(hr_image.size, Image.BICUBIC)
            
            # Convert to tensors (H, W, 3) -> (3, H, W) with [0, 1]
            hr_tensor = transforms.ToTensor()(hr_image)  # Already normalized by ToTensor
            lr_tensor = transforms.ToTensor()(lr_image)
            
            # Apply augmentations (training only)
            if self.split == 'train':
                hr_tensor, lr_tensor = self._augment(hr_tensor, lr_tensor)
            
            return {
                'lr_image': lr_tensor,
                'hr_image': hr_tensor,
                'filename': filename
            }
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            # Return dummy pair
            dummy = torch.zeros((3, 512, 512))
            return {
                'lr_image': dummy,
                'hr_image': dummy,
                'filename': filename
            }
    
    def _augment(self, hr_tensor: torch.Tensor, lr_tensor: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply consistent augmentations to LR and HR.
        Used for training only (split == 'train').
        """
        # Convert to PIL for augmentation
        hr_pil = TF.to_pil_image(hr_tensor)
        lr_pil = TF.to_pil_image(lr_tensor)
        
        # Random horizontal flip (most common for faces)
        if self.augmentation.get('flip', True) and np.random.rand() > 0.5:
            hr_pil = TF.hflip(hr_pil)
            lr_pil = TF.hflip(lr_pil)
        
        # Random rotation (less common for full images, disabled by default)
        if self.augmentation.get('rotate', False) and np.random.rand() > 0.8:
            angle = int(np.random.choice([90, 180, 270]))
            hr_pil = TF.rotate(hr_pil, angle)
            lr_pil = TF.rotate(lr_pil, angle)
        
        # Random brightness/contrast
        if self.augmentation.get('brightness', False) and np.random.rand() > 0.8:
            brightness_factor = float(np.random.uniform(0.85, 1.15))
            hr_pil = TF.adjust_brightness(hr_pil, brightness_factor)
            lr_pil = TF.adjust_brightness(lr_pil, brightness_factor)
        
        # Convert back to tensor
        hr_tensor = transforms.ToTensor()(hr_pil)
        lr_tensor = transforms.ToTensor()(lr_pil)
        
        return hr_tensor, lr_tensor


def get_full_resolution_dataloaders(
    data_root: str,
    batch_size: int = 4,  # Smaller batch size for full images
    scale_factor: int = 2,
    num_workers: int = 2,  # Fewer workers for memory efficiency
    train_fraction: float = 0.8,
    augmentation_train: Dict[str, bool] = None,
    augmentation_val: Dict[str, bool] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create dataloaders for full-resolution SRCNN training.
    
    Args:
        data_root: Root directory with hr_images/ and lr_images_Nx/
        batch_size: Batch size (use smaller for full images due to memory)
        scale_factor: Super-resolution factor
        num_workers: Data loading workers
        train_fraction: Fraction for training split
        augmentation_train: Augmentation dict for training
        augmentation_val: Augmentation dict for validation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if augmentation_train is None:
        augmentation_train = {'flip': True, 'rotate': False, 'brightness': False}
    
    if augmentation_val is None:
        augmentation_val = {'flip': False, 'rotate': False, 'brightness': False}
    
    # Training dataset
    train_dataset = FullResolutionFaceSRDataset(
        data_root=data_root,
        scale_factor=scale_factor,
        split='train',
        augmentation=augmentation_train,
        train_fraction=train_fraction
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation dataset
    val_loader = None
    
    if train_fraction < 1.0:
        val_dataset = FullResolutionFaceSRDataset(
            data_root=data_root,
            scale_factor=scale_factor,
            split='val',
            augmentation=augmentation_val,
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


if __name__ == '__main__':
    # Test the dataset
    print("Testing FullResolutionFaceSRDataset...")
    
    data_root = 'dataset'  # Change to your data root
    
    train_loader, val_loader = get_full_resolution_dataloaders(
        data_root=data_root,
        batch_size=2,
        scale_factor=2,
        train_fraction=0.8
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader) if val_loader else 'None'}")
    
    # Show first batch
    batch = next(iter(train_loader))
    print(f"Batch shapes:")
    print(f"  LR shape: {batch['lr_image'].shape}")
    print(f"  HR shape: {batch['hr_image'].shape}")
    print(f"  Filenames: {batch['filename']}")
