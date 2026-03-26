"""
Dataset loader for numpy-based astronomical images
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Callable


class NumpyAstronomicalDataset(Dataset):
    """
    Dataset for loading astronomical images stored as numpy files
    
    Assumes directory structure:
        data/
            x2/train_hr_patch/*.npy
            x2/eval_dataloader.txt (optional, contains filenames for evaluation)
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 img_size: int = 64,
                 normalize: bool = True,
                 preprocessing_fn: Optional[Callable] = None,
                 scale: float = 1.0):
        """
        Args:
            data_dir: Root directory of dataset
            split: 'train' or 'eval'
            img_size: Target image size (will be center cropped if needed)
            normalize: Whether to normalize images
            preprocessing_fn: Optional preprocessing function to apply
            scale: Scale factor for images (default 1.0 for no scaling)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.normalize = normalize
        self.preprocessing_fn = preprocessing_fn
        self.scale = scale
        
        # Determine patch directory
        self.patch_dir = self.data_dir / 'x2' / 'train_hr_patch'
        
        if not self.patch_dir.exists():
            raise FileNotFoundError(f"Patch directory not found: {self.patch_dir}")
        
        # Load file list
        self._load_file_list()
        
        if len(self.file_list) == 0:
            raise ValueError(f"No numpy files found in {self.patch_dir}")
        
        print(f"Loaded {len(self.file_list)} images for {split} split")
    
    def _load_file_list(self):
        """Load list of numpy files"""
        self.file_list = sorted([
            f for f in os.listdir(self.patch_dir)
            if f.endswith('.npy')
        ])
        
        # If eval list is provided, filter accordingly
        eval_list_path = self.data_dir / 'x2' / 'dataload_filename' / f'{self.split}_dataloader.txt'
        if eval_list_path.exists():
            with open(eval_list_path, 'r') as f:
                allowed_files = {line.strip() for line in f}
            self.file_list = [f for f in self.file_list if f in allowed_files]
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Load a sample
        
        Returns:
            image: Tensor of shape (1, H, W) or (C, H, W)
            metadata: Dict with filename and other info
        """
        file_path = self.patch_dir / self.file_list[idx]
        
        # Load numpy array
        image = np.load(file_path).astype(np.float32)
        
        # Handle different array shapes
        if image.ndim == 2:
            # Grayscale image: add channel dimension
            image = image[np.newaxis, ...]
        elif image.ndim == 3:
            # Already has channel dimension
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Scale image
        if self.scale != 1.0:
            image = image * self.scale
        
        # Center crop to target size if needed
        image = self._center_crop(image, self.img_size)
        
        # Normalize
        if self.normalize:
            image = self._normalize(image)
        
        # Apply preprocessing if provided
        if self.preprocessing_fn is not None:
            image = self.preprocessing_fn(image)
        
        # Convert to torch
        image_tensor = torch.from_numpy(image).float()
        
        metadata = {
            'filename': self.file_list[idx],
            'shape': image.shape,
        }
        
        return image_tensor, metadata
    
    @staticmethod
    def _center_crop(image: np.ndarray, target_size: int) -> np.ndarray:
        """Center crop image to target size"""
        _, h, w = image.shape
        
        if h == target_size and w == target_size:
            return image
        
        if h < target_size or w < target_size:
            # Pad if needed
            pad_h = max(0, target_size - h)
            pad_w = max(0, target_size - w)
            image = np.pad(
                image,
                ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                mode='constant',
                constant_values=0
            )
        
        # Center crop
        _, h, w = image.shape
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        
        return image[:, start_h:start_h + target_size, start_w:start_w + target_size]
    
    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """Normalize image to [-1, 1] range"""
        min_val = image.min()
        max_val = image.max()
        
        if max_val - min_val > 0:
            image = 2.0 * (image - min_val) / (max_val - min_val) - 1.0
        else:
            image = image - min_val
        
        return image


class DataLoaderFactory:
    """Factory for creating data loaders"""
    
    @staticmethod
    def create_train_loader(
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 64,
        shuffle: bool = True,
        **kwargs
    ) -> DataLoader:
        """Create training data loader"""
        dataset = NumpyAstronomicalDataset(
            data_dir=data_dir,
            split='train',
            img_size=img_size,
            **kwargs
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    @staticmethod
    def create_eval_loader(
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 64,
        **kwargs
    ) -> DataLoader:
        """Create evaluation data loader"""
        dataset = NumpyAstronomicalDataset(
            data_dir=data_dir,
            split='eval',
            img_size=img_size,
            **kwargs
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


# Convenience functions
def get_dataset(data_dir: str, split: str = 'train', **kwargs) -> NumpyAstronomicalDataset:
    """Get a dataset"""
    return NumpyAstronomicalDataset(data_dir=data_dir, split=split, **kwargs)


def get_dataloader(data_dir: str,
                   split: str = 'train',
                   batch_size: int = 32,
                   num_workers: int = 4,
                   **kwargs) -> DataLoader:
    """Get a data loader"""
    dataset = get_dataset(data_dir, split=split, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
    )
