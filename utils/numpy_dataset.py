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
            x2/train_lr_patch/*.npy
            x2/train_dataloader.txt OR x2/eval_dataloader.txt (depending on chosen split)
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
        
        # Determine patch directories
        self.lr_dir = self.data_dir / f'{split}_lr_patch'
        self.hr_dir = self.data_dir / f'{split}_hr_patch'
        
        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR patch directory not found: {self.lr_dir}")
        if not self.hr_dir.exists():
            raise FileNotFoundError(f"HR patch directory not found: {self.hr_dir}")
        
        # Load image pairs
        self._load_image_pairs()
        
        if len(self._image_pairs) == 0:
            raise ValueError(f"No LR/HR pairs found in dataset.\nCheck directories: {self.lr_dir},"
                             f" {self.hr_dir}")

    def _load_image_pairs(self):
        """Load image pairs from manifest file"""

        self._image_pairs = []

        # open split manifest
        split_manifest_path = self.data_dir / 'dataload_filename' / f'{self.split}_dataloader.txt'
        if split_manifest_path.exists():
            with open(split_manifest_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        # tokenize line (is in CSV format)
                        tokenized = line.split(',')
                        hr_path = tokenized[0]
                        lr_path = tokenized[1]

                        # extract filenames
                        hr_filename = Path(hr_path).name
                        lr_filename = Path(lr_path).name

                        # check if files exist in patch directories
                        if (self.hr_dir / hr_filename).exists() and (self.lr_dir / lr_filename).exists():
                            self._image_pairs.append((lr_filename, hr_filename))
                        else:
                            print(f"[WARN] File pair specified in manifest not found: {lr_filename}, {hr_filename}")
    
    def __len__(self) -> int:
        return len(self._image_pairs)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Load a sample from the dataset
        """
        lr_name, hr_name = self._image_pairs[idx]
        lr_path = self.lr_dir / lr_name
        hr_path = self.hr_dir / hr_name

        # load LR and HR .npy files
        lr_data = np.load(lr_path, allow_pickle=True).item()
        hr_data = np.load(hr_path, allow_pickle=True).item()
        
        # load images from .npy files
        lr_image = lr_data['image'].astype(np.float32)
        hr_image = hr_data['image'].astype(np.float32)

        # Normalize images to [0, 1] range
        if self.normalize:
            lr_min, lr_max = lr_image.min(), lr_image.max()
            hr_min, hr_max = hr_image.min(), hr_image.max()
            
            # Avoid division by zero
            if lr_max > lr_min:
                lr_image = (lr_image - lr_min) / (lr_max - lr_min + 1e-8)
            else:
                lr_image = np.zeros_like(lr_image)
            
            if hr_max > hr_min:
                hr_image = (hr_image - hr_min) / (hr_max - hr_min + 1e-8)
            else:
                hr_image = np.zeros_like(hr_image)
        
        # Scale if needed
        if self.scale != 1.0:
            lr_image = lr_image * self.scale
            hr_image = hr_image * self.scale

        # extract masks for loss computation
        hr_mask = hr_data['mask'].astype(np.float32)
        lr_mask = lr_data['mask'].astype(np.float32)

        return {
            'lr_image': torch.from_numpy(lr_image).unsqueeze(0), # add channel dimension
            'hr_image': torch.from_numpy(hr_image).unsqueeze(0),
            'hr_mask': torch.from_numpy(hr_mask).unsqueeze(0),
            'lr_mask': torch.from_numpy(lr_mask).unsqueeze(0),
        }

    @staticmethod
    def get_hr_filename_from_lr(lr_filename: str) -> str:
        """
        Get the corresponding HR image filename from the LR image filename.
        """
        # Replace _hr_lr_patch_ with _hr_hr_patch_
        hr_filename = lr_filename.replace("_hr_lr_patch_", "_hr_hr_patch_")
        return hr_filename

    def get_random_sample(self) -> dict[str, torch.Tensor]:
        """
        Get a random LR/HR pair tensor sample from the dataset
        - LR image tensor shape: (1, C, H, W)
        - HR image tensor shape: (1, C, H, W)
        """

        # get random index
        idx = np.random.randint(0, len(self._image_pairs) - 1)

        return self.__getitem__(idx)


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
