"""
PyTorch Dataset Classes for HOE Simulation
Converted from TensorFlow/Keras inputOutput.py

Datasets:
- InverseDesignDataset: For inverse design tasks
- MetalineDataset: For E-field estimation tasks
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from pathlib import Path


class BaseHOEDataset(Dataset):
    """Base dataset class for HOE simulation data"""
    
    def __init__(self, data_path, transform=None, normalize=True):
        """
        Args:
            data_path: Path to dataset directory
            transform: Optional transform to be applied
            normalize: If True, normalize images
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize = normalize
        self.image_names = []
        
    def _normalize_image(self, img):
        """Normalize image (channel-wise)"""
        if not self.normalize:
            return img
        
        normalized = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[2]):
            if img[:, :, i].std() != 0:
                normalized[:, :, i] = (img[:, :, i] - img[:, :, i].mean()) / img[:, :, i].std()
        return normalized
    
    def _list_png_files(self, directory):
        """List all PNG files in directory"""
        files = os.listdir(directory)
        return [f[:-4] for f in files if f.endswith('.png')]
    
    def _list_npy_files(self, directory):
        """List all NPY files in directory"""
        files = os.listdir(directory)
        return [f[:-4] for f in files if f.endswith('.npy')]
    
    def __len__(self):
        return len(self.image_names)


class InverseDesignDataset(BaseHOEDataset):
    """Dataset for inverse design tasks
    
    Data structure:
        data_path/
            inputs/
                image1.png (or .npy for phase maps)
                image2.png (or .npy)
                ...
            outputs/
                image1.txt (or .png for pillar patterns)
                image2.txt (or .png)
                ...
    """
    
    def __init__(self, data_path, input_extension='png', output_extension='txt', 
                 output_type='R', transform=None, normalize=True):
        """
        Args:
            data_path: Path to dataset directory
            input_extension: 'png' or 'npy'
            output_extension: 'txt' or 'png'
            output_type: 'R' for regression, 'BC' for binary classification, 
                        'MC' for multi-class classification
            transform: Optional transform
            normalize: If True, normalize images
        """
        super().__init__(data_path, transform, normalize)
        
        self.input_extension = input_extension
        self.output_extension = output_extension
        self.output_type = output_type
        
        # List all input images
        input_dir = self.data_path / 'inputs'
        if input_dir.exists():
            if input_extension == 'npy':
                self.image_names = self._list_npy_files(input_dir)
            else:
                self.image_names = self._list_png_files(input_dir)
        else:
            raise ValueError(f"Input directory not found: {input_dir}")
    
    def _read_output(self, image_name):
        """Read output file (txt or png)"""
        output_path = self.data_path / 'outputs' / f'{image_name}.{self.output_extension}'
        
        if self.output_extension == 'png':
            output = cv2.imread(str(output_path))
            output = output[:, :, 0]  # Take first channel
        else:  # txt
            with open(output_path, 'r') as f:
                data = f.read()
                if self.output_type in ['BC', 'MC']:
                    output = [list(map(int, line.split(' ')[:-1])) 
                             for line in data.split('\n')[:-1]]
                else:  # 'R'
                    output = [list(map(float, line.split(' ')[:-1])) 
                             for line in data.split('\n')[:-1]]
        
        return np.array(output, dtype=np.float32)
    
    def __getitem__(self, idx):
        """Get one sample"""
        image_name = self.image_names[idx]
        
        # Load input
        input_path = self.data_path / 'inputs' / f'{image_name}.{self.input_extension}'
        
        if self.input_extension == 'npy':
            # Load NPY file (e.g., phase map)
            image = np.load(input_path).astype(np.float32)
            
            # Ensure 2D -> 3D (H, W) -> (H, W, 1)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
        else:
            # Load PNG file
            image = cv2.imread(str(input_path))
            
            if image is None:
                raise ValueError(f"Failed to load image: {input_path}")
        
        # Normalize
        if self.normalize:
            if self.input_extension == 'npy':
                # Min-Max normalization for intensity maps to [0, 1]
                # Avoid division by zero
                img_min = image.min()
                img_max = image.max()
                if img_max > img_min:
                    image = (image - img_min) / (img_max - img_min)
                else:
                    image = np.zeros_like(image)
            elif self.input_extension == 'png':
                image = self._normalize_image(image)
        
        # Load output
        output = self._read_output(image_name)
        
        # Convert to torch tensors
        # Image: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Output: (H, W) -> (1, H, W) for single channel
        if output.ndim == 2:
            output = torch.from_numpy(output).unsqueeze(0).float()
        else:
            output = torch.from_numpy(output).permute(2, 0, 1).float()
        
        # Normalize output to [0, 1] if it's a pillar pattern (0-255)
        if self.output_extension == 'png':
            output = output / 255.0
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            output = self.transform(output)
        
        return {
            'image': image,
            'target': output,
            'name': image_name
        }


class ForwardPhaseDataset(BaseHOEDataset):
    """Dataset for forward EM intensity prediction
    
    Predicts EM near-field intensity map from random pillar pattern.
    Output: |Ex|² + |Ey|² + |Ez|²
    
    Data structure:
        data_path/
            inputs/
                sample_0000.png  (random pillar binary mask, grayscale 0-255)
                sample_0001.png
                ...
            outputs/
                sample_0000.npy  (EM intensity map, float32)
                sample_0001.npy
                ...
    """
    
    def __init__(self, data_path, transform=None, normalize=True):
        """
        Args:
            data_path: Path to dataset directory
            transform: Optional transform
            normalize: If True, normalize images
        """
        super().__init__(data_path, transform, normalize)
        
        # List all input images
        input_dir = self.data_path / 'inputs'
        if input_dir.exists():
            self.image_names = self._list_png_files(input_dir)
        else:
            raise ValueError(f"Input directory not found: {input_dir}")
    
    def __getitem__(self, idx):
        """Get one sample"""
        image_name = self.image_names[idx]
        
        # Load input mask (grayscale PNG, 0-255)
        input_path = self.data_path / 'inputs' / f'{image_name}.png'
        mask = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Failed to load mask: {input_path}")
        
        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        # Add channel dimension (H, W) -> (H, W, 1)
        mask = mask[:, :, np.newaxis]
        
        # Load output intensity map (.npy, float32)
        output_path = self.data_path / 'outputs' / f'{image_name}.npy'
        intensity_map = np.load(output_path).astype(np.float32)
        
        # Normalize intensity map to [0, 1] if requested
        if self.normalize:
            # Min-Max normalization
            img_min = intensity_map.min()
            img_max = intensity_map.max()
            if img_max > img_min:
                intensity_map = (intensity_map - img_min) / (img_max - img_min)
            else:
                intensity_map = np.zeros_like(intensity_map)
        
        # Convert to torch tensors
        # Mask: (H, W, 1) -> (1, H, W)
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        
        # Intensity map: (H, W) -> (1, H, W)
        if intensity_map.ndim == 2:
            intensity_map = torch.from_numpy(intensity_map).unsqueeze(0).float()
        else:
            intensity_map = torch.from_numpy(intensity_map).float()
        
        # Apply transforms
        if self.transform:
            mask = self.transform(mask)
            intensity_map = self.transform(intensity_map)
        
        return {
            'image': mask,
            'target': intensity_map,
            'name': image_name
        }


class MetalineDataset(BaseHOEDataset):
    """Dataset for E-field estimation tasks
    
    Data structure:
        data_path/
            inputs/
                image1.png
                image2.png
                ...
            outputs/
                image1_x1.txt
                image1_x2.txt
                image1_y1.txt
                image1_y2.txt
                image1_z1.txt
                image1_z2.txt
                ...
    """
    
    def __init__(self, data_path, mode='cascaded', 
                 transform=None, normalize=True):
        """
        Args:
            data_path: Path to dataset directory
            mode: 'single', 'multi', or 'cascaded'
            transform: Optional transform
            normalize: If True, normalize images
        """
        super().__init__(data_path, transform, normalize)
        
        self.mode = mode
        
        # List all input images
        input_dir = self.data_path / 'inputs'
        if input_dir.exists():
            self.image_names = self._list_png_files(input_dir)
        else:
            raise ValueError(f"Input directory not found: {input_dir}")
    
    def _read_efield_component(self, image_name, component):
        """Read one E-field component (x1, x2, y1, y2, z1, z2)"""
        output_path = self.data_path / 'outputs' / f'{image_name}_{component}.txt'
        
        with open(output_path, 'r') as f:
            data = f.read()
            values = [list(map(float, line.split(','))) 
                     for line in data.split('\n')[:-1]]
        
        return np.array(values, dtype=np.float32)
    
    def _normalize_component(self, component):
        """Normalize one component"""
        mean = component.mean()
        std = component.std()
        if std != 0:
            return (component - mean) / std
        return component
    
    def __getitem__(self, idx):
        """Get one sample"""
        image_name = self.image_names[idx]
        
        # Load input image
        input_path = self.data_path / 'inputs' / f'{image_name}.png'
        image = cv2.imread(str(input_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {input_path}")
        
        # Normalize and take first channel only (grayscale)
        image = self._normalize_image(image)
        image = image[:, :, 0:1]  # Keep channel dimension
        
        # Load E-field components
        components = []
        for comp_name in ['x1', 'x2', 'y1', 'y2', 'z1', 'z2']:
            comp = self._read_efield_component(image_name, comp_name)
            comp = self._normalize_component(comp)
            components.append(comp)
        
        # Convert to torch tensors
        # Image: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # E-field components
        if self.mode == 'single':
            # Stack all 6 components into one tensor (6, H, W)
            target = torch.from_numpy(np.stack(components, axis=0)).float()
            
            result = {
                'image': image,
                'target': target,
                'name': image_name
            }
            
        elif self.mode in ['multi', 'cascaded']:
            # Separate tensors for each component
            targets = {
                'x1': torch.from_numpy(components[0]).unsqueeze(0).float(),
                'x2': torch.from_numpy(components[1]).unsqueeze(0).float(),
                'y1': torch.from_numpy(components[2]).unsqueeze(0).float(),
                'y2': torch.from_numpy(components[3]).unsqueeze(0).float(),
                'z1': torch.from_numpy(components[4]).unsqueeze(0).float(),
                'z2': torch.from_numpy(components[5]).unsqueeze(0).float(),
            }
            
            # If cascaded mode, also load segmentation label (if exists)
            if self.mode == 'cascaded':
                seg_path = self.data_path / 'outputs' / f'{image_name}_label.png'
                if seg_path.exists():
                    seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
                    targets['segmentation'] = torch.from_numpy(seg).unsqueeze(0).float()
            
            result = {
                'image': image,
                'targets': targets,
                'name': image_name
            }
        
        # Apply transforms
        if self.transform:
            result['image'] = self.transform(result['image'])
        
        return result


def create_dataloaders(dataset_path, dataset_type='inverse', 
                       batch_size=8, num_workers=4,
                       train_split=0.8, val_split=0.1,
                       **dataset_kwargs):
    """Create train/val/test dataloaders
    
    Args:
        dataset_path: Path to dataset
        dataset_type: 'inverse', 'forward_phase', or 'metaline'
        batch_size: Batch size
        num_workers: Number of workers for data loading
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    if dataset_type == 'inverse':
        # Inverse: phase map (.npy) -> pillar pattern (.png)
        full_dataset = InverseDesignDataset(
            dataset_path, 
            input_extension='npy',
            output_extension='png',
            **dataset_kwargs
        )
    elif dataset_type == 'forward_phase':
        full_dataset = ForwardPhaseDataset(dataset_path, **dataset_kwargs)
    elif dataset_type == 'metaline':
        full_dataset = MetalineDataset(dataset_path, **dataset_kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Split into train/val/test
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_end = int(train_split * dataset_size)
    val_end = train_end + int(val_split * dataset_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    
    print(f"\n📊 Dataset split:")
    print(f"  • Total samples: {dataset_size}")
    print(f"  • Train: {len(train_indices)} ({len(train_indices)/dataset_size*100:.1f}%)")
    print(f"  • Validation: {len(val_indices)} ({len(val_indices)/dataset_size*100:.1f}%)")
    print(f"  • Test: {len(test_indices)} ({len(test_indices)/dataset_size*100:.1f}%)")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("\n=== Dataset classes created successfully ===\n")
    print("Usage examples:")
    print("\n1. Inverse Design Dataset:")
    print("   dataset = InverseDesignDataset('data/inverse')")
    print("   sample = dataset[0]")
    print("   image, target = sample['image'], sample['target']")
    
    print("\n2. Forward Phase Dataset:")
    print("   dataset = ForwardPhaseDataset('data/forward_phase')")
    print("   sample = dataset[0]")
    print("   mask = sample['image']  # Random pillar mask (1, H, W)")
    print("   phase_map = sample['target']  # Phase map (1, H, W)")
    
    print("\n3. Metaline Dataset:")
    print("   dataset = MetalineDataset('data/metaline', mode='cascaded')")
    print("   sample = dataset[0]")
    print("   image = sample['image']")
    print("   targets = sample['targets']  # Dict with x1, x2, y1, y2, z1, z2")
    
    print("\n4. Create DataLoaders:")
    print("   train_loader, val_loader, test_loader = create_dataloaders(")
    print("       'data/forward_phase', dataset_type='forward_phase',")
    print("       batch_size=8)")
    
    print("\n✓ Dataset module ready!")


# Alias for clarity (ForwardPhaseDataset is now for intensity)
ForwardIntensityDataset = ForwardPhaseDataset

