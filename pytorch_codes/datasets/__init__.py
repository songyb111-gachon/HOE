"""
PyTorch Datasets for HOE Simulation
"""

from .hoe_dataset import (
    BaseHOEDataset,
    InverseDesignDataset,
    ForwardPhaseDataset,
    ForwardIntensityDataset,  # Alias for ForwardPhaseDataset (now for intensity)
    MetalineDataset,
    create_dataloaders
)

__all__ = [
    'BaseHOEDataset',
    'InverseDesignDataset',
    'ForwardPhaseDataset',
    'ForwardIntensityDataset',  # Alias
    'MetalineDataset',
    'create_dataloaders',
]

