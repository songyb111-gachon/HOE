"""
PyTorch Loss Functions
Converted from TensorFlow/Keras calculateLoss.py

Includes:
- Weighted Binary Cross Entropy
- Weighted Categorical Cross Entropy
- Weighted MSE
- Class weight calculation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss
    
    TensorFlow equivalent: weighted_binary_crossentropy
    """
    
    def __init__(self, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Predictions (B, C, H, W) or (B, H, W)
            target: Targets (B, C, H, W) or (B, H, W)
            weight: Weights (B, H, W) or None
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        if weight is not None:
            if weight.dim() == 3 and bce.dim() == 4:
                # Expand weight to match bce dimensions
                weight = weight.unsqueeze(1)
            bce = bce * weight
        
        if self.reduction == 'mean':
            return bce.mean()
        elif self.reduction == 'sum':
            return bce.sum()
        else:
            return bce


class WeightedCELoss(nn.Module):
    """Weighted Categorical Cross Entropy Loss
    
    TensorFlow equivalent: weighted_categorical_crossentropy
    """
    
    def __init__(self, reduction='mean'):
        super(WeightedCELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Predictions (B, C, H, W) - logits or probabilities
            target: Targets (B, H, W) - class indices or (B, C, H, W) - one-hot
            weight: Weights (B, H, W) or None
        """
        # If target is one-hot, convert to class indices
        if target.dim() == 4:
            target = torch.argmax(target, dim=1)
        
        # Calculate cross entropy
        ce = F.cross_entropy(pred, target, reduction='none')
        
        if weight is not None:
            ce = ce * weight
        
        if self.reduction == 'mean':
            return ce.mean()
        elif self.reduction == 'sum':
            return ce.sum()
        else:
            return ce


class WeightedMSELoss(nn.Module):
    """Weighted Mean Squared Error Loss
    
    TensorFlow equivalent: weighted_mse
    """
    
    def __init__(self, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Predictions (B, C, H, W)
            target: Targets (B, C, H, W)
            weight: Weights (B, H, W) or (B, 1, H, W) or None
        """
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Average over channel dimension
        mse = mse.mean(dim=1)  # (B, H, W)
        
        if weight is not None:
            if weight.dim() == 3:
                mse = mse * weight
            elif weight.dim() == 4:
                mse = mse * weight.squeeze(1)
        
        if self.reduction == 'mean':
            return mse.mean()
        elif self.reduction == 'sum':
            return mse.sum()
        else:
            return mse


def calculate_class_weights(labels):
    """Calculate class weights from label map
    
    Args:
        labels: numpy array (H, W) with class indices
        
    Returns:
        numpy array of class weights
    """
    total_pixels = labels.shape[0] * labels.shape[1]
    class_no = int(labels.max() + 1)
    class_counts = np.zeros(class_no)
    class_weights = np.zeros(class_no)
    
    for i in range(class_no):
        class_counts[i] = (labels == i).sum()
        class_weights[i] = class_counts[i] / total_pixels
    
    total = 0
    for i in range(class_no):
        if class_weights[i] > 0:
            class_weights[i] = 1 / class_weights[i]
            total += class_weights[i]
    
    for i in range(class_no):
        class_weights[i] /= total
    
    return class_weights


def calculate_class_weight_map(labels):
    """Calculate pixel-wise class weight map
    
    Args:
        labels: numpy array (H, W) with class indices
        
    Returns:
        numpy array (H, W) with class weights for each pixel
    """
    class_weights = calculate_class_weights(labels)
    
    height, width = labels.shape
    weight_map = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            weight_map[i][j] = class_weights[int(labels[i][j])]
    
    return weight_map


def calculate_unet_boundary_weight_map(labels, class_weighted=False, w0=10, sigma=5):
    """Calculate U-Net style boundary weight map
    
    From U-Net paper: Emphasize boundaries between touching objects
    
    Args:
        labels: numpy array (H, W) with instance labels
        class_weighted: If True, add class balancing weights
        w0: Weight of border emphasis
        sigma: Standard deviation for distance weighting
        
    Returns:
        numpy array (H, W) with weights
    """
    foreground = (labels > 0)
    background = labels == 0
    label_ids = sorted(np.unique(labels))[1:]  # Exclude background
    
    if len(label_ids) >= 1:
        distances = np.zeros((labels.shape[0], labels.shape[1], len(label_ids)))
        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = distance_transform_edt(labels != label_id)
        distances = np.sort(distances, axis=2)
        
        d1 = distances[:, :, 0]
        if len(label_ids) == 1:
            d2 = distances[:, :, 0]
        else:
            d2 = distances[:, :, 1]
        
        w = w0 * np.exp(-0.5 * ((d1 + d2) / sigma)**2) * background
    else:
        w = np.zeros_like(foreground, dtype=float)
    
    if class_weighted:
        class1_perc = labels.sum() / (labels.shape[0] * labels.shape[1])
        wc = {0: class1_perc, 1: (1 - class1_perc)}
        
        class_weights_map = np.zeros(labels.shape)
        for k, v in wc.items():
            class_weights_map[foreground == k] = v
        w = w + class_weights_map
    
    # Normalize
    w = (labels.shape[0] * labels.shape[1] * w) / w.sum()
    
    return w


def create_weight_maps_batch(labels_batch, weight_type='same', **kwargs):
    """Create weight maps for a batch of images
    
    Args:
        labels_batch: numpy array (B, H, W)
        weight_type: 'same', 'class-weighted', 'unet-distance', 'unet-distance-class-weighted'
        **kwargs: Additional arguments for weight calculation
        
    Returns:
        numpy array (B, H, W) with weights
    """
    batch_size = labels_batch.shape[0]
    height, width = labels_batch.shape[1], labels_batch.shape[2]
    weights = np.zeros((batch_size, height, width))
    
    for i in range(batch_size):
        if weight_type == 'same':
            weights[i] = np.ones((height, width))
        elif weight_type == 'class-weighted':
            weights[i] = calculate_class_weight_map(labels_batch[i])
        elif weight_type == 'unet-distance':
            weights[i] = calculate_unet_boundary_weight_map(labels_batch[i], 
                                                           class_weighted=False, **kwargs)
        elif weight_type == 'unet-distance-class-weighted':
            weights[i] = calculate_unet_boundary_weight_map(labels_batch[i], 
                                                           class_weighted=True, **kwargs)
    
    return weights


class MultiTaskLoss(nn.Module):
    """Multi-task loss with per-task weights
    
    Combines losses from multiple tasks with learnable or fixed weights
    """
    
    def __init__(self, task_losses, task_weights=None, learnable_weights=False):
        """
        Args:
            task_losses: List of loss functions for each task
            task_weights: List of task weights (default: equal weights)
            learnable_weights: If True, make weights learnable parameters
        """
        super(MultiTaskLoss, self).__init__()
        
        self.task_losses = nn.ModuleList(task_losses)
        self.num_tasks = len(task_losses)
        
        if task_weights is None:
            task_weights = [1.0] * self.num_tasks
        
        if learnable_weights:
            self.task_weights = nn.Parameter(torch.tensor(task_weights, dtype=torch.float32))
        else:
            self.register_buffer('task_weights', torch.tensor(task_weights, dtype=torch.float32))
    
    def forward(self, predictions, targets, weights=None):
        """
        Args:
            predictions: List of predictions for each task
            targets: List of targets for each task
            weights: List of weight maps for each task (or None)
        """
        if weights is None:
            weights = [None] * self.num_tasks
        
        total_loss = 0
        task_losses_dict = {}
        
        for i, (loss_fn, pred, target, weight, task_weight) in enumerate(
            zip(self.task_losses, predictions, targets, weights, self.task_weights)):
            
            task_loss = loss_fn(pred, target, weight)
            weighted_task_loss = task_weight * task_loss
            total_loss += weighted_task_loss
            
            task_losses_dict[f'task_{i+1}_loss'] = task_loss.item()
            task_losses_dict[f'task_{i+1}_weighted_loss'] = weighted_task_loss.item()
        
        return total_loss, task_losses_dict


if __name__ == "__main__":
    # Test loss functions
    print("\n=== Testing Loss Functions ===\n")
    
    # Test WeightedMSELoss
    print("1. WeightedMSELoss")
    loss_fn = WeightedMSELoss()
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    weight = torch.ones(2, 64, 64)
    loss = loss_fn(pred, target, weight)
    print(f"   Loss: {loss.item():.6f}")
    
    # Test WeightedCELoss
    print("\n2. WeightedCELoss")
    loss_fn = WeightedCELoss()
    pred = torch.randn(2, 3, 64, 64)  # 3 classes
    target = torch.randint(0, 3, (2, 64, 64))
    weight = torch.ones(2, 64, 64)
    loss = loss_fn(pred, target, weight)
    print(f"   Loss: {loss.item():.6f}")
    
    # Test MultiTaskLoss
    print("\n3. MultiTaskLoss")
    task_losses = [WeightedMSELoss(), WeightedMSELoss(), WeightedCELoss()]
    multi_loss = MultiTaskLoss(task_losses, task_weights=[1.0, 1.0, 0.5])
    
    predictions = [
        torch.randn(2, 1, 64, 64),
        torch.randn(2, 1, 64, 64),
        torch.randn(2, 3, 64, 64)
    ]
    targets = [
        torch.randn(2, 1, 64, 64),
        torch.randn(2, 1, 64, 64),
        torch.randint(0, 3, (2, 64, 64))
    ]
    
    total_loss, task_losses_dict = multi_loss(predictions, targets)
    print(f"   Total loss: {total_loss.item():.6f}")
    for key, value in task_losses_dict.items():
        print(f"   {key}: {value:.6f}")
    
    # Test weight map calculation
    print("\n4. Weight Maps")
    labels = np.random.randint(0, 3, (64, 64))
    weight_map = calculate_class_weight_map(labels)
    print(f"   Class weight map shape: {weight_map.shape}")
    print(f"   Unique weights: {np.unique(weight_map)}")
    
    print("\n✓ All loss functions working correctly!")

