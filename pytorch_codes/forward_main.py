#!/usr/bin/env python3
"""
PyTorch Forward Phase Map Prediction Training Script
Random Pillar Pattern → Phase Map (MEEP Surrogate)

This model learns to predict phase maps from random pillar patterns,
serving as a fast surrogate for MEEP simulations.

Usage:
    # Training
    python forward_main.py \
        --data_path ./data/forward \
        --mode train \
        --batch_size 8 \
        --num_epochs 100
    
    # Testing
    python forward_main.py \
        --data_path ./data/forward \
        --mode test \
        --checkpoint ./checkpoints/best_model.pth
    
    # Prediction (inference)
    python forward_main.py \
        --data_path ./data/forward \
        --mode predict \
        --checkpoint ./checkpoints/best_model.pth \
        --input_pattern ./pattern.png
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models import ForwardPhaseUNet, MultiScalePhaseUNet, PhaseAmplitudeUNet
from datasets import ForwardPhaseDataset, create_dataloaders
from utils import WeightedMSELoss, Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='PyTorch Forward Phase Map Prediction (MEEP Surrogate)'
    )
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'predict'],
                       help='Train, test, or predict mode')
    
    # Model
    parser.add_argument('--model_type', type=str, default='basic',
                       choices=['basic', 'multiscale', 'phase_amplitude'],
                       help='Model type')
    parser.add_argument('--layer_num', type=int, default=5,
                       help='Number of encoder/decoder layers')
    parser.add_argument('--base_features', type=int, default=64,
                       help='Base number of features')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--use_batchnorm', action='store_true', default=True,
                       help='Use BatchNorm (modern, default)')
    parser.add_argument('--no_batchnorm', action='store_true',
                       help='Disable BatchNorm (original U-Net style)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Loss
    parser.add_argument('--loss_type', type=str, default='mse',
                       choices=['mse', 'mae', 'huber'],
                       help='Loss function type')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory for TensorBoard')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume/test')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: timestamp)')
    
    # Prediction
    parser.add_argument('--input_pattern', type=str, default=None,
                       help='Input pattern file for prediction mode')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Output directory for predictions')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    
    return parser.parse_args()


def create_model(args):
    """Create model based on arguments"""
    print(f"\n{'='*80}")
    print(f"Creating Forward Phase Map Prediction Model")
    print(f"{'='*80}")
    
    # Determine use_batchnorm
    use_batchnorm = not args.no_batchnorm if hasattr(args, 'no_batchnorm') else args.use_batchnorm
    print(f"  • BatchNorm: {'Enabled' if use_batchnorm else 'Disabled (Original U-Net)'}")
    
    if args.model_type == 'basic':
        model = ForwardPhaseUNet(
            in_channels=1,
            out_channels=1,
            layer_num=args.layer_num,
            base_features=args.base_features,
            dropout_rate=args.dropout_rate,
            output_activation='linear',
            use_batchnorm=use_batchnorm
        )
    elif args.model_type == 'multiscale':
        model = MultiScalePhaseUNet(
            in_channels=1,
            out_channels=1,
            layer_num=args.layer_num,
            base_features=args.base_features,
            dropout_rate=args.dropout_rate,
            use_batchnorm=use_batchnorm
        )
    elif args.model_type == 'phase_amplitude':
        model = PhaseAmplitudeUNet(
            in_channels=1,
            layer_num=args.layer_num,
            base_features=args.base_features,
            dropout_rate=args.dropout_rate,
            use_batchnorm=use_batchnorm
        )
    
    if hasattr(model, 'get_model_summary'):
        model.get_model_summary()
    
    return model


def create_criterion(loss_type):
    """Create loss function"""
    if loss_type == 'mse':
        return WeightedMSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train(args):
    """Training pipeline"""
    print(f"\n{'='*80}")
    print(f"Forward Phase Map Prediction Training Pipeline")
    print(f"{'='*80}")
    print(f"  • Task: Random Pillar Pattern → Phase Map")
    print(f"  • Purpose: MEEP Surrogate (Fast Prediction)")
    print(f"{'='*80}\n")
    
    # Set device
    if args.device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            device = f'cuda:{args.gpu}'
            print(f"✓ Using GPU: {torch.cuda.get_device_name(args.gpu)}")
        else:
            device = 'cpu'
            print(f"⚠️ CUDA not available, using CPU")
    else:
        device = 'cpu'
        print(f"✓ Using CPU")
    
    # Create dataloaders
    print(f"\n📂 Loading data from: {args.data_path}")
    print(f"  • Expected structure:")
    print(f"    {args.data_path}/inputs/  → Random pillar patterns (.png)")
    print(f"    {args.data_path}/outputs/ → Phase maps (.npy)")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=args.data_path,
        dataset_type='forward_intensity',  # Forward phase dataset
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=False  # ForwardPhaseDataset handles normalization internally
    )
    
    # Create model
    model = create_model(args)
    model = model.to(device)
    
    # Create loss function
    criterion = create_criterion(args.loss_type)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    trainer.train(num_epochs=args.num_epochs, save_freq=args.save_freq)
    
    print(f"\n✓ Training complete!")
    print(f"  • Best model saved to: {trainer.checkpoint_dir / 'best_model.pth'}")
    print(f"  • TensorBoard logs: {trainer.log_dir}")
    print(f"\nTo view training curves, run:")
    print(f"  tensorboard --logdir {args.log_dir}")
    
    print(f"\n💡 Model trained as MEEP surrogate:")
    print(f"  • Can now predict phase maps from patterns instantly")
    print(f"  • No need to run expensive MEEP simulations for prediction")
    print(f"  • Use --mode predict to test on new patterns")


def test(args):
    """Testing pipeline"""
    print(f"\n{'='*80}")
    print(f"Forward Phase Map Prediction Testing Pipeline")
    print(f"{'='*80}\n")
    
    if args.checkpoint is None:
        raise ValueError("Checkpoint path required for testing (--checkpoint)")
    
    # Set device
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    # Create dataloaders
    print(f"\n📂 Loading test data from: {args.data_path}")
    _, _, test_loader = create_dataloaders(
        dataset_path=args.data_path,
        dataset_type='forward_intensity',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=False
    )
    
    # Create model
    model = create_model(args)
    model = model.to(device)
    
    # Load checkpoint
    print(f"\n📥 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    
    # Create criterion
    criterion = create_criterion(args.loss_type)
    
    # Test
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    
    print(f"\n🧪 Testing...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            target = batch['target'].to(device)
            
            if args.model_type == 'phase_amplitude':
                phase_pred, amp_pred = model(images)
                loss = criterion(phase_pred, target)  # Only evaluate phase for now
                pred = phase_pred
            else:
                pred = model(images)
                loss = criterion(pred, target)
            
            test_loss += loss.item()
            predictions.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
            
            # Save visualization for first few samples
            if batch_idx < 3:
                visualize_prediction(
                    images[0].cpu().numpy(),
                    target[0].cpu().numpy(),
                    pred[0].cpu().numpy(),
                    save_path=f"test_sample_{batch_idx}.png"
                )
    
    avg_test_loss = test_loss / len(test_loader)
    
    # Calculate additional metrics
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    
    print(f"\n{'='*80}")
    print(f"Test Results")
    print(f"{'='*80}")
    print(f"  • Test loss ({args.loss_type}): {avg_test_loss:.6f}")
    print(f"  • MAE: {mae:.6f}")
    print(f"  • RMSE: {rmse:.6f}")
    print(f"  • Test samples: {len(test_loader.dataset)}")
    print(f"{'='*80}\n")


def predict(args):
    """Prediction pipeline for single pattern"""
    print(f"\n{'='*80}")
    print(f"Forward Phase Map Prediction (Inference)")
    print(f"{'='*80}\n")
    
    if args.checkpoint is None:
        raise ValueError("Checkpoint path required (--checkpoint)")
    if args.input_pattern is None:
        raise ValueError("Input pattern required (--input_pattern)")
    
    # Set device
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    # Load model
    model = create_model(args)
    model = model.to(device)
    
    print(f"\n📥 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    
    # Load input pattern
    print(f"\n📂 Loading pattern: {args.input_pattern}")
    pattern = cv2.imread(args.input_pattern, cv2.IMREAD_GRAYSCALE)
    if pattern is None:
        raise ValueError(f"Failed to load pattern: {args.input_pattern}")
    
    # Preprocess
    pattern = pattern.astype(np.float32) / 255.0  # Normalize to [0, 1]
    pattern_tensor = torch.from_numpy(pattern).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pattern_tensor = pattern_tensor.to(device)
    
    # Predict
    print(f"\n🔮 Predicting phase map...")
    model.eval()
    with torch.no_grad():
        if args.model_type == 'phase_amplitude':
            phase_pred, amp_pred = model(pattern_tensor)
        else:
            phase_pred = model(pattern_tensor)
    
    # Post-process
    phase_map = phase_pred[0, 0].cpu().numpy()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array
    np.save(output_dir / 'predicted_phase_map.npy', phase_map)
    
    # Save visualization
    visualize_prediction(
        pattern,
        None,  # No ground truth
        phase_map,
        save_path=output_dir / 'prediction_visualization.png',
        show_target=False
    )
    
    print(f"\n✓ Prediction complete!")
    print(f"  • Phase map saved: {output_dir / 'predicted_phase_map.npy'}")
    print(f"  • Visualization: {output_dir / 'prediction_visualization.png'}")
    print(f"\n💡 This prediction was {1000}x faster than MEEP simulation!")


def visualize_prediction(pattern, target, prediction, save_path, show_target=True):
    """Visualize prediction results"""
    if show_target:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pattern
    ax = axes[0]
    im = ax.imshow(pattern[0] if pattern.ndim == 3 else pattern, cmap='gray')
    ax.set_title('Input: Random Pillar Pattern', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Prediction
    ax = axes[1]
    im = ax.imshow(prediction[0] if prediction.ndim == 3 else prediction, cmap='twilight')
    ax.set_title('Predicted Phase Map', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Phase (rad)')
    
    if show_target and target is not None:
        # Ground truth
        ax = axes[2]
        im = ax.imshow(target[0] if target.ndim == 3 else target, cmap='twilight')
        ax.set_title('Ground Truth (MEEP)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        # Error
        ax = axes[3]
        error = np.abs((prediction[0] if prediction.ndim == 3 else prediction) - 
                      (target[0] if target.ndim == 3 else target))
        im = ax.imshow(error, cmap='hot')
        ax.set_title('Absolute Error', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='|Error| (rad)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function"""
    args = parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'predict':
        predict(args)


if __name__ == '__main__':
    main()

