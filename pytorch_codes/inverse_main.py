#!/usr/bin/env python3
"""
PyTorch Inverse Design Training Script
Converted from TensorFlow/Keras inverse_codes

Usage:
    python inverse_main.py --data_path ./data/inverse --mode train
    python inverse_main.py --data_path ./data/inverse --mode test --checkpoint ./checkpoints/best_model.pth
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from models import InverseUNet, MultiTaskInverseUNet
from datasets import create_dataloaders
from utils import WeightedMSELoss, MultiTaskLoss, Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch Inverse Design Training')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Train or test mode')
    
    # Model
    parser.add_argument('--in_channels', type=int, default=1,
                       help='Number of input channels')
    parser.add_argument('--out_channels', type=int, nargs='+', default=[1],
                       help='Number of output channels (list for multiple outputs)')
    parser.add_argument('--layer_num', type=int, default=4,
                       help='Number of encoder/decoder layers (1-7)')
    parser.add_argument('--base_features', type=int, default=32,
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
    print(f"Creating Inverse Design Model")
    print(f"{'='*80}")
    
    # Determine use_batchnorm
    use_batchnorm = not args.no_batchnorm if hasattr(args, 'no_batchnorm') else args.use_batchnorm
    print(f"  • BatchNorm: {'Enabled' if use_batchnorm else 'Disabled (Original U-Net)'}")
    
    if len(args.out_channels) == 1:
        # Single output model
        model = InverseUNet(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            layer_num=args.layer_num,
            base_features=args.base_features,
            dropout_rate=args.dropout_rate,
            use_batchnorm=use_batchnorm
        )
    else:
        # Multi-output model
        model = InverseUNet(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            layer_num=args.layer_num,
            base_features=args.base_features,
            dropout_rate=args.dropout_rate,
            output_activations=['linear'] * len(args.out_channels),
            use_batchnorm=use_batchnorm
        )
    
    model.get_model_summary()
    return model


def train(args):
    """Training pipeline"""
    print(f"\n{'='*80}")
    print(f"Inverse Design Training Pipeline")
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
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=args.data_path,
        dataset_type='inverse',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_extension='txt',
        output_type='R'
    )
    
    # Create model
    model = create_model(args)
    model = model.to(device)
    
    # Create loss function
    if len(args.out_channels) == 1:
        criterion = WeightedMSELoss()
    else:
        # Multi-task loss
        task_losses = [WeightedMSELoss() for _ in args.out_channels]
        criterion = MultiTaskLoss(task_losses)
    
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


def test(args):
    """Testing pipeline"""
    print(f"\n{'='*80}")
    print(f"Inverse Design Testing Pipeline")
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
        dataset_type='inverse',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_extension='txt',
        output_type='R'
    )
    
    # Create model
    model = create_model(args)
    model = model.to(device)
    
    # Load checkpoint
    print(f"\n📥 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    
    # Test
    model.eval()
    test_loss = 0.0
    criterion = WeightedMSELoss()
    
    print(f"\n🧪 Testing...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\n{'='*80}")
    print(f"Test Results")
    print(f"{'='*80}")
    print(f"  • Test loss: {avg_test_loss:.6f}")
    print(f"  • Test samples: {len(test_loader.dataset)}")
    print(f"{'='*80}\n")


def main():
    """Main function"""
    args = parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


if __name__ == '__main__':
    main()

