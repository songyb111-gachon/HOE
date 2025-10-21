#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sliding Window Prediction for Large Phase Maps

Usage:
    python predict_with_sliding_window.py \
        --input_mask random_pillar_mask.png \
        --checkpoint checkpoints/best_model.pth \
        --output_dir predictions \
        --tile_size 256 \
        --stride 64
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import models
import sys
sys.path.append('pytorch_codes')
from models import ForwardPhaseUNet, MultiScalePhaseUNet, PhaseAmplitudeUNet


def sliding_window_inference(model, image, tile_size=256, stride=64, device='cuda'):
    """Perform inference using sliding window with averaging
    
    Args:
        model: Trained PyTorch model
        image: Input image (H, W) numpy array
        tile_size: Size of each window
        stride: Stride for sliding window
        device: Device to run inference on
        
    Returns:
        prediction: Averaged prediction map (H, W)
        count_map: Count of predictions per pixel
    """
    h, w = image.shape
    
    # Initialize prediction and count maps
    prediction_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.int32)
    
    # Calculate number of tiles
    n_tiles_y = (h - tile_size) // stride + 1
    n_tiles_x = (w - tile_size) // stride + 1
    
    total_tiles = n_tiles_y * n_tiles_x
    
    print(f"\n🔍 Sliding Window Inference:")
    print(f"  • Image size: {h}×{w}")
    print(f"  • Tile size: {tile_size}×{tile_size}")
    print(f"  • Stride: {stride}")
    print(f"  • Number of tiles: {n_tiles_y}×{n_tiles_x} = {total_tiles}")
    
    model.eval()
    with torch.no_grad():
        # Sliding window
        pbar = tqdm(total=total_tiles, desc="Processing tiles")
        
        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                # Calculate tile position
                top = i * stride
                left = j * stride
                bottom = min(top + tile_size, h)
                right = min(left + tile_size, w)
                
                # Handle edge cases
                if bottom - top < tile_size:
                    top = max(0, bottom - tile_size)
                if right - left < tile_size:
                    left = max(0, right - tile_size)
                
                # Extract tile
                tile = image[top:bottom, left:right]
                
                # Prepare input
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
                tile_tensor = tile_tensor / 255.0  # Normalize to [0, 1]
                tile_tensor = tile_tensor.to(device)
                
                # Inference
                output = model(tile_tensor)
                
                # Extract prediction
                pred = output.squeeze().cpu().numpy()  # (H, W)
                
                # Add to prediction map
                prediction_map[top:bottom, left:right] += pred
                count_map[top:bottom, left:right] += 1
                
                pbar.update(1)
        
        pbar.close()
    
    # Average predictions
    # Avoid division by zero
    count_map = np.maximum(count_map, 1)
    prediction_map = prediction_map / count_map
    
    print(f"  ✓ Inference complete!")
    print(f"  • Average predictions per pixel: {np.mean(count_map):.1f}")
    print(f"  • Min predictions per pixel: {np.min(count_map)}")
    print(f"  • Max predictions per pixel: {np.max(count_map)}")
    
    return prediction_map, count_map


def load_model(checkpoint_path, model_type='basic', device='cuda'):
    """Load trained model from checkpoint"""
    print(f"\n📥 Loading model from: {checkpoint_path}")
    
    # Create model
    if model_type == 'basic':
        model = ForwardPhaseUNet(in_channels=1, out_channels=1)
    elif model_type == 'multiscale':
        model = MultiScalePhaseUNet(in_channels=1, out_channels=1)
    elif model_type == 'phase_amplitude':
        model = PhaseAmplitudeUNet(in_channels=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded successfully")
    print(f"  • Model type: {model_type}")
    print(f"  • Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  • Val Loss: {checkpoint.get('val_loss', 'unknown')}")
    
    return model


def visualize_results(input_mask, prediction, count_map, output_dir):
    """Visualize prediction results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Input mask
    axes[0, 0].imshow(input_mask, cmap='gray')
    axes[0, 0].set_title('Input: Random Pillar Mask', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Predicted phase map
    im1 = axes[0, 1].imshow(prediction, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('Predicted Phase Map', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], label='Phase (rad)')
    
    # Count map (coverage)
    im2 = axes[1, 0].imshow(count_map, cmap='viridis')
    axes[1, 0].set_title('Prediction Count Map\n(Overlapping Predictions)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], label='Count')
    
    # Phase histogram
    axes[1, 1].hist(prediction.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Phase (rad)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Phase Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=np.mean(prediction), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(prediction):.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save
    vis_path = output_dir / 'prediction_visualization.png'
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Visualization saved: {vis_path}")


def main():
    parser = argparse.ArgumentParser(description='Sliding Window Prediction')
    parser.add_argument('--input_mask', type=str, required=True,
                       help='Path to input pillar mask (PNG)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Output directory')
    parser.add_argument('--model_type', type=str, default='basic',
                       choices=['basic', 'multiscale', 'phase_amplitude'],
                       help='Model type')
    parser.add_argument('--tile_size', type=int, default=256,
                       help='Tile size (default: 256)')
    parser.add_argument('--stride', type=int, default=64,
                       help='Stride for sliding window (default: 64)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 Sliding Window Phase Map Prediction")
    print("="*80)
    
    # Set device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"\n✓ Using device: {device}")
    
    # Load input mask
    print(f"\n📂 Loading input mask: {args.input_mask}")
    input_mask = cv2.imread(args.input_mask, cv2.IMREAD_GRAYSCALE)
    
    if input_mask is None:
        raise ValueError(f"Failed to load input mask: {args.input_mask}")
    
    print(f"  ✓ Input mask loaded")
    print(f"  • Size: {input_mask.shape}")
    print(f"  • Fill ratio: {np.sum(input_mask > 128) / input_mask.size * 100:.1f}%")
    
    # Load model
    model = load_model(args.checkpoint, args.model_type, device)
    
    # Perform sliding window inference
    prediction, count_map = sliding_window_inference(
        model=model,
        image=input_mask,
        tile_size=args.tile_size,
        stride=args.stride,
        device=device
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving results...")
    
    # Save predicted phase map
    phase_path = output_dir / 'predicted_phase_map.npy'
    np.save(phase_path, prediction.astype(np.float32))
    print(f"  ✓ Phase map saved: {phase_path}")
    
    # Save count map
    count_path = output_dir / 'count_map.npy'
    np.save(count_path, count_map.astype(np.int32))
    print(f"  ✓ Count map saved: {count_path}")
    
    # Visualize
    print(f"\n🎨 Generating visualization...")
    visualize_results(input_mask, prediction, count_map, output_dir)
    
    # Statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"  • Mean phase: {np.mean(prediction):.4f} rad ({np.mean(prediction)/np.pi:.2f}π)")
    print(f"  • Std phase: {np.std(prediction):.4f} rad ({np.std(prediction)/np.pi:.2f}π)")
    print(f"  • Phase range: [{np.min(prediction):.4f}, {np.max(prediction):.4f}] rad")
    
    print(f"\n✅ Prediction complete!")
    print(f"   Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

