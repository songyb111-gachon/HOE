#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Training Tiles from Large Samples using Sliding Window Approach

Usage:
    python create_training_tiles.py \
        --data_dir data/forward_phase \
        --output_dir data/forward_phase_tiles \
        --tile_size 256 \
        --num_tiles_per_sample 1000 \
        --train_samples 8 \
        --val_samples 2
"""

import numpy as np
import cv2
import argparse
from pathlib import Path
import json
import random
from tqdm import tqdm


def extract_random_tile(image, tile_size):
    """Extract a random tile from the image
    
    Args:
        image: 2D numpy array (H, W)
        tile_size: Size of the tile (tile_size × tile_size)
        
    Returns:
        tile: Extracted tile
        top_left: (y, x) coordinate of top-left corner
    """
    h, w = image.shape
    
    # Random top-left position
    max_y = h - tile_size
    max_x = w - tile_size
    
    if max_y <= 0 or max_x <= 0:
        raise ValueError(f"Image size {image.shape} is too small for tile size {tile_size}")
    
    top_y = random.randint(0, max_y)
    top_x = random.randint(0, max_x)
    
    # Extract tile
    tile = image[top_y:top_y+tile_size, top_x:top_x+tile_size]
    
    return tile, (top_y, top_x)


def create_tiles_from_dataset(data_dir, output_dir, 
                              tile_size=256, 
                              num_tiles_per_sample=1000,
                              train_samples=8,
                              val_samples=2):
    """Create training and validation tiles from large samples
    
    Args:
        data_dir: Directory containing large samples (inputs/ and outputs/)
        output_dir: Output directory for tiles
        tile_size: Size of each tile
        num_tiles_per_sample: Number of tiles to extract from each sample
        train_samples: Number of samples to use for training
        val_samples: Number of samples to use for validation
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    print(f"\n{'='*80}")
    print(f"🎯 Creating Training Tiles with Sliding Window Approach")
    print(f"{'='*80}")
    print(f"Source directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tile size: {tile_size}×{tile_size}")
    print(f"Tiles per sample: {num_tiles_per_sample}")
    print(f"Train samples: {train_samples}")
    print(f"Val samples: {val_samples}")
    
    # Create output directories
    for split in ['train', 'val']:
        (output_dir / split / 'inputs').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'outputs').mkdir(parents=True, exist_ok=True)
    
    # Get list of all samples
    input_dir = data_dir / 'inputs'
    all_samples = sorted(list(input_dir.glob('*.png')))
    
    if len(all_samples) < train_samples + val_samples:
        raise ValueError(f"Not enough samples: found {len(all_samples)}, need {train_samples + val_samples}")
    
    print(f"\n📂 Found {len(all_samples)} samples")
    
    # Split into train and val
    random.shuffle(all_samples)
    train_sample_files = all_samples[:train_samples]
    val_sample_files = all_samples[train_samples:train_samples+val_samples]
    
    print(f"  • Training samples: {train_samples}")
    print(f"  • Validation samples: {val_samples}")
    
    metadata = {
        'tile_size': tile_size,
        'num_tiles_per_sample': num_tiles_per_sample,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'train_total_tiles': train_samples * num_tiles_per_sample,
        'val_total_tiles': val_samples * num_tiles_per_sample,
        'train_sample_files': [str(f.name) for f in train_sample_files],
        'val_sample_files': [str(f.name) for f in val_sample_files]
    }
    
    # Generate training tiles
    print(f"\n{'='*80}")
    print(f"🔨 Generating Training Tiles...")
    print(f"{'='*80}")
    
    tile_idx = 0
    for sample_file in tqdm(train_sample_files, desc="Training samples"):
        # Load input and output
        input_path = data_dir / 'inputs' / sample_file.name
        output_path = data_dir / 'outputs' / sample_file.stem + '.npy'
        
        input_img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        output_phase = np.load(output_path)
        
        if input_img is None:
            print(f"  ⚠️ Failed to load {input_path}, skipping...")
            continue
        
        # Check sizes match
        if input_img.shape != output_phase.shape:
            print(f"  ⚠️ Size mismatch: input {input_img.shape} vs output {output_phase.shape}")
            print(f"     Resizing input to match output...")
            input_img = cv2.resize(input_img, (output_phase.shape[1], output_phase.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Extract tiles
        for _ in range(num_tiles_per_sample):
            try:
                # Extract input tile
                input_tile, (top_y, top_x) = extract_random_tile(input_img, tile_size)
                
                # Extract corresponding output tile
                output_tile = output_phase[top_y:top_y+tile_size, top_x:top_x+tile_size]
                
                # Save tiles
                tile_name = f"tile_{tile_idx:06d}"
                cv2.imwrite(str(output_dir / 'train' / 'inputs' / f"{tile_name}.png"), input_tile)
                np.save(str(output_dir / 'train' / 'outputs' / f"{tile_name}.npy"), output_tile)
                
                tile_idx += 1
                
            except Exception as e:
                print(f"  ⚠️ Failed to extract tile: {e}")
                continue
    
    print(f"\n✅ Generated {tile_idx} training tiles")
    
    # Generate validation tiles
    print(f"\n{'='*80}")
    print(f"🔨 Generating Validation Tiles...")
    print(f"{'='*80}")
    
    tile_idx = 0
    for sample_file in tqdm(val_sample_files, desc="Validation samples"):
        # Load input and output
        input_path = data_dir / 'inputs' / sample_file.name
        output_path = data_dir / 'outputs' / sample_file.stem + '.npy'
        
        input_img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        output_phase = np.load(output_path)
        
        if input_img is None:
            print(f"  ⚠️ Failed to load {input_path}, skipping...")
            continue
        
        # Check sizes match
        if input_img.shape != output_phase.shape:
            print(f"  ⚠️ Size mismatch: input {input_img.shape} vs output {output_phase.shape}")
            print(f"     Resizing input to match output...")
            input_img = cv2.resize(input_img, (output_phase.shape[1], output_phase.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Extract tiles
        for _ in range(num_tiles_per_sample):
            try:
                # Extract input tile
                input_tile, (top_y, top_x) = extract_random_tile(input_img, tile_size)
                
                # Extract corresponding output tile
                output_tile = output_phase[top_y:top_y+tile_size, top_x:top_x+tile_size]
                
                # Save tiles
                tile_name = f"tile_{tile_idx:06d}"
                cv2.imwrite(str(output_dir / 'val' / 'inputs' / f"{tile_name}.png"), input_tile)
                np.save(str(output_dir / 'val' / 'outputs' / f"{tile_name}.npy"), output_tile)
                
                tile_idx += 1
                
            except Exception as e:
                print(f"  ⚠️ Failed to extract tile: {e}")
                continue
    
    print(f"\n✅ Generated {tile_idx} validation tiles")
    
    # Save metadata
    metadata_path = output_dir / 'tiles_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🎉 Tile Generation Complete!")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"  • Training tiles: {metadata['train_total_tiles']}")
    print(f"  • Validation tiles: {metadata['val_total_tiles']}")
    print(f"  • Tile size: {tile_size}×{tile_size}")
    print(f"\n📁 Output structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── inputs/   ({metadata['train_total_tiles']} tiles)")
    print(f"    │   └── outputs/  ({metadata['train_total_tiles']} tiles)")
    print(f"    ├── val/")
    print(f"    │   ├── inputs/   ({metadata['val_total_tiles']} tiles)")
    print(f"    │   └── outputs/  ({metadata['val_total_tiles']} tiles)")
    print(f"    └── tiles_metadata.json")
    
    print(f"\n💡 Usage:")
    print(f"  python forward_main.py \\")
    print(f"      --data_path {output_dir}/train \\")
    print(f"      --mode train")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Create training tiles from large samples')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing large samples')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for tiles')
    parser.add_argument('--tile_size', type=int, default=256,
                       help='Size of each tile (default: 256)')
    parser.add_argument('--num_tiles_per_sample', type=int, default=1000,
                       help='Number of tiles to extract per sample (default: 1000)')
    parser.add_argument('--train_samples', type=int, default=8,
                       help='Number of samples for training (default: 8)')
    parser.add_argument('--val_samples', type=int, default=2,
                       help='Number of samples for validation (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create tiles
    create_tiles_from_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        num_tiles_per_sample=args.num_tiles_per_sample,
        train_samples=args.train_samples,
        val_samples=args.val_samples
    )


if __name__ == "__main__":
    main()

