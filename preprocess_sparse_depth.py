#!/usr/bin/env python3
"""
Preprocess depth maps to create sparse/coarse depth data.

Creates sparse_depth_{method} folders with processed depth maps.
Supports various simplification methods for coarse depth classification.

Usage:
    python preprocess_sparse_depth.py --method downup_015 --dataset_dir /path/to/BatvisionV2

Methods:
    - downup_XXX: Down-up sampling with scale=0.XX (e.g., downup_015 = scale=0.15)
    - superpixel_XXX: Superpixel with n=XXX segments
    - quantized_XXX: Quantization with XXX levels
    - sp_extreme_XXX: Superpixel extreme with n=XXX
    - sp_hier_XXX: Superpixel hierarchical (XXX = config name)
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# Smoothing Functions (from EDA.ipynb)
# ============================================================================

def fill_holes_completely(depth, invalid_value=0.0):
    """Fill holes using inpainting and morphological closing."""
    mask = (depth > invalid_value) & ~np.isnan(depth)
    depth_filled = depth.copy().astype(np.float32)
    
    if mask.all():
        return depth_filled
    
    mask_uint8 = (~mask).astype(np.uint8) * 255
    depth_filled = cv2.inpaint(depth_filled, mask_uint8, 10, cv2.INPAINT_TELEA)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    depth_filled = cv2.morphologyEx(depth_filled, cv2.MORPH_CLOSE, kernel)
    
    return depth_filled


def smooth_downup(depth, scale=0.15, invalid_value=0.0):
    """Down-up sampling for simplification."""
    H, W = depth.shape
    depth_filled = fill_holes_completely(depth, invalid_value)
    
    h_small = max(1, int(H * scale))
    w_small = max(1, int(W * scale))
    depth_small = cv2.resize(depth_filled, (w_small, h_small), interpolation=cv2.INTER_AREA)
    depth_coarse = cv2.resize(depth_small, (W, H), interpolation=cv2.INTER_LINEAR)
    
    return depth_coarse


def smooth_superpixel(depth, n_segments=100, invalid_value=0.0):
    """Superpixel-based simplification."""
    from skimage.segmentation import slic
    
    depth_filled = fill_holes_completely(depth, invalid_value)
    
    depth_norm = (depth_filled - depth_filled.min()) / (depth_filled.max() - depth_filled.min() + 1e-8)
    depth_3ch = np.stack([depth_norm] * 3, axis=-1)
    
    segments = slic(depth_3ch, n_segments=n_segments, compactness=10,
                    start_label=0, channel_axis=-1)
    
    depth_simplified = np.zeros_like(depth_filled)
    for region_id in np.unique(segments):
        mask = segments == region_id
        depth_simplified[mask] = depth_filled[mask].mean()
    
    return depth_simplified


def smooth_sp_extreme(depth, n_segments=30, blur_sigma=3.0, invalid_value=0.0):
    """Extreme superpixel with blur."""
    from skimage.segmentation import slic
    
    depth_filled = fill_holes_completely(depth, invalid_value)
    
    depth_norm = (depth_filled - depth_filled.min()) / (depth_filled.max() - depth_filled.min() + 1e-8)
    depth_3ch = np.stack([depth_norm] * 3, axis=-1)
    
    segments = slic(depth_3ch, n_segments=n_segments, compactness=30,
                    start_label=0, channel_axis=-1)
    
    depth_simplified = np.zeros_like(depth_filled)
    for region_id in np.unique(segments):
        mask = segments == region_id
        depth_simplified[mask] = depth_filled[mask].mean()
    
    if blur_sigma > 0:
        kernel_size = int(blur_sigma * 4) | 1
        depth_simplified = cv2.GaussianBlur(depth_simplified.astype(np.float32),
                                            (kernel_size, kernel_size), blur_sigma)
    
    return depth_simplified


def smooth_sp_hierarchical(depth, levels=[200, 50, 15], invalid_value=0.0):
    """Hierarchical superpixel simplification."""
    from skimage.segmentation import slic
    
    depth_filled = fill_holes_completely(depth, invalid_value)
    depth_result = depth_filled.copy()
    
    for n_seg in levels:
        depth_norm = (depth_result - depth_result.min()) / (depth_result.max() - depth_result.min() + 1e-8)
        depth_3ch = np.stack([depth_norm] * 3, axis=-1)
        
        segments = slic(depth_3ch, n_segments=n_seg, compactness=20,
                        start_label=0, channel_axis=-1)
        
        for region_id in np.unique(segments):
            mask = segments == region_id
            depth_result[mask] = depth_result[mask].mean()
    
    return depth_result


def smooth_quantized(depth, n_levels=32, invalid_value=0.0):
    """Depth quantization."""
    depth_filled = fill_holes_completely(depth, invalid_value)
    
    valid_mask = depth_filled > invalid_value
    if not valid_mask.any():
        return depth_filled
    
    min_d, max_d = depth_filled[valid_mask].min(), depth_filled[valid_mask].max()
    bins = np.linspace(min_d, max_d, n_levels + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    indices = np.digitize(depth_filled, bins) - 1
    indices = np.clip(indices, 0, n_levels - 1)
    depth_quantized = bin_centers[indices]
    depth_quantized = cv2.GaussianBlur(depth_quantized.astype(np.float32), (5, 5), 1.5)
    
    return depth_quantized


def smooth_planar_grid(depth, grid_size=32, invalid_value=0.0):
    """Grid-based planar simplification."""
    H, W = depth.shape
    depth_filled = fill_holes_completely(depth, invalid_value)
    depth_planar = np.zeros_like(depth_filled)
    
    for i in range(0, H, grid_size):
        for j in range(0, W, grid_size):
            i_end = min(i + grid_size, H)
            j_end = min(j + grid_size, W)
            block = depth_filled[i:i_end, j:j_end]
            depth_planar[i:i_end, j:j_end] = block.mean()
    
    depth_planar = cv2.GaussianBlur(depth_planar.astype(np.float32), (15, 15), 3.0)
    return depth_planar


def smooth_iterative_blur(depth, iterations=3, blur_sigma=5.0, invalid_value=0.0):
    """Iterative inpainting + blur."""
    depth_result = depth.copy().astype(np.float32)
    
    for _ in range(iterations):
        mask = (depth_result <= invalid_value) | np.isnan(depth_result)
        if mask.any():
            mask_uint8 = mask.astype(np.uint8) * 255
            depth_result = cv2.inpaint(depth_result, mask_uint8, 5, cv2.INPAINT_TELEA)
        
        kernel_size = int(blur_sigma * 4) | 1
        depth_result = cv2.GaussianBlur(depth_result, (kernel_size, kernel_size), blur_sigma)
    
    return depth_result


# ============================================================================
# Method Parser
# ============================================================================

def parse_method(method_str):
    """
    Parse method string to function and parameters.
    
    Examples:
        downup_015 -> smooth_downup(scale=0.15)
        superpixel_100 -> smooth_superpixel(n_segments=100)
        quantized_32 -> smooth_quantized(n_levels=32)
        sp_extreme_30 -> smooth_sp_extreme(n_segments=30)
        sp_hier_200_50_15 -> smooth_sp_hierarchical(levels=[200, 50, 15])
        grid_32 -> smooth_planar_grid(grid_size=32)
        blur_5_3 -> smooth_iterative_blur(blur_sigma=5, iterations=3)
    """
    parts = method_str.split('_')
    
    if parts[0] == 'downup':
        scale = int(parts[1]) / 100.0  # e.g., 015 -> 0.15
        return lambda d: smooth_downup(d, scale=scale)
    
    elif parts[0] == 'superpixel':
        n_segments = int(parts[1])
        return lambda d: smooth_superpixel(d, n_segments=n_segments)
    
    elif parts[0] == 'quantized':
        n_levels = int(parts[1])
        return lambda d: smooth_quantized(d, n_levels=n_levels)
    
    elif parts[0:2] == ['sp', 'extreme']:
        n_segments = int(parts[2])
        blur_sigma = float(parts[3]) if len(parts) > 3 else 3.0
        return lambda d: smooth_sp_extreme(d, n_segments=n_segments, blur_sigma=blur_sigma)
    
    elif parts[0:2] == ['sp', 'hier']:
        levels = [int(x) for x in parts[2:]]
        return lambda d: smooth_sp_hierarchical(d, levels=levels)
    
    elif parts[0] == 'grid':
        grid_size = int(parts[1])
        return lambda d: smooth_planar_grid(d, grid_size=grid_size)
    
    elif parts[0] == 'blur':
        blur_sigma = float(parts[1])
        iterations = int(parts[2]) if len(parts) > 2 else 3
        return lambda d: smooth_iterative_blur(d, iterations=iterations, blur_sigma=blur_sigma)
    
    else:
        raise ValueError(f"Unknown method: {method_str}")


# ============================================================================
# Main Processing
# ============================================================================

def process_sequence(sequence_path, method_str, smooth_func):
    """Process all depth files in a sequence."""
    depth_dir = sequence_path / 'depth'
    output_dir = sequence_path / f'sparse_depth_{method_str}'
    
    if not depth_dir.exists():
        print(f"  Warning: {depth_dir} not found")
        return 0
    
    output_dir.mkdir(exist_ok=True)
    
    depth_files = sorted(depth_dir.glob('depth_*.npy'))
    
    if len(depth_files) == 0:
        print(f"  Warning: No depth files in {depth_dir}")
        return 0
    
    processed = 0
    for depth_file in tqdm(depth_files, desc=f"  {sequence_path.name}", leave=False):
        try:
            # Load depth (in mm)
            depth = np.load(depth_file).astype(np.float32)
            
            # Handle NaN/Inf
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply smoothing
            depth_smooth = smooth_func(depth)
            
            # Save
            output_file = output_dir / depth_file.name
            np.save(output_file, depth_smooth.astype(np.float32))
            
            processed += 1
        except Exception as e:
            print(f"  Error processing {depth_file.name}: {e}")
    
    return processed


def main():
    parser = argparse.ArgumentParser(description='Preprocess depth for coarse classification')
    
    parser.add_argument('--dataset_dir', type=str, 
                        default='/root/dev/data/dataset/Batvision/BatvisionV2',
                        help='BatvisionV2 dataset directory')
    parser.add_argument('--method', type=str, default='downup_015',
                        help='Preprocessing method (e.g., downup_015, superpixel_100)')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='Specific sequences to process (default: all)')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_dir)
    
    # Parse method
    try:
        smooth_func = parse_method(args.method)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nAvailable methods:")
        print("  downup_XXX      - Down-up sampling (XXX = scale * 100, e.g., 015 = 0.15)")
        print("  superpixel_XXX  - Superpixel (XXX = n_segments)")
        print("  quantized_XXX   - Quantization (XXX = n_levels)")
        print("  sp_extreme_XXX  - Superpixel extreme (XXX = n_segments)")
        print("  sp_hier_A_B_C   - Hierarchical superpixel (levels)")
        print("  grid_XXX        - Planar grid (XXX = grid_size)")
        print("  blur_S_I        - Iterative blur (S = sigma, I = iterations)")
        return
    
    print("=" * 60)
    print("Sparse Depth Preprocessing")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Method: {args.method}")
    print(f"Output folder: sparse_depth_{args.method}")
    
    # Find sequences
    if args.sequences:
        sequences = [dataset_path / seq for seq in args.sequences]
    else:
        sequences = [
            p for p in dataset_path.iterdir()
            if p.is_dir() and not p.name.startswith('.') and not p.name.startswith('__')
        ]
    
    print(f"Sequences: {len(sequences)}")
    print()
    
    total_processed = 0
    
    for seq_path in sequences:
        if not seq_path.exists():
            print(f"Warning: {seq_path} not found")
            continue
        
        print(f"Processing: {seq_path.name}")
        processed = process_sequence(seq_path, args.method, smooth_func)
        
        if processed > 0:
            print(f"  -> {processed} files processed")
            total_processed += processed
        else:
            print(f"  -> Skipped (no files)")
    
    print()
    print("=" * 60)
    print(f"Total: {total_processed} files processed")
    print("=" * 60)


if __name__ == '__main__':
    main()

