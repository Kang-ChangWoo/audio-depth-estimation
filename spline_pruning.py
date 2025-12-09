"""
MACP-style Spline Control Point Pruning

This script implements "Motion-Adaptive Control Point Pruning" adapted for
depth estimation ("Depth-Adaptive Control Point Pruning").

Key idea from SplineGS:
- Start with many control points
- Iteratively remove control points that don't significantly affect reconstruction
- Use least-squares error threshold to decide which points to prune

This script:
1. Analyzes GT depth maps to find optimal spline parameterization
2. Prunes unnecessary control points
3. Outputs statistics on optimal ctrl_x, ctrl_y for different scene complexities

Usage:
    python spline_pruning.py --dataset batvisionv2 --output_dir ./spline_analysis
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.linalg import lstsq
import json

try:
    from dataloader.BatvisionV1_Dataset import BatvisionV1Dataset
    from dataloader.BatvisionV2_Dataset import BatvisionV2Dataset
    from config_loader import load_config
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    print("Warning: Dataset modules not available. Using synthetic data for testing.")


class CubicHermiteSpline1D:
    """
    1D Cubic Hermite Spline for fitting depth profiles.
    """
    
    def __init__(self, n_ctrl: int):
        self.n_ctrl = n_ctrl
    
    def fit(self, y_data: np.ndarray) -> np.ndarray:
        """
        Fit spline control points to 1D data using least squares.
        
        Args:
            y_data: [N] 1D signal to fit
            
        Returns:
            ctrl_points: [n_ctrl] fitted control points
        """
        N = len(y_data)
        n_ctrl = self.n_ctrl
        
        # Build basis matrix
        # Each column is the contribution of one control point to all output positions
        t = np.linspace(0, 1, N)
        B = np.zeros((N, n_ctrl))
        
        for i in range(N):
            for k in range(n_ctrl):
                B[i, k] = self._hermite_basis(t[i], k, n_ctrl)
        
        # Solve least squares: B @ ctrl = y_data
        ctrl_points, _, _, _ = lstsq(B, y_data)
        return ctrl_points
    
    def evaluate(self, ctrl_points: np.ndarray, n_out: int) -> np.ndarray:
        """
        Evaluate spline at n_out uniformly spaced points.
        """
        t = np.linspace(0, 1, n_out)
        y = np.zeros(n_out)
        
        n_ctrl = len(ctrl_points)
        for i in range(n_out):
            for k in range(n_ctrl):
                y[i] += ctrl_points[k] * self._hermite_basis(t[i], k, n_ctrl)
        
        return y
    
    def _hermite_basis(self, t: float, k: int, n_ctrl: int) -> float:
        """
        Compute Hermite basis function value for control point k at position t.
        Uses simplified linear interpolation basis (for speed).
        For exact Hermite, would need tangent computation.
        """
        # Position of control point k in [0, 1]
        tk = k / (n_ctrl - 1) if n_ctrl > 1 else 0.5
        
        # Tent function (linear B-spline basis)
        dt = 1.0 / (n_ctrl - 1) if n_ctrl > 1 else 1.0
        if abs(t - tk) < dt:
            return 1.0 - abs(t - tk) / dt
        else:
            return 0.0


class LowRankSplineDepth:
    """
    Low-rank spline decomposition for 2D depth maps:
        D(x,y) ≈ Σ_r u_r(x) * v_r(y)
    """
    
    def __init__(self, rank: int, ctrl_x: int, ctrl_y: int):
        self.rank = rank
        self.ctrl_x = ctrl_x
        self.ctrl_y = ctrl_y
        self.spline_x = CubicHermiteSpline1D(ctrl_x)
        self.spline_y = CubicHermiteSpline1D(ctrl_y)
    
    def fit(self, depth_map: np.ndarray) -> tuple:
        """
        Fit low-rank spline to depth map using alternating least squares.
        
        Args:
            depth_map: [H, W] depth image
            
        Returns:
            Px: [rank, ctrl_x] x-direction control points
            Py: [rank, ctrl_y] y-direction control points
            reconstruction: [H, W] reconstructed depth
        """
        H, W = depth_map.shape
        
        # Initialize using SVD
        U, S, Vt = np.linalg.svd(depth_map, full_matrices=False)
        
        Px = np.zeros((self.rank, self.ctrl_x))
        Py = np.zeros((self.rank, self.ctrl_y))
        
        for r in range(min(self.rank, len(S))):
            # U[:, r] corresponds to y-direction (columns), V[r, :] to x-direction
            u_init = U[:, r] * np.sqrt(S[r])
            v_init = Vt[r, :] * np.sqrt(S[r])
            
            # Fit splines
            Px[r] = self.spline_x.fit(v_init)
            Py[r] = self.spline_y.fit(u_init)
        
        # Compute reconstruction
        reconstruction = self.reconstruct(Px, Py, H, W)
        
        return Px, Py, reconstruction
    
    def reconstruct(self, Px: np.ndarray, Py: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Reconstruct depth map from control points.
        """
        depth = np.zeros((H, W))
        
        for r in range(self.rank):
            u = self.spline_x.evaluate(Px[r], W)  # [W]
            v = self.spline_y.evaluate(Py[r], H)  # [H]
            depth += np.outer(v, u)  # [H, W]
        
        return depth
    
    def compute_error(self, depth_gt: np.ndarray, depth_pred: np.ndarray) -> dict:
        """
        Compute reconstruction error metrics.
        """
        valid = depth_gt > 0
        if not valid.any():
            return {'mse': 0, 'mae': 0, 'max_error': 0}
        
        diff = np.abs(depth_gt[valid] - depth_pred[valid])
        
        return {
            'mse': np.mean(diff ** 2),
            'mae': np.mean(diff),
            'max_error': np.max(diff),
            'rmse': np.sqrt(np.mean(diff ** 2)),
        }


class MACPPruner:
    """
    Motion-Adaptive Control Point Pruning (adapted for depth).
    
    Iteratively removes control points that contribute least to 
    reconstruction quality.
    """
    
    def __init__(self, error_threshold: float = 0.01):
        """
        Args:
            error_threshold: Maximum allowed error increase (relative) when pruning a point
        """
        self.error_threshold = error_threshold
    
    def prune(self, depth_map: np.ndarray, initial_ctrl: int = 16, 
              min_ctrl: int = 4, rank: int = 4) -> dict:
        """
        Find optimal number of control points via pruning.
        
        Args:
            depth_map: [H, W] GT depth map
            initial_ctrl: Initial number of control points
            min_ctrl: Minimum control points to keep
            rank: Rank for low-rank decomposition
            
        Returns:
            dict with optimal parameters and analysis
        """
        H, W = depth_map.shape
        
        # Track errors at different control point counts
        results = {
            'ctrl_x_history': [],
            'ctrl_y_history': [],
            'error_history': [],
        }
        
        current_ctrl_x = initial_ctrl
        current_ctrl_y = initial_ctrl
        
        # Initial fit
        model = LowRankSplineDepth(rank, current_ctrl_x, current_ctrl_y)
        Px, Py, recon = model.fit(depth_map)
        current_error = model.compute_error(depth_map, recon)['rmse']
        
        results['ctrl_x_history'].append(current_ctrl_x)
        results['ctrl_y_history'].append(current_ctrl_y)
        results['error_history'].append(current_error)
        
        # Alternating pruning for x and y
        while current_ctrl_x > min_ctrl or current_ctrl_y > min_ctrl:
            best_action = None
            best_error_increase = float('inf')
            
            # Try pruning x
            if current_ctrl_x > min_ctrl:
                test_ctrl_x = current_ctrl_x - 1
                model_x = LowRankSplineDepth(rank, test_ctrl_x, current_ctrl_y)
                _, _, recon_x = model_x.fit(depth_map)
                error_x = model_x.compute_error(depth_map, recon_x)['rmse']
                error_increase_x = (error_x - current_error) / max(current_error, 1e-6)
                
                if error_increase_x < best_error_increase:
                    best_error_increase = error_increase_x
                    best_action = ('x', test_ctrl_x, current_ctrl_y, error_x)
            
            # Try pruning y
            if current_ctrl_y > min_ctrl:
                test_ctrl_y = current_ctrl_y - 1
                model_y = LowRankSplineDepth(rank, current_ctrl_x, test_ctrl_y)
                _, _, recon_y = model_y.fit(depth_map)
                error_y = model_y.compute_error(depth_map, recon_y)['rmse']
                error_increase_y = (error_y - current_error) / max(current_error, 1e-6)
                
                if error_increase_y < best_error_increase:
                    best_error_increase = error_increase_y
                    best_action = ('y', current_ctrl_x, test_ctrl_y, error_y)
            
            # Check if we should continue pruning
            if best_action is None or best_error_increase > self.error_threshold:
                break
            
            # Apply the best pruning action
            _, current_ctrl_x, current_ctrl_y, current_error = best_action
            
            results['ctrl_x_history'].append(current_ctrl_x)
            results['ctrl_y_history'].append(current_ctrl_y)
            results['error_history'].append(current_error)
        
        results['optimal_ctrl_x'] = current_ctrl_x
        results['optimal_ctrl_y'] = current_ctrl_y
        results['final_error'] = current_error
        
        return results


def analyze_dataset(cfg, dataset, output_dir: str, num_samples: int = 100, 
                    rank: int = 4, initial_ctrl: int = 16):
    """
    Analyze dataset to find optimal spline parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pruner = MACPPruner(error_threshold=0.05)  # 5% error increase threshold
    
    all_results = []
    optimal_ctrl_x = []
    optimal_ctrl_y = []
    
    # Sample from dataset
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    print(f"Analyzing {len(indices)} samples from {cfg.dataset.name}...")
    
    for idx in tqdm(indices):
        try:
            audio, depth = dataset[idx]
            
            # Convert to numpy
            if isinstance(depth, torch.Tensor):
                depth_np = depth.numpy()
            else:
                depth_np = depth
            
            # Handle channel dimension
            if depth_np.ndim == 3:
                depth_np = depth_np[0]
            
            # Denormalize if needed
            if cfg.dataset.depth_norm and cfg.dataset.max_depth:
                depth_np = depth_np * cfg.dataset.max_depth
            
            # Run pruning analysis
            result = pruner.prune(depth_np, initial_ctrl=initial_ctrl, 
                                  min_ctrl=4, rank=rank)
            
            all_results.append(result)
            optimal_ctrl_x.append(result['optimal_ctrl_x'])
            optimal_ctrl_y.append(result['optimal_ctrl_y'])
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Aggregate statistics
    stats = {
        'num_samples': len(all_results),
        'rank': rank,
        'initial_ctrl': initial_ctrl,
        'ctrl_x': {
            'mean': float(np.mean(optimal_ctrl_x)),
            'std': float(np.std(optimal_ctrl_x)),
            'min': int(np.min(optimal_ctrl_x)),
            'max': int(np.max(optimal_ctrl_x)),
            'median': float(np.median(optimal_ctrl_x)),
        },
        'ctrl_y': {
            'mean': float(np.mean(optimal_ctrl_y)),
            'std': float(np.std(optimal_ctrl_y)),
            'min': int(np.min(optimal_ctrl_y)),
            'max': int(np.max(optimal_ctrl_y)),
            'median': float(np.median(optimal_ctrl_y)),
        },
    }
    
    # Save statistics
    with open(os.path.join(output_dir, 'spline_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n=== Spline Analysis Results ===")
    print(f"Analyzed {stats['num_samples']} samples")
    print(f"Optimal ctrl_x: {stats['ctrl_x']['mean']:.1f} ± {stats['ctrl_x']['std']:.1f} (range: {stats['ctrl_x']['min']}-{stats['ctrl_x']['max']})")
    print(f"Optimal ctrl_y: {stats['ctrl_y']['mean']:.1f} ± {stats['ctrl_y']['std']:.1f} (range: {stats['ctrl_y']['min']}-{stats['ctrl_y']['max']})")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(optimal_ctrl_x, bins=range(3, initial_ctrl + 2), edgecolor='black', alpha=0.7)
    axes[0].axvline(stats['ctrl_x']['mean'], color='r', linestyle='--', label=f"Mean: {stats['ctrl_x']['mean']:.1f}")
    axes[0].set_xlabel('Optimal ctrl_x')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Optimal X Control Points')
    axes[0].legend()
    
    axes[1].hist(optimal_ctrl_y, bins=range(3, initial_ctrl + 2), edgecolor='black', alpha=0.7)
    axes[1].axvline(stats['ctrl_y']['mean'], color='r', linestyle='--', label=f"Mean: {stats['ctrl_y']['mean']:.1f}")
    axes[1].set_xlabel('Optimal ctrl_y')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Optimal Y Control Points')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ctrl_distribution.png'), dpi=150)
    plt.close()
    
    # Plot error vs control points for a sample
    if all_results:
        sample_result = all_results[0]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        iterations = range(len(sample_result['error_history']))
        ax.plot(iterations, sample_result['error_history'], 'b-o', markersize=4)
        ax.set_xlabel('Pruning Iteration')
        ax.set_ylabel('RMSE')
        ax.set_title('Error vs Pruning Iterations (Sample)')
        
        # Add control point counts as secondary axis labels
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ctrl_labels = [f"({sample_result['ctrl_x_history'][i]},{sample_result['ctrl_y_history'][i]})" 
                       for i in range(0, len(iterations), max(1, len(iterations)//5))]
        ax2.set_xticks(range(0, len(iterations), max(1, len(iterations)//5)))
        ax2.set_xticklabels(ctrl_labels, fontsize=8)
        ax2.set_xlabel('(ctrl_x, ctrl_y)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_vs_pruning.png'), dpi=150)
        plt.close()
    
    return stats


def visualize_spline_fit(depth_map: np.ndarray, rank: int, ctrl_x: int, ctrl_y: int,
                         output_path: str):
    """
    Visualize spline fitting quality for a single depth map.
    """
    model = LowRankSplineDepth(rank, ctrl_x, ctrl_y)
    Px, Py, recon = model.fit(depth_map)
    error = model.compute_error(depth_map, recon)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original
    im0 = axes[0].imshow(depth_map, cmap='viridis')
    axes[0].set_title('Original Depth')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Reconstruction
    im1 = axes[1].imshow(recon, cmap='viridis')
    axes[1].set_title(f'Spline Reconstruction (R={rank}, Cx={ctrl_x}, Cy={ctrl_y})')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Error
    error_map = np.abs(depth_map - recon)
    im2 = axes[2].imshow(error_map, cmap='hot')
    axes[2].set_title(f'Absolute Error (RMSE={error["rmse"]:.4f})')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return error


def test_synthetic():
    """
    Test with synthetic data when dataset is not available.
    """
    print("Testing with synthetic depth maps...")
    
    # Create synthetic depth maps with varying complexity
    H, W = 256, 256
    
    # Simple: smooth gradient
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)
    depth_simple = 5 + 3 * X + 2 * Y + 0.5 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    
    # Complex: multiple frequency components
    depth_complex = (5 + 3 * X + 2 * Y + 
                    np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y) +
                    0.5 * np.sin(8 * np.pi * X) +
                    0.3 * np.cos(6 * np.pi * Y))
    
    output_dir = './spline_analysis_synthetic'
    os.makedirs(output_dir, exist_ok=True)
    
    for name, depth in [('simple', depth_simple), ('complex', depth_complex)]:
        print(f"\nAnalyzing {name} depth map...")
        
        # Test different configurations
        for rank in [2, 4, 8]:
            for ctrl in [4, 8, 12]:
                error = visualize_spline_fit(
                    depth, rank, ctrl, ctrl,
                    os.path.join(output_dir, f'{name}_R{rank}_C{ctrl}.png')
                )
                print(f"  R={rank}, C={ctrl}: RMSE={error['rmse']:.4f}")
        
        # Run MACP pruning
        pruner = MACPPruner(error_threshold=0.05)
        result = pruner.prune(depth, initial_ctrl=16, min_ctrl=4, rank=4)
        print(f"  Optimal: ctrl_x={result['optimal_ctrl_x']}, ctrl_y={result['optimal_ctrl_y']}, error={result['final_error']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='MACP-style Spline Control Point Pruning')
    
    parser.add_argument('--dataset', type=str, default='batvisionv2',
                        choices=['batvisionv1', 'batvisionv2'],
                        help='Dataset to analyze')
    parser.add_argument('--output_dir', type=str, default='./spline_analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to analyze')
    parser.add_argument('--rank', type=int, default=4,
                        help='Rank for low-rank decomposition')
    parser.add_argument('--initial_ctrl', type=int, default=16,
                        help='Initial number of control points')
    parser.add_argument('--test_synthetic', action='store_true',
                        help='Run tests with synthetic data')
    
    args = parser.parse_args()
    
    if args.test_synthetic or not DATASET_AVAILABLE:
        test_synthetic()
        return
    
    # Load configuration and dataset
    cfg = load_config(dataset_name=args.dataset, mode='train', experiment_name='spline_analysis')
    
    if args.dataset == 'batvisionv1':
        dataset = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_train)
    else:
        dataset = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_train)
    
    print(f"Loaded {len(dataset)} samples from {args.dataset}")
    
    # Run analysis
    stats = analyze_dataset(
        cfg, dataset, args.output_dir,
        num_samples=args.num_samples,
        rank=args.rank,
        initial_ctrl=args.initial_ctrl,
    )
    
    # Print recommendations
    print("\n=== Recommendations ===")
    recommended_ctrl_x = int(np.ceil(stats['ctrl_x']['mean'] + stats['ctrl_x']['std']))
    recommended_ctrl_y = int(np.ceil(stats['ctrl_y']['mean'] + stats['ctrl_y']['std']))
    print(f"Recommended ctrl_x: {recommended_ctrl_x}")
    print(f"Recommended ctrl_y: {recommended_ctrl_y}")
    print(f"\nTo train with these settings:")
    print(f"  python train_spline.py --dataset {args.dataset} --rank {args.rank} --ctrl_x {recommended_ctrl_x} --ctrl_y {recommended_ctrl_y}")


if __name__ == '__main__':
    main()


