"""
Sparse/Coarse Depth Dataset for BatvisionV2

Supports loading sparse depth from different preprocessing methods:
- sparse_depth_downup_015
- sparse_depth_superpixel_100
- sparse_depth_quantized_128bins
- etc.

Usage:
    dataset = SparseDepthDataset(cfg, 'train.csv', sparse_depth_method='downup_015')
"""

import os
import torch
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np

from .utils_dataset import get_transform


class SparseDepthDataset(Dataset):
    """
    Dataset for training on sparse/coarse depth maps.
    
    Args:
        cfg: Config object
        annotation_file: CSV annotation file name (e.g., 'train.csv')
        sparse_depth_method: Method name for sparse depth (e.g., 'downup_015', 'superpixel_100')
        n_bins: Number of bins for classification (if using binned depth)
        use_original_depth: If True, also return original depth for comparison
    """
    
    def __init__(
        self, 
        cfg, 
        annotation_file, 
        sparse_depth_method='downup_015',
        n_bins=128,
        use_original_depth=False,
        location_blacklist=None
    ):
        self.cfg = cfg
        self.root_dir = cfg.dataset.dataset_dir
        self.audio_format = cfg.dataset.audio_format
        self.sparse_depth_method = sparse_depth_method
        self.n_bins = n_bins
        self.use_original_depth = use_original_depth
        
        # Sparse depth folder name
        self.sparse_depth_folder = f"sparse_depth_{sparse_depth_method}"
        
        # Find valid locations
        location_list = [
            item for item in os.listdir(self.root_dir) 
            if os.path.isdir(os.path.join(self.root_dir, item))
            and not item.startswith('.')
            and not item.startswith('__')
            and not item.endswith('_unzipped')
        ]
        
        if location_blacklist:
            location_list = [loc for loc in location_list if loc not in location_blacklist]
        
        # Check which locations have sparse depth folder
        location_csv_paths = []
        for location in location_list:
            csv_path = os.path.join(self.root_dir, location, annotation_file)
            sparse_dir = os.path.join(self.root_dir, location, self.sparse_depth_folder)
            
            if os.path.exists(csv_path) and os.path.exists(sparse_dir):
                location_csv_paths.append((location, csv_path))
            else:
                if not os.path.exists(sparse_dir):
                    print(f"Warning: {sparse_dir} not found, skipping {location}")
                elif not os.path.exists(csv_path):
                    print(f"Warning: {csv_path} not found, skipping {location}")
        
        if len(location_csv_paths) == 0:
            raise ValueError(
                f"No valid locations found with {self.sparse_depth_folder} in {self.root_dir}. "
                f"Checked {len(location_list)} directories."
            )
        
        # Load annotations
        self.instances = []
        for location, csv_path in location_csv_paths:
            df = pd.read_csv(csv_path)
            df['location'] = location
            self.instances.append(df)
        
        self.instances = pd.concat(self.instances, ignore_index=True)
        
        print(f"SparseDepthDataset: {len(self.instances)} samples from {len(location_csv_paths)} locations")
        print(f"  Sparse depth method: {sparse_depth_method}")
        print(f"  Sparse depth folder: {self.sparse_depth_folder}")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances.iloc[idx]
        location = instance['location']
        
        # Get filenames
        depth_filename = instance['depth file name']
        audio_filename = instance['audio file name']
        
        # Paths
        sparse_depth_path = os.path.join(
            self.root_dir, location, self.sparse_depth_folder, depth_filename
        )
        # Use audio path directly from CSV (e.g., "Outdoor_Cobblestone_Path/audio/")
        audio_path = os.path.join(
            self.root_dir, instance['audio path'], audio_filename
        )
        
        # Load sparse depth
        sparse_depth = np.load(sparse_depth_path).astype(np.float32)
        sparse_depth = sparse_depth / 1000  # mm to m
        
        if self.cfg.dataset.max_depth:
            sparse_depth[sparse_depth > self.cfg.dataset.max_depth] = self.cfg.dataset.max_depth
        sparse_depth[sparse_depth < 0] = 0
        
        # Transform depth
        depth_transform = get_transform(
            self.cfg, convert=True, depth_norm=self.cfg.dataset.depth_norm
        )
        gt_sparse_depth = depth_transform(sparse_depth)
        
        # Load original depth if needed
        if self.use_original_depth:
            original_depth_path = os.path.join(
                self.root_dir, instance['depth path'], depth_filename
            )
            original_depth = np.load(original_depth_path).astype(np.float32)
            original_depth = original_depth / 1000
            if self.cfg.dataset.max_depth:
                original_depth[original_depth > self.cfg.dataset.max_depth] = self.cfg.dataset.max_depth
            original_depth[original_depth < 0] = 0
            gt_original_depth = depth_transform(original_depth)
        
        # Load and process audio
        audio2return = self._load_audio(audio_path)
        
        if self.use_original_depth:
            return audio2return, gt_sparse_depth, gt_original_depth
        else:
            return audio2return, gt_sparse_depth

    def _load_audio(self, audio_path):
        """Load and process audio file."""
        waveform = None
        sr = None
        
        # Try different loading methods
        try:
            waveform, sr = torchaudio.load(audio_path)
        except (RuntimeError, ValueError):
            try:
                from scipy.io import wavfile
                sr, audio_data = wavfile.read(audio_path)
                if audio_data.ndim == 1:
                    waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
                else:
                    waveform = torch.from_numpy(audio_data.T).float()
                if waveform.max() > 1:
                    waveform = waveform / 32768.0
            except Exception as e:
                try:
                    import soundfile as sf
                    audio_data, sr = sf.read(audio_path)
                    if audio_data.ndim == 1:
                        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
                    else:
                        waveform = torch.from_numpy(audio_data.T).float()
                except Exception as e2:
                    raise RuntimeError(f"Could not load audio: {audio_path}")
        
        # STFT parameters
        win_length = 200
        n_fft = 400
        hop_length = 100
        
        # Cut audio for max depth
        if self.cfg.dataset.max_depth:
            cut = int((2 * self.cfg.dataset.max_depth / 340) * sr)
            waveform = waveform[:, :cut]
            win_length = 64
            n_fft = 512
            hop_length = 64 // 4
        
        # Process to spectrogram
        if 'spectrogram' in self.audio_format:
            if 'mel' in self.audio_format:
                spec = self._get_melspectrogram(waveform, n_fft=n_fft, power=1.0, win_length=win_length)
            else:
                spec = self._get_spectrogram(waveform, n_fft=n_fft, power=1.0, 
                                            win_length=win_length, hop_length=hop_length)
            
            spec = torch.log(spec + 1e-8)
            
            # Normalize
            for c in range(spec.shape[0]):
                spec_min = spec[c].min()
                spec_max = spec[c].max()
                if spec_max > spec_min:
                    spec[c] = (spec[c] - spec_min) / (spec_max - spec_min)
                else:
                    spec[c] = torch.zeros_like(spec[c])
            
            spec_transform = get_transform(self.cfg, convert=False)
            return spec_transform(spec)
        else:
            return waveform

    def _get_spectrogram(self, waveform, n_fft=400, power=1.0, win_length=400, hop_length=100):
        spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            power=power,
            hop_length=hop_length,
        )
        return spectrogram(waveform)

    def _get_melspectrogram(self, waveform, n_fft=400, power=1.0, win_length=400, 
                           f_min=20.0, f_max=20000.0):
        melspectrogram = T.MelSpectrogram(
            sample_rate=44100,
            n_fft=n_fft,
            win_length=win_length,
            power=power,
            f_min=f_min,
            f_max=f_max,
            n_mels=32,
        )
        return melspectrogram(waveform)


class BinnedDepthDataset(SparseDepthDataset):
    """
    Dataset that returns binned (classification) depth labels.
    
    Converts continuous depth to discrete bin indices for classification.
    """
    
    def __init__(
        self,
        cfg,
        annotation_file,
        sparse_depth_method='downup_015',
        n_bins=128,
        bin_mode='linear',  # 'linear', 'log', 'sid'
        sid_alpha=0.6,
        depth_min=None,
        depth_max=None,
        **kwargs
    ):
        super().__init__(
            cfg, annotation_file, 
            sparse_depth_method=sparse_depth_method,
            n_bins=n_bins,
            **kwargs
        )
        
        self.bin_mode = bin_mode
        self.sid_alpha = sid_alpha
        
        # Set depth range
        self.depth_min = depth_min if depth_min else 0.1  # 0.1m minimum
        self.depth_max = depth_max if depth_max else cfg.dataset.max_depth
        
        # Compute bin edges and centers
        self._compute_bins()
        
        print(f"  Binning mode: {bin_mode}")
        print(f"  Number of bins: {n_bins}")
        print(f"  Depth range: {self.depth_min:.2f}m ~ {self.depth_max:.2f}m")

    def _compute_bins(self):
        """Compute bin edges and centers based on binning mode."""
        if self.bin_mode == 'linear':
            self.bin_edges = np.linspace(self.depth_min, self.depth_max, self.n_bins + 1)
        elif self.bin_mode == 'log':
            self.bin_edges = np.logspace(
                np.log10(self.depth_min), 
                np.log10(self.depth_max), 
                self.n_bins + 1
            )
        elif self.bin_mode == 'sid':
            # Spacing-Increasing Discretization (DORN paper)
            t = np.linspace(0, 1, self.n_bins + 1)
            self.bin_edges = self.depth_min * (self.depth_max / self.depth_min) ** (t ** self.sid_alpha)
        else:
            raise ValueError(f"Unknown bin_mode: {self.bin_mode}")
        
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_edges = torch.from_numpy(self.bin_edges.astype(np.float32))
        self.bin_centers = torch.from_numpy(self.bin_centers.astype(np.float32))

    def depth_to_bins(self, depth):
        """Convert depth map to bin indices."""
        # depth: [C, H, W] or [H, W]
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        
        # Digitize
        bin_indices = torch.bucketize(depth, self.bin_edges[1:-1])  # [C, H, W]
        bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)
        
        return bin_indices.squeeze(0)  # Return as [H, W] for single channel

    def bins_to_depth(self, bin_indices):
        """Convert bin indices back to depth values."""
        return self.bin_centers[bin_indices]

    def __getitem__(self, idx):
        # Get base data
        if self.use_original_depth:
            audio, sparse_depth, original_depth = super().__getitem__(idx)
            bin_indices = self.depth_to_bins(sparse_depth)
            return audio, bin_indices, sparse_depth, original_depth
        else:
            audio, sparse_depth = super().__getitem__(idx)
            bin_indices = self.depth_to_bins(sparse_depth)
            return audio, bin_indices, sparse_depth

