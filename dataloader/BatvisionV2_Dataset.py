import os
import torch
import pandas as pd
import torchaudio
import cv2
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np

from .utils_dataset import get_transform

class BatvisionV2Dataset(Dataset):
    def __init__(self, cfg, annotation_file, location_blacklist=None, use_image=False):
   
        self.cfg = cfg
        self.root_dir = cfg.dataset.dataset_dir
        self.audio_format = cfg.dataset.audio_format
        self.use_image = use_image  # If True, load camera images instead of audio

        # Only include directories, not files
        # Exclude special directories like __pycache__, hidden directories, _unzipped directories, etc.
        location_list = [item for item in os.listdir(self.root_dir) 
                        if os.path.isdir(os.path.join(self.root_dir, item))
                        and not item.startswith('.')
                        and not item.startswith('__')
                        and not item.endswith('_unzipped')]  # Exclude _unzipped directories
        
        if location_blacklist:
            location_list = [location for location in location_list if location not in location_blacklist]
        
        # Only include locations that have the annotation file
        location_csv_paths = []
        for location in location_list:
            csv_path = os.path.join(self.root_dir, location, annotation_file)
            if os.path.exists(csv_path):
                location_csv_paths.append(csv_path)
            else:
                print(f"Warning: {csv_path} not found, skipping location {location}")
        
        if len(location_csv_paths) == 0:
            raise ValueError(f"No valid locations found with {annotation_file} in {self.root_dir}. "
                           f"Checked {len(location_list)} directories: {location_list[:5]}...")
                
        self.instances = []
        
        for location_csv in location_csv_paths:
            self.instances.append(pd.read_csv(location_csv))
            
        self.instances = pd.concat(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances.iloc[idx]

        """
        Depth loading
        - Load depth map
        - Convert to meters
        - Clip depth values to max depth
        - Resize depth map to image size
        - Convert to tensor
        """
        depth_filename = instance['depth file name']        
        depth_path = os.path.join(self.root_dir, instance['depth path'], depth_filename)
        
        depth = np.load(depth_path).astype(np.float32)
        depth = depth / 1000.0  # mm to m
        if self.cfg.dataset.max_depth:
            depth[depth > self.cfg.dataset.max_depth] = self.cfg.dataset.max_depth
        depth[depth < 0] = 0
        

        
        depth = cv2.resize(depth, (self.cfg.dataset.images_size, self.cfg.dataset.images_size), 
                           interpolation=cv2.INTER_NEAREST)
        gt_depth = torch.from_numpy(depth).unsqueeze(0)
        

        """
        Image or Audio loading (if needed)
        - If use_image is True, load camera image
        - If use_image is False, load audio
        - Return image or audio tensor
        """
        if self.use_image:
            image_filename = instance['camera file name']
            image_path = os.path.join(self.root_dir, instance['camera path'], image_filename)
            input_data = self._load_image(image_path)
        else:
            audio_filename = instance['audio file name']
            audio_path = os.path.join(self.root_dir, instance['audio path'], audio_filename)
            waveform, sr = self._load_audio(audio_path)

            # STFT parameters for full length audio
            win_length = 200 
            n_fft = 400
            hop_length = 100

            # Cut audio to fit max depth
            if self.cfg.dataset.max_depth:
                cut = int((2*self.cfg.dataset.max_depth / 340) * sr)
                waveform = waveform[:,:cut]
                # Update STFT parameters 
                win_length = 64
                n_fft = 512
                hop_length=64//4

            # Process sound
            if 'spectrogram' in self.audio_format:
                if 'mel' in self.audio_format:
                    # Use mel spectrogram
                    spec = self._get_melspectrogram(waveform, n_fft=n_fft, power=1.0, win_length=win_length)
                else:
                    # Use regular spectrogram
                    spec = self._get_spectrogram(waveform, n_fft=n_fft, power=1.0, win_length=win_length, hop_length=hop_length)
                
                # Apply log scale (log mel spectrogram or log spectrogram)
                # This is necessary because raw spectrogram values are too large (e.g., 172877545472)
                # Without log scaling, the model cannot learn properly
                spec = torch.log(spec + 1e-8)
                
                # Min-max normalize each channel independently to [0, 1]
                # This ensures consistent input range for the model
                for c in range(spec.shape[0]):
                    spec_min = spec[c].min()
                    spec_max = spec[c].max()
                    if spec_max > spec_min:
                        spec[c] = (spec[c] - spec_min) / (spec_max - spec_min)
                    else:
                        spec[c] = torch.zeros_like(spec[c])
                
                spec_transform = get_transform(self.cfg, convert=False)  # convert False because already a tensor
                input_data = spec_transform(spec)
            elif 'waveform' in self.audio_format:
                input_data = waveform
        
        return input_data, gt_depth
    

    def _load_audio(self, audio_path):
        """Load audio with multiple fallback methods"""
        try:
            waveform, sr = torchaudio.load(audio_path)
        except (RuntimeError, ValueError) as e:
            try:
                import torchaudio.backend.soundfile as soundfile_backend
                waveform, sr = torchaudio.load(audio_path, backend="soundfile")
            except:
                try:
                    from scipy.io import wavfile
                    sr, audio_data = wavfile.read(audio_path)
                    if audio_data.ndim == 1:
                        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
                    else:
                        waveform = torch.from_numpy(audio_data.T).float()
                    if waveform.dtype == torch.int16:
                        waveform = waveform / 32768.0
                    elif waveform.dtype == torch.int32:
                        waveform = waveform / 2147483648.0
                except Exception as e2:
                    try:
                        import soundfile as sf
                        audio_data, sr = sf.read(audio_path)
                        if audio_data.ndim == 1:
                            waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
                        else:
                            waveform = torch.from_numpy(audio_data.T).float()
                    except Exception as e3:
                        raise RuntimeError(
                            f"Could not load audio file {audio_path} with any method. "
                            f"Tried: torchaudio (error: {e}), scipy (error: {e2}), soundfile (error: {e3})"
                        )
        return waveform, sr
    
    def _get_spectrogram(self, waveform, n_fft=400, power=1.0, win_length=400, hop_length=100): 

        spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            power=power,
            hop_length=hop_length,
        )
        return spectrogram(waveform)
    
    def _get_melspectrogram(self, waveform, n_fft=400, power=1.0, win_length=400, f_min=20.0, f_max=20000.0):
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
    
    def _load_image(self, image_path):
        """Load and process camera image - Match notebook exactly"""
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Could not load image file {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.cfg.dataset.images_size, self.cfg.dataset.images_size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    