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
    
    def __init__(self, cfg, annotation_file, location_blacklist=None):
   
        self.cfg = cfg
        self.root_dir = cfg.dataset.dataset_dir
        self.audio_format = cfg.dataset.audio_format

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
        # Access instance 
        instance = self.instances.iloc[idx]
        
        # Use original file names directly (no conversion needed)
        depth_filename = instance['depth file name']
        audio_filename = instance['audio file name']
        
        # Load path
        depth_path = os.path.join(self.root_dir, instance['depth path'], depth_filename)
        audio_path = os.path.join(self.root_dir, instance['audio path'], audio_filename)
        
        ## Depth
        # Load depth map
        depth = np.load(depth_path).astype(np.float32)
        depth = depth / 1000 # to go from mm to m
        if self.cfg.dataset.max_depth:
            depth[depth > self.cfg.dataset.max_depth] = self.cfg.dataset.max_depth 
        depth[depth < 0] = 0  # Clip negative values
        # Transform
        depth_transform = get_transform(self.cfg, convert=True, depth_norm = self.cfg.dataset.depth_norm)
        gt_depth = depth_transform(depth)
        
        ## Audio 
        # Load audio binaural waveform
        # Try different methods to load audio
        waveform = None
        sr = None
        
        # Method 1: Try torchaudio with default backend
        try:
            waveform, sr = torchaudio.load(audio_path)
        except (RuntimeError, ValueError) as e:
            # Method 2: Try with soundfile backend if available
            try:
                import torchaudio.backend.soundfile as soundfile_backend
                waveform, sr = torchaudio.load(audio_path, backend="soundfile")
            except:
                # Method 3: Use scipy.io.wavfile as fallback
                try:
                    from scipy.io import wavfile
                    # torch is already imported at the top of the file
                    sr, audio_data = wavfile.read(audio_path)
                    # Convert to torch tensor
                    if audio_data.ndim == 1:
                        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
                    else:
                        waveform = torch.from_numpy(audio_data.T).float()
                    # Normalize to [-1, 1] range
                    if waveform.dtype == torch.int16:
                        waveform = waveform / 32768.0
                    elif waveform.dtype == torch.int32:
                        waveform = waveform / 2147483648.0
                except Exception as e2:
                    # Method 4: Try soundfile library directly
                    try:
                        import soundfile as sf
                        audio_data, sr = sf.read(audio_path)
                        # torch is already imported at the top of the file
                        if audio_data.ndim == 1:
                            waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
                        else:
                            waveform = torch.from_numpy(audio_data.T).float()
                    except Exception as e3:
                        raise RuntimeError(
                            f"Could not load audio file {audio_path} with any method. "
                            f"Tried: torchaudio (error: {e}), scipy (error: {e2}), soundfile (error: {e3})"
                        )
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
                spec = self._get_melspectrogram(waveform, n_fft = n_fft, power = 1.0, win_length = win_length)
            else:
                # Use regular spectrogram
                spec = self._get_spectrogram(waveform, n_fft = n_fft, power = 1.0, win_length = win_length, hop_length =  hop_length)
            
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
            
            spec_transform =  get_transform(self.cfg, convert = False) # convert False because already a tensor
            audio2return = spec_transform(spec)
        elif 'waveform' in self.audio_format:
            audio2return = waveform
        
        return audio2return, gt_depth
    
    # audio transformation: spectrogram
    def _get_spectrogram(self, waveform, n_fft = 400, power = 1.0, win_length = 400, hop_length=100): 

        spectrogram = T.Spectrogram(
          n_fft=n_fft,
          win_length=win_length,
          power=power,
          hop_length=hop_length,
        )
        #db = T.AmplitudeToDB(stype = 'magnitude')
        return spectrogram(waveform)
    
    # audio transformation: mel spectrogram
    def _get_melspectrogram(self, waveform, n_fft = 400, power = 1.0, win_length = 400, f_min = 20.0, f_max = 20000.0): 

        melspectrogram = T.MelSpectrogram(sample_rate = 44100, 
          n_fft=n_fft,
          win_length=win_length,
          power=power,
          f_min = f_min, 
          f_max = f_max,
          n_mels = 32, 
        )
        return melspectrogram(waveform)
    