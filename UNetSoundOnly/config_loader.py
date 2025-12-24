"""
Simple config loader to replace Hydra
"""
import os
import re
from types import SimpleNamespace

def _parse_yaml_simple(filepath):
    """Simple YAML parser without external dependencies"""
    config = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse key: value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Remove inline comments
                if '#' in value:
                    value = value.split('#')[0].strip()
                # Convert value types
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'null' or value == '':
                    value = None
                elif value.isdigit():
                    value = int(value)
                elif re.match(r'^-?\d+\.\d+$', value):
                    value = float(value)
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                config[key] = value
    return config

def load_config(dataset_name='batvisionv2', mode='train', experiment_name='default', model_name='unet_baseline'):
    """
    Load configuration from yaml files
    
    Args:
        dataset_name: 'batvisionv1' or 'batvisionv2'
        mode: 'train' or 'test'
        experiment_name: name of the experiment
        model_name: 'unet_baseline' or 'spline_depth'
    """
    try:
        import yaml
        use_yaml = True
    except ImportError:
        use_yaml = False
    
    conf_dir = os.path.join(os.path.dirname(__file__), 'conf')
    
    # Load dataset config
    dataset_file = os.path.join(conf_dir, 'dataset', f'{dataset_name}.yaml')
    if use_yaml:
        with open(dataset_file, 'r') as f:
            dataset_cfg = yaml.safe_load(f)
    else:
        dataset_cfg = _parse_yaml_simple(dataset_file)
    
    # Load mode config
    mode_file = os.path.join(conf_dir, 'mode', f'{mode}.yaml')
    if use_yaml:
        with open(mode_file, 'r') as f:
            mode_cfg = yaml.safe_load(f)
    else:
        mode_cfg = _parse_yaml_simple(mode_file)
    
    # Load model config (support both unet_baseline and spline_depth)
    model_file = os.path.join(conf_dir, 'model', f'{model_name}.yaml')
    if not os.path.exists(model_file):
        # Fallback to unet_baseline
        model_file = os.path.join(conf_dir, 'model', 'unet_baseline.yaml')
    
    if use_yaml:
        with open(model_file, 'r') as f:
            model_cfg = yaml.safe_load(f)
    else:
        model_cfg = _parse_yaml_simple(model_file)
    
    # Combine configs into a nested namespace
    cfg = SimpleNamespace()
    cfg.dataset = SimpleNamespace(**dataset_cfg)
    cfg.mode = SimpleNamespace(**mode_cfg)
    cfg.mode.mode = mode  # Ensure mode is set
    cfg.mode.experiment_name = experiment_name
    cfg.model = SimpleNamespace(**model_cfg)
    
    return cfg