"""Configuration loader."""

import os
from types import SimpleNamespace
import yaml


def load_config(config_name='foa', mode='train', experiment_name='default'):
    """Load configuration from config/{config_name}.yaml.

    Args:
        config_name: 'baseline' or 'foa' (selects config/baseline.yaml or config/foa.yaml)
        mode: 'train' or 'test'
        experiment_name: name of the experiment
    """
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
    config_path = os.path.join(config_dir, f'{config_name}.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}. "
                                f"Available: {', '.join(f.replace('.yaml','') for f in os.listdir(config_dir) if f.endswith('.yaml'))}")

    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    cfg = SimpleNamespace()
    cfg.dataset = SimpleNamespace(**raw['dataset'])
    cfg.model = SimpleNamespace(**raw['model'])

    mode_cfg = dict(raw.get(mode, {}))
    mode_cfg['mode'] = mode
    mode_cfg['experiment_name'] = experiment_name
    cfg.mode = SimpleNamespace(**mode_cfg)

    return cfg
