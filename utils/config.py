"""Configuration loader."""

import os
from types import SimpleNamespace
import yaml


def load_config(mode='train', experiment_name='default'):
    """Load configuration from config.yaml.

    Args:
        mode: 'train' or 'test'
        experiment_name: name of the experiment
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
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
