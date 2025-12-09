"""
Example script for using Weights & Biases Sweeps (Weave) for hyperparameter tuning

This script demonstrates how to set up a sweep configuration for hyperparameter tuning
using W&B Sweeps, which integrates with Weave for advanced hyperparameter optimization.

Usage:
1. Install wandb: pip install wandb
2. Login to wandb: wandb login
3. Create a sweep: wandb sweep sweep_config.yaml
4. Run agents: wandb agent <sweep_id>
"""

import wandb

# Example sweep configuration for hyperparameter tuning
sweep_config = {
    'method': 'bayes',  # Options: 'grid', 'random', 'bayes'
    'metric': {
        'name': 'val/abs_rel',  # Metric to optimize (lower is better)
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 0.0001,
            'max': 0.01
        },
        'batch_size': {
            'values': [128, 256, 512]
        },
        'criterion': {
            'values': ['L1', 'SIlog']
        },
        'silog_lambda': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.9,
            'conditions': {
                'criterion': ['SIlog']  # Only used when criterion is SIlog
            }
        },
        'optimizer': {
            'values': ['Adam', 'AdamW']
        }
    }
}

# Alternative: Grid search configuration
grid_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val/abs_rel',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.002, 0.005]
        },
        'batch_size': {
            'values': [128, 256]
        },
        'criterion': {
            'values': ['L1', 'SIlog']
        },
        'optimizer': {
            'values': ['AdamW']
        }
    }
}

# Example: Random search with early termination
random_sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val/abs_rel',
        'goal': 'minimize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 20,
        'eta': 2
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 0.0001,
            'max': 0.01
        },
        'batch_size': {
            'values': [128, 256, 512]
        },
        'criterion': {
            'values': ['L1', 'SIlog']
        },
        'silog_lambda': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.9
        }
    }
}

if __name__ == '__main__':
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project='batvision-depth-estimation')
    print(f"Sweep created with ID: {sweep_id}")
    print(f"Run agents with: wandb agent {sweep_id}")





