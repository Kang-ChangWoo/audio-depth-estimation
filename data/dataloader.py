"""DataLoader factory."""

from torch.utils.data import DataLoader
from .dataset import SoundSpacesDataset
from .dataset_rotated import SoundSpacesDatasetRotated


def _select_dataset_class(cfg):
    """Pick the FOA-rotated dataset when rotate_canonical is enabled."""
    if (getattr(cfg.dataset, 'rotate_canonical', False)
            and getattr(cfg.dataset, 'use_ambisonic', False)):
        return SoundSpacesDatasetRotated
    return SoundSpacesDataset


def make_dataloader(cfg, split, batch_size=None, shuffle=None):
    """Create a DataLoader for the given split.

    Args:
        cfg: config object with dataset and mode attributes
        split: 'train', 'val', or 'test'
        batch_size: override cfg.mode.batch_size if provided
        shuffle: override default (True for train, False otherwise)
    Returns:
        (dataset, dataloader) tuple
    """
    dataset_cls = _select_dataset_class(cfg)
    dataset = dataset_cls(cfg, split=split)
    if batch_size is None:
        batch_size = cfg.mode.batch_size
    if shuffle is None:
        shuffle = (split == 'train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=cfg.mode.num_threads, pin_memory=True)
    return dataset, loader
