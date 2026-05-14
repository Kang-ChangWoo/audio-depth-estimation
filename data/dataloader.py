"""DataLoader factory."""

from torch.utils.data import DataLoader
from .dataset import SoundSpacesDataset, SoundSpacesDatasetRotated


def _select_dataset_class(cfg):
    """Pick dataset class based on config flags.

    RGB input (use_rgb) is no longer a separate class — it's handled
    inline by an if-branch in SoundSpacesDataset.__getitem__, so only
    the FOA-rotation choice affects class selection here.
    """
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
    nw = int(cfg.mode.num_threads)
    is_train = (split == 'train')

    # Val / test DIFFER from train in two specific knobs:
    #   pin_memory=False
    #   persistent_workers=False
    #
    # Background: exp360–363 hung at the first validation pass despite
    # DataLoader(timeout=120). /proc diagnostics (wchan =
    # futex_wait_queue_me, CPU=159%) showed the main thread blocked on
    # the *pin_memory* queue — the `timeout` kwarg only guards the
    # worker queue one hop upstream. With pin_memory=True the pin
    # thread sits between the worker and the main thread and the main
    # thread's read is unguarded.
    #
    # With pin_memory=False the main thread reads from the worker
    # queue directly, so timeout=120 now actually protects it: a dead
    # val worker raises RuntimeError after 2 min instead of hanging.
    # persistent_workers=False tears down the val pool between val
    # runs so a death in one val doesn't carry over.
    #
    # Train keeps pin_memory=True + persistent_workers=True: it runs
    # hot and throughput matters. Val/test are not bottlenecked on
    # data loading, so we trade a little speed for a lot of safety.
    pm = is_train
    persistent = is_train and (nw > 0)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=nw, pin_memory=pm,
                        persistent_workers=persistent,
                        prefetch_factor=(4 if nw > 0 else None),
                        timeout=(120 if nw > 0 else 0))
    return dataset, loader
