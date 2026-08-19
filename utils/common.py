"""
utils/common.py
----------------
Small shared helpers used across train.py / evaluate.py / inference.py.
"""

import os
import random
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    """Return a usable torch.device, falling back to CPU if CUDA is unavailable."""
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available -> falling back to CPU.")
        return torch.device("cpu")
    return torch.device(preferred)


def save_checkpoint(state: dict, path) -> None:
    """Atomically save a checkpoint dict to `path`."""
    tmp_path = str(path) + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logging(log_file=None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def make_dataloader(dataset, batch_size: int, shuffle: bool = False, sampler=None,
                     num_workers: int = 0, drop_last: bool = False,
                     pin_memory: bool = True) -> DataLoader:
    """
    Thin wrapper around torch.utils.data.DataLoader that avoids a classic
    PyTorch deadlock: when a CUDA context has already been initialized in
    the main process (e.g. via `model.to('cuda')`, which every training/eval
    script here does before its first DataLoader iteration) and worker
    processes are then created with the default 'fork' start method on
    Linux, the forked workers can hang indefinitely on first iteration --
    with no error, no traceback, just silence. This is a well-documented
    PyTorch + CUDA + multiprocessing interaction, not a bug in this
    project's data loading logic.

    Using the 'spawn' start method for worker processes avoids this: each
    worker gets a fresh Python interpreter that never inherits the parent's
    CUDA context. The trade-off is slower worker startup (each spawned
    worker re-imports this project's modules, including gudhi/ripser/timm),
    so `persistent_workers=True` is enabled alongside it -- workers are
    created once and reused across epochs instead of being torn down and
    respawned every epoch, which would otherwise repeatedly pay that
    spawn cost.
    """
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin_memory, drop_last=drop_last)
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)
