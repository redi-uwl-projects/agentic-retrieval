# Utility function helpers: logging, seeding, device detection.

import logging
import random
import sys

import numpy as np
import torch

# Console Logging for log output
def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    
    if log_level.upper() != "DEBUG":
        for noisy in ("urllib3", "filelock", "sentence_transformers", "huggingface_hub", "datasets", "fsspec"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

# The set_seed function to ensure reproducibility across runs.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except ImportError:
        pass

# Check if the requested GPU device is available, otherwise fallback to CPU.
def detect_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
