"""
Utility functions: device detection, reproducibility, metrics, logging.
"""

import os
import random
import json
from pathlib import Path

import numpy as np
import torch


# Device

def get_device():
    """Auto-detect best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# Reproducibility

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: MPS does not support manual_seed_all as of PyTorch 2.x
    # torch.backends.cudnn settings for CUDA reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Paths

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
DATA_DIR = PROJECT_ROOT / "data"

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True)


# Results I/O

def save_results(results: dict, filename: str):
    """Save experiment results as JSON."""
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def load_results(filename: str) -> dict:
    """Load experiment results from JSON."""
    path = RESULTS_DIR / filename
    with open(path, "r") as f:
        return json.load(f)


# Metrics

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0)
