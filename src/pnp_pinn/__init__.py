"""Local JAX/Equinox PINN package for the 1D Poisson-Nernst-Planck problem."""

from __future__ import annotations

import os


# JAX chooses its backend when it is imported. On the local M4, CPU is the stable
# default for the nested automatic differentiation used by this PINN.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from .checkpointing import load_model, make_run_dir, print_saved_artifacts, save_model
from .config import (
    PNPConfig,
    TrainConfig,
    high_quality_configs,
    high_quality_hard_phi_configs,
    local_good_configs,
    local_good_hard_phi_configs,
    local_smoke_configs,
)
from .losses import loss_terms, total_loss
from .model import PNP
from .train import load_resume_state, resume_saved_run, train

__all__ = [
    "PNP",
    "PNPConfig",
    "TrainConfig",
    "high_quality_configs",
    "high_quality_hard_phi_configs",
    "local_good_configs",
    "local_good_hard_phi_configs",
    "local_smoke_configs",
    "load_model",
    "save_model",
    "make_run_dir",
    "print_saved_artifacts",
    "loss_terms",
    "total_loss",
    "train",
    "load_resume_state",
    "resume_saved_run",
]
