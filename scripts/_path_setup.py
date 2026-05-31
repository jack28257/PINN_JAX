"""Make the src-layout package importable when scripts are run from the repo."""

from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Set this before pnp_pinn imports JAX. CPU is currently the reliable local M4
# backend for this nested-derivative PINN.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
