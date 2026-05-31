"""Checkpoint and metadata helpers."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
from typing import Any, Mapping

import equinox as eqx
import jax
import numpy as np

from .config import PNPConfig, TrainConfig, high_quality_configs
from .model import PNP
from .types import Array


def save_pytree(pytree: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, pytree)


def load_pytree(path: str | Path, template: Any) -> Any:
    return eqx.tree_deserialise_leaves(path, template)


def save_model(model: PNP, path: str | Path) -> None:
    """Save model weights for evaluation or model-weight continuation."""

    save_pytree(model, path)


def load_model(path: str | Path, cfg: PNPConfig, key: Array | None = None) -> PNP:
    """Load weights into a freshly-created PNP template."""

    if key is None:
        key = jax.random.PRNGKey(0)
    model = PNP(cfg, key)
    return load_pytree(path, model)


def make_run_dir(label: str = "pnp_1d_high_quality", base_dir: str | Path = "checkpoints/runs") -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{label}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2, default=_to_jsonable)
    tmp_path.replace(path)


def save_history(history: list[dict[str, float]], run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(history, run_dir / "history.json")

    if not history:
        return

    fieldnames = list(history[0].keys())
    csv_path = run_dir / "history.csv"
    tmp_path = csv_path.with_name(csv_path.name + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    tmp_path.replace(csv_path)


def load_history(run_dir: str | Path) -> list[dict[str, float]]:
    history_path = Path(run_dir) / "history.json"
    if not history_path.exists():
        return []
    with history_path.open() as f:
        return json.load(f)


def _pnp_config_from_dict(data: Mapping[str, Any]) -> PNPConfig:
    data = dict(data)
    for name in ["dom_resid_weights", "bl_resid_weights", "bc_resid_weights"]:
        if name in data:
            data[name] = tuple(data[name])
    data.pop("ic_resid_weights", None)
    return PNPConfig(**data)


def _train_config_from_dict(data: Mapping[str, Any], run_dir: str | Path) -> TrainConfig:
    data = dict(data)
    data.pop("ic_batch_size", None)
    data["checkpoint_path"] = str(Path(run_dir) / "model_latest.eqx")
    return TrainConfig(**data)


def load_run_configs(run_dir: str | Path) -> tuple[PNPConfig, TrainConfig]:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return high_quality_configs(run_dir)
    with metadata_path.open() as f:
        metadata = json.load(f)
    return (
        _pnp_config_from_dict(metadata["pnp_config"]),
        _train_config_from_dict(metadata["train_config"], run_dir),
    )


def save_run_metadata(
    pnp_cfg: PNPConfig,
    train_cfg: TrainConfig,
    run_dir: str | Path,
    *,
    status: str,
    latest_step: int,
) -> None:
    run_dir = Path(run_dir)
    metadata = {
        "status": status,
        "latest_step": latest_step,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "pnp_config": asdict(pnp_cfg),
        "train_config": asdict(train_cfg),
        "checkpoint_path": train_cfg.checkpoint_path,
        "model_latest": str(run_dir / "model_latest.eqx"),
        "model_final": str(run_dir / "model_final.eqx"),
        "model_interrupted": str(run_dir / "model_interrupted.eqx"),
        "training_state_latest": str(run_dir / "training_state_latest.eqx"),
        "training_state_final": str(run_dir / "training_state_final.eqx"),
        "training_state_interrupted": str(run_dir / "training_state_interrupted.eqx"),
        "history_json": str(run_dir / "history.json"),
        "history_csv": str(run_dir / "history.csv"),
    }
    save_json(metadata, run_dir / "run_metadata.json")


def print_saved_artifacts(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    print(f"Run artifacts saved in: {run_dir}")
    for artifact in sorted(run_dir.glob("*")):
        print(f"  - {artifact.name}")
