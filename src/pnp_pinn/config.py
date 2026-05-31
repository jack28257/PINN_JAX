"""Configuration objects and standard local training presets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class PNPConfig:
    x_min: float = 0.0
    x_max: float = 1.0
    t_max: float = 0.2

    dp_min: float = 0.5
    dp_max: float = 2.0
    dn_min: float = 0.5
    dn_max: float = 2.0

    zp: float = 1.0
    zn: float = -1.0
    epsilon: float = 1.0

    cp_init: float = 1.0
    cn_init: float = 1.0
    v_left: float = -0.5
    v_right: float = 0.5

    width_size: int = 256
    depth: int = 8
    hard_ic: bool = True
    hard_phi_bc: bool = False
    normalize_inputs: bool = True
    boundary_layer_width: float = 0.05

    dom_resid_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    bl_resid_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    bc_resid_weights: tuple[float, float, float] = (50.0, 50.0, 10.0)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    max_steps: int = 1_000
    dom_batch_size: int = 1_024
    bl_batch_size: int = 1_024
    bc_batch_size: int = 512
    learning_rate: float = 5e-5
    lr_decay_rate: float = 0.95
    lr_decay_steps: int = 2_000
    max_grad_norm: float = 10.0
    print_every: int = 100
    use_ntk_weights: bool = False
    ntk_update_every: int = 100
    checkpoint_every: int = 0
    checkpoint_path: str | None = None


def local_smoke_configs() -> tuple[PNPConfig, TrainConfig]:
    """Tiny configuration for import checks and quick smoke tests."""

    pnp_cfg = PNPConfig(width_size=32, depth=3)
    train_cfg = TrainConfig(
        max_steps=3,
        dom_batch_size=16,
        bl_batch_size=16,
        bc_batch_size=8,
        learning_rate=1e-3,
        print_every=1,
    )
    return pnp_cfg, train_cfg


def local_good_configs(run_dir: str | Path | None = None) -> tuple[PNPConfig, TrainConfig]:
    """Moderate local run that is useful before committing to a long run."""

    checkpoint_path = (
        Path(run_dir) / "model_latest.eqx"
        if run_dir is not None
        else Path("checkpoints") / "pnp_1d_local_latest.eqx"
    )
    pnp_cfg = PNPConfig(width_size=128, depth=6, hard_ic=True, normalize_inputs=True)
    train_cfg = TrainConfig(
        seed=0,
        max_steps=50_000,
        dom_batch_size=512,
        bl_batch_size=512,
        bc_batch_size=512,
        learning_rate=1e-3,
        lr_decay_rate=0.95,
        lr_decay_steps=2_000,
        max_grad_norm=10.0,
        print_every=500,
        use_ntk_weights=False,
        ntk_update_every=250,
        checkpoint_every=2_500,
        checkpoint_path=str(checkpoint_path),
    )
    return pnp_cfg, train_cfg


def high_quality_configs(run_dir: str | Path | None = None) -> tuple[PNPConfig, TrainConfig]:
    """High-capacity configuration intended for the main local training run."""

    checkpoint_path = (
        Path(run_dir) / "model_latest.eqx"
        if run_dir is not None
        else Path("checkpoints") / "pnp_1d_high_quality_latest.eqx"
    )
    pnp_cfg = PNPConfig(
        width_size=256,
        depth=8,
        hard_ic=True,
        normalize_inputs=True,
        boundary_layer_width=0.05,
        dom_resid_weights=(1.0, 1.0, 1.0),
        bl_resid_weights=(1.0, 1.0, 1.0),
        bc_resid_weights=(50.0, 50.0, 10.0),
    )
    train_cfg = TrainConfig(
        seed=0,
        max_steps=500_000,
        dom_batch_size=512,
        bl_batch_size=512,
        bc_batch_size=512,
        learning_rate=5e-4,
        lr_decay_rate=0.95,
        lr_decay_steps=2_000,
        max_grad_norm=10.0,
        print_every=1_000,
        use_ntk_weights=True,
        ntk_update_every=500,
        checkpoint_every=10_000,
        checkpoint_path=str(checkpoint_path),
    )
    return pnp_cfg, train_cfg


def local_good_hard_phi_configs(run_dir: str | Path | None = None) -> tuple[PNPConfig, TrainConfig]:
    """Moderate local run with the voltage boundary condition enforced exactly."""

    pnp_cfg, train_cfg = local_good_configs(run_dir)
    pnp_cfg = replace(
        pnp_cfg,
        hard_phi_bc=True,
        bc_resid_weights=(50.0, 50.0, 0.0),
    )
    return pnp_cfg, train_cfg


def high_quality_hard_phi_configs(run_dir: str | Path | None = None) -> tuple[PNPConfig, TrainConfig]:
    """Main local training preset for the hard-constrained voltage trial."""

    pnp_cfg, train_cfg = high_quality_configs(run_dir)
    pnp_cfg = replace(
        pnp_cfg,
        hard_phi_bc=True,
        bc_resid_weights=(50.0, 50.0, 0.0),
    )
    return pnp_cfg, train_cfg
