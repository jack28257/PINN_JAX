"""Phase A diagnostics for trained PNP PINNs.

These checks are PINN-only: they do not compare against a finite-difference
reference solver yet. They verify training history, held-out residuals, boundary
conditions, the hard initial condition, and basic physical sanity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .checkpointing import load_history, load_model, load_run_configs, save_json
from .config import PNPConfig
from .model import PNP
from .sampling import sample_bc, sample_bl, sample_dom, sample_theta
from .types import Array, ConstraintBatch


DEFAULT_PARAMETER_CASES: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),
    (2.0, 2.0),
    (0.5, 2.0),
    (2.0, 0.5),
    (1.25, 1.25),
)


def _float(value: Any) -> float:
    return float(np.asarray(value))


def _array_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    abs_values = np.abs(values)
    return {
        "mean": float(np.mean(values)),
        "mean_abs": float(np.mean(abs_values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "median_abs": float(np.median(abs_values)),
        "p95_abs": float(np.percentile(abs_values, 95)),
        "p99_abs": float(np.percentile(abs_values, 99)),
        "max_abs": float(np.max(abs_values)),
    }


def _component_stats(values: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    values = np.asarray(values)
    return {name: _array_stats(values[:, i]) for i, name in enumerate(names)}


def _total_rms(component_summary: dict[str, dict[str, float]]) -> float:
    return float(np.sqrt(sum(stats["rms"] ** 2 for stats in component_summary.values())))


def _worst_boundary_examples(
    residuals: np.ndarray,
    *,
    names: list[str],
    side: str,
    t: Array,
    theta: dict[str, Array],
) -> dict[str, dict[str, float | str]]:
    t_np = np.asarray(t)
    dp_np = np.asarray(theta["Dp"])
    dn_np = np.asarray(theta["Dn"])
    examples: dict[str, dict[str, float | str]] = {}
    for i, name in enumerate(names):
        values = residuals[:, i]
        idx = int(np.argmax(np.abs(values)))
        examples[name] = {
            "side": side,
            "value": float(values[idx]),
            "abs_value": float(abs(values[idx])),
            "t": float(t_np[idx]),
            "Dp": float(dp_np[idx]),
            "Dn": float(dn_np[idx]),
        }
    return examples


def summarize_training_history(run_dir: str | Path) -> dict[str, Any]:
    history = load_history(run_dir)
    if not history:
        return {"available": False}

    first = history[0]
    final = history[-1]
    best = min(history, key=lambda row: float(row["loss"]))
    summary = {
        "available": True,
        "num_logged_rows": len(history),
        "first_step": int(float(first["step"])),
        "final_step": int(float(final["step"])),
        "first_loss": float(first["loss"]),
        "final_loss": float(final["loss"]),
        "best_loss": float(best["loss"]),
        "best_loss_step": int(float(best["step"])),
        "loss_decrease_ratio": float(first["loss"]) / max(float(final["loss"]), 1e-30),
        "final_dom": float(final["dom"]),
        "final_bl": float(final["bl"]),
        "final_bc": float(final["bc"]),
    }
    if "average_steps_per_second" in final:
        summary["average_steps_per_second"] = float(final["average_steps_per_second"])
    if "elapsed_seconds" in final:
        summary["elapsed_seconds"] = float(final["elapsed_seconds"])
    return summary


def _batched_residuals(model: PNP, batch: ConstraintBatch, resid_name: str) -> np.ndarray:
    resid_fn = getattr(model, resid_name)
    residuals = jax.vmap(resid_fn)(batch.x, batch.t, batch.theta)
    residuals.block_until_ready()
    return np.asarray(residuals)


def held_out_residual_diagnostics(
    model: PNP,
    cfg: PNPConfig,
    *,
    seed: int = 123,
    n_domain: int = 2048,
    n_boundary_layer: int = 1024,
    n_boundary: int = 1024,
) -> dict[str, Any]:
    key = jax.random.PRNGKey(seed)
    key_dom, key_bl, key_bc = jax.random.split(key, 3)

    dom_batch = sample_dom(key_dom, n_domain, cfg)
    bl_batch = sample_bl(key_bl, n_boundary_layer, cfg)
    bc_batch = sample_bc(key_bc, n_boundary, cfg)

    dom = _batched_residuals(model, dom_batch, "dom_resid")
    bl = _batched_residuals(model, bl_batch, "dom_resid")
    bc = _batched_residuals(model, bc_batch, "bc_resid")

    dom_stats = _component_stats(dom, ["np_positive", "np_negative", "poisson"])
    bl_stats = _component_stats(bl, ["np_positive", "np_negative", "poisson"])
    bc_stats = _component_stats(bc, ["flux_positive", "flux_negative", "phi_error"])

    return {
        "seed": seed,
        "sample_counts": {
            "domain": n_domain,
            "boundary_layer": n_boundary_layer,
            "boundary": n_boundary,
        },
        "domain": dom_stats,
        "boundary_layer": bl_stats,
        "boundary": bc_stats,
        "domain_total_rms": _total_rms(dom_stats),
        "boundary_layer_total_rms": _total_rms(bl_stats),
        "boundary_total_rms": _total_rms(bc_stats),
    }


def boundary_condition_diagnostics(
    model: PNP,
    cfg: PNPConfig,
    *,
    seed: int = 456,
    n_per_side: int = 1024,
) -> dict[str, Any]:
    key = jax.random.PRNGKey(seed)
    key_t, key_theta = jax.random.split(key)
    t = jax.random.uniform(key_t, (n_per_side,), minval=0.0, maxval=cfg.t_max)
    theta = sample_theta(key_theta, n_per_side, cfg)

    left_x = jnp.full((n_per_side,), cfg.x_min)
    right_x = jnp.full((n_per_side,), cfg.x_max)
    left = _batched_residuals(model, ConstraintBatch(left_x, t, theta), "bc_resid")
    right = _batched_residuals(model, ConstraintBatch(right_x, t, theta), "bc_resid")

    names = ["flux_positive", "flux_negative", "phi_error"]
    return {
        "seed": seed,
        "n_per_side": n_per_side,
        "left": _component_stats(left, names),
        "right": _component_stats(right, names),
        "combined": _component_stats(np.concatenate([left, right], axis=0), names),
        "worst_examples": {
            "left": _worst_boundary_examples(left, names=names, side="left", t=t, theta=theta),
            "right": _worst_boundary_examples(right, names=names, side="right", t=t, theta=theta),
        },
    }


def initial_condition_diagnostics(
    model: PNP,
    cfg: PNPConfig,
    *,
    seed: int = 789,
    n_points: int = 2048,
) -> dict[str, Any]:
    key = jax.random.PRNGKey(seed)
    key_x, key_theta = jax.random.split(key)
    x = jax.random.uniform(key_x, (n_points,), minval=cfg.x_min, maxval=cfg.x_max)
    t = jnp.zeros((n_points,))
    theta = sample_theta(key_theta, n_points, cfg)
    u = jax.vmap(model)(x, t, theta)
    u.block_until_ready()
    u_np = np.asarray(u)
    errors = np.stack(
        [
            u_np[:, 0] - cfg.cp_init,
            u_np[:, 1] - cfg.cn_init,
        ],
        axis=1,
    )
    return {
        "seed": seed,
        "n_points": n_points,
        "components": _component_stats(errors, ["cp_minus_initial", "cn_minus_initial"]),
    }


def physical_sanity_diagnostics(
    model: PNP,
    cfg: PNPConfig,
    *,
    parameter_cases: tuple[tuple[float, float], ...] = DEFAULT_PARAMETER_CASES,
    nx: int = 128,
    nt: int = 41,
) -> dict[str, Any]:
    x_grid = jnp.linspace(cfg.x_min, cfg.x_max, nx)
    t_grid = jnp.linspace(0.0, cfg.t_max, nt)
    x_np = np.asarray(x_grid)
    target_cp_mass = cfg.cp_init * (cfg.x_max - cfg.x_min)
    target_cn_mass = cfg.cn_init * (cfg.x_max - cfg.x_min)

    cases: list[dict[str, Any]] = []
    global_cp_min = np.inf
    global_cn_min = np.inf
    global_phi_abs_max = 0.0
    global_mass_drift_cp = 0.0
    global_mass_drift_cn = 0.0

    for dp, dn in parameter_cases:
        theta = {"Dp": jnp.array(dp), "Dn": jnp.array(dn)}

        def eval_at_t(tt: Array) -> Array:
            return jax.vmap(lambda xx: model(xx, tt, theta))(x_grid)

        values = jax.vmap(eval_at_t)(t_grid)
        values.block_until_ready()
        values_np = np.asarray(values)
        cp = values_np[:, :, 0]
        cn = values_np[:, :, 1]
        phi = values_np[:, :, 2]

        cp_mass = np.trapezoid(cp, x_np, axis=1)
        cn_mass = np.trapezoid(cn, x_np, axis=1)
        cp_mass_drift = np.max(np.abs(cp_mass - target_cp_mass)) / max(abs(target_cp_mass), 1e-30)
        cn_mass_drift = np.max(np.abs(cn_mass - target_cn_mass)) / max(abs(target_cn_mass), 1e-30)

        case = {
            "Dp": float(dp),
            "Dn": float(dn),
            "cp_min": float(np.min(cp)),
            "cp_max": float(np.max(cp)),
            "cn_min": float(np.min(cn)),
            "cn_max": float(np.max(cn)),
            "phi_min": float(np.min(phi)),
            "phi_max": float(np.max(phi)),
            "max_abs_phi": float(np.max(np.abs(phi))),
            "cp_mass_initial": float(cp_mass[0]),
            "cp_mass_final": float(cp_mass[-1]),
            "cp_mass_relative_drift_max": float(cp_mass_drift),
            "cn_mass_initial": float(cn_mass[0]),
            "cn_mass_final": float(cn_mass[-1]),
            "cn_mass_relative_drift_max": float(cn_mass_drift),
        }
        cases.append(case)

        global_cp_min = min(global_cp_min, case["cp_min"])
        global_cn_min = min(global_cn_min, case["cn_min"])
        global_phi_abs_max = max(global_phi_abs_max, case["max_abs_phi"])
        global_mass_drift_cp = max(global_mass_drift_cp, case["cp_mass_relative_drift_max"])
        global_mass_drift_cn = max(global_mass_drift_cn, case["cn_mass_relative_drift_max"])

    return {
        "grid": {"nx": nx, "nt": nt},
        "cases": cases,
        "summary": {
            "cp_min_global": float(global_cp_min),
            "cn_min_global": float(global_cn_min),
            "max_abs_phi_global": float(global_phi_abs_max),
            "cp_mass_relative_drift_max": float(global_mass_drift_cp),
            "cn_mass_relative_drift_max": float(global_mass_drift_cn),
        },
    }


def make_phase_a_report(summary: dict[str, Any]) -> str:
    training = summary["training"]
    residuals = summary["held_out_residuals"]
    boundary = summary["boundary_conditions"]["combined"]
    initial = summary["initial_condition"]["components"]
    physical = summary["physical_sanity"]["summary"]
    worst_left_phi = summary["boundary_conditions"]["worst_examples"]["left"]["phi_error"]
    worst_right_phi = summary["boundary_conditions"]["worst_examples"]["right"]["phi_error"]

    lines = [
        "# Phase A PINN Diagnostics",
        "",
        f"Run directory: `{summary['run_dir']}`",
        f"Checkpoint: `{summary['checkpoint']}`",
        "",
        "## Training Summary",
        "",
        f"- Status: `{summary['metadata_status']}`",
        f"- Final step: `{training.get('final_step', 'unknown')}`",
        f"- Final loss: `{training.get('final_loss', float('nan')):.6e}`",
        f"- Final dom/bl/bc: `{training.get('final_dom', float('nan')):.6e}`, "
        f"`{training.get('final_bl', float('nan')):.6e}`, "
        f"`{training.get('final_bc', float('nan')):.6e}`",
        f"- Best loss: `{training.get('best_loss', float('nan')):.6e}` "
        f"at step `{training.get('best_loss_step', 'unknown')}`",
        "",
        "## Held-Out Residual RMS",
        "",
        f"- Domain total RMS: `{residuals['domain_total_rms']:.6e}`",
        f"- Boundary-layer total RMS: `{residuals['boundary_layer_total_rms']:.6e}`",
        f"- Boundary total RMS: `{residuals['boundary_total_rms']:.6e}`",
        "",
        "## Boundary Combined Max Errors",
        "",
        f"- Flux positive max abs: `{boundary['flux_positive']['max_abs']:.6e}`",
        f"- Flux negative max abs: `{boundary['flux_negative']['max_abs']:.6e}`",
        f"- Phi wall max abs: `{boundary['phi_error']['max_abs']:.6e}`",
        f"- Worst left phi error: `{worst_left_phi['value']:.6e}` at "
        f"`t={worst_left_phi['t']:.6e}`, "
        f"`Dp={worst_left_phi['Dp']:.3f}`, `Dn={worst_left_phi['Dn']:.3f}`",
        f"- Worst right phi error: `{worst_right_phi['value']:.6e}` at "
        f"`t={worst_right_phi['t']:.6e}`, "
        f"`Dp={worst_right_phi['Dp']:.3f}`, `Dn={worst_right_phi['Dn']:.3f}`",
        "",
        "## Initial Condition Max Errors",
        "",
        f"- cp(x,0)-cp_init max abs: `{initial['cp_minus_initial']['max_abs']:.6e}`",
        f"- cn(x,0)-cn_init max abs: `{initial['cn_minus_initial']['max_abs']:.6e}`",
        "",
        "## Physical Sanity",
        "",
        f"- Global cp min: `{physical['cp_min_global']:.6e}`",
        f"- Global cn min: `{physical['cn_min_global']:.6e}`",
        f"- Max |phi|: `{physical['max_abs_phi_global']:.6e}`",
        f"- Max cp mass drift: `{physical['cp_mass_relative_drift_max']:.6e}`",
        f"- Max cn mass drift: `{physical['cn_mass_relative_drift_max']:.6e}`",
        "",
        "## Files",
        "",
        "- `phase_a_summary.json`: full numeric diagnostics",
        "- `phase_a_report.md`: this report",
    ]
    return "\n".join(lines) + "\n"


def run_phase_a_diagnostics(
    run_dir: str | Path,
    *,
    checkpoint_name: str = "model_final.eqx",
    seed: int = 123,
    n_domain: int = 2048,
    n_boundary_layer: int = 1024,
    n_boundary: int = 1024,
    n_initial: int = 2048,
    physical_nx: int = 128,
    physical_nt: int = 41,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    pnp_cfg, _ = load_run_configs(run_dir)
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = load_model(checkpoint_path, pnp_cfg)

    metadata_path = run_dir / "run_metadata.json"
    metadata_status = "unknown"
    if metadata_path.exists():
        import json

        with metadata_path.open() as f:
            metadata_status = json.load(f).get("status", "unknown")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "metadata_status": metadata_status,
        "training": summarize_training_history(run_dir),
        "held_out_residuals": held_out_residual_diagnostics(
            model,
            pnp_cfg,
            seed=seed,
            n_domain=n_domain,
            n_boundary_layer=n_boundary_layer,
            n_boundary=n_boundary,
        ),
        "boundary_conditions": boundary_condition_diagnostics(
            model,
            pnp_cfg,
            seed=seed + 1,
            n_per_side=n_boundary,
        ),
        "initial_condition": initial_condition_diagnostics(
            model,
            pnp_cfg,
            seed=seed + 2,
            n_points=n_initial,
        ),
        "physical_sanity": physical_sanity_diagnostics(
            model,
            pnp_cfg,
            nx=physical_nx,
            nt=physical_nt,
        ),
    }

    save_json(summary, diagnostics_dir / "phase_a_summary.json")
    report = make_phase_a_report(summary)
    (diagnostics_dir / "phase_a_report.md").write_text(report)
    return summary
