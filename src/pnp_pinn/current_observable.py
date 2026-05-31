"""Current-observation utilities for the 1D PNP inverse problem."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .config import PNPConfig
from .inference import summarize_posterior_grid
from .model import PNP
from .reference_solver import ReferenceSolution


@dataclass(frozen=True)
class CurrentObservations:
    """Noisy interval-averaged charging-current observations.

    The sign convention is fixed by the selected electrode side. For the left
    electrode, the dimensionless electrode charge is computed from the Gauss-law
    identity

        Q_L(t) = -epsilon * (V_R - V_L) / L
                 - integral ((x_R - x) / L) rho(x, t) dx.

    This is equivalent to -epsilon * dphi/dx at the left electrode for the exact
    PNP solution, but it is more stable for PINN evaluation because it avoids
    differentiating the learned potential at the wall. The observed current is
    the interval average dQ_L/dt over each sampled time interval. This represents
    the externally measured charging current in the blocking-electrode setup,
    not the boundary ionic flux.
    """

    t_start: np.ndarray
    t_end: np.ndarray
    t_mid: np.ndarray
    clean_current: np.ndarray
    observed_current: np.ndarray
    sigma: np.ndarray
    clean_charge_start: np.ndarray
    clean_charge_end: np.ndarray
    true_dp: float
    true_dn: float
    side: str
    charge_quadrature_nx: int
    seed: int

    @property
    def n(self) -> int:
        return int(self.observed_current.size)


def _validate_side(side: str) -> str:
    side = side.lower().strip()
    if side not in {"left", "right"}:
        raise ValueError("side must be either 'left' or 'right'")
    return side


def reference_electrode_charge(
    solution: ReferenceSolution,
    cfg: PNPConfig,
    *,
    side: str = "left",
) -> np.ndarray:
    """Compute electrode charge from the reference ionic charge density."""

    side = _validate_side(side)
    length = cfg.x_max - cfg.x_min
    charge_density = cfg.zp * solution.cp + cfg.zn * solution.cn
    if side == "left":
        weight = (cfg.x_max - solution.x) / length
        wall_charge = -cfg.epsilon * (cfg.v_right - cfg.v_left) / length
    else:
        weight = (solution.x - cfg.x_min) / length
        wall_charge = cfg.epsilon * (cfg.v_right - cfg.v_left) / length

    induced_charge = np.trapezoid(weight[None, :] * charge_density, solution.x, axis=1)
    return wall_charge - induced_charge


def interval_current_from_charge(
    t: Sequence[float],
    charge: Sequence[float],
    *,
    skip_initial_intervals: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return interval-averaged current and matching interval metadata."""

    t = np.asarray(t, dtype=np.float64)
    charge = np.asarray(charge, dtype=np.float64)
    if t.ndim != 1 or charge.ndim != 1:
        raise ValueError("t and charge must be one-dimensional")
    if t.size != charge.size:
        raise ValueError("t and charge must have the same length")
    if t.size < 2:
        raise ValueError("at least two time points are required")
    if skip_initial_intervals < 0:
        raise ValueError("skip_initial_intervals must be nonnegative")
    if skip_initial_intervals >= t.size - 1:
        raise ValueError("skip_initial_intervals skips all available intervals")

    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("time points must be strictly increasing")

    current = np.diff(charge) / dt
    start = skip_initial_intervals
    return (
        t[:-1][start:],
        t[1:][start:],
        0.5 * (t[:-1][start:] + t[1:][start:]),
        charge[:-1][start:],
        charge[1:][start:],
        current[start:],
    )


def make_current_observations(
    solution: ReferenceSolution,
    cfg: PNPConfig,
    *,
    side: str = "left",
    sigma: float,
    seed: int,
    skip_initial_intervals: int = 0,
) -> CurrentObservations:
    """Generate noisy charging-current observations from a reference solution."""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    side = _validate_side(side)
    charge = reference_electrode_charge(solution, cfg, side=side)
    t_start, t_end, t_mid, q_start, q_end, clean_current = interval_current_from_charge(
        solution.t,
        charge,
        skip_initial_intervals=skip_initial_intervals,
    )

    rng = np.random.default_rng(seed)
    observed = clean_current + rng.normal(loc=0.0, scale=sigma, size=clean_current.shape)
    return CurrentObservations(
        t_start=t_start,
        t_end=t_end,
        t_mid=t_mid,
        clean_current=clean_current,
        observed_current=observed,
        sigma=np.full_like(clean_current, float(sigma), dtype=np.float64),
        clean_charge_start=q_start,
        clean_charge_end=q_end,
        true_dp=float(solution.dp),
        true_dn=float(solution.dn),
        side=side,
        charge_quadrature_nx=int(solution.nx),
        seed=int(seed),
    )


def save_current_observations_csv(observations: CurrentObservations, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "t_start",
                "t_end",
                "t_mid",
                "clean_charge_start",
                "clean_charge_end",
                "clean_current",
                "observed_current",
                "sigma",
                "side",
            ]
        )
        for i in range(observations.n):
            writer.writerow(
                [
                    i,
                    f"{observations.t_start[i]:.17g}",
                    f"{observations.t_end[i]:.17g}",
                    f"{observations.t_mid[i]:.17g}",
                    f"{observations.clean_charge_start[i]:.17g}",
                    f"{observations.clean_charge_end[i]:.17g}",
                    f"{observations.clean_current[i]:.17g}",
                    f"{observations.observed_current[i]:.17g}",
                    f"{observations.sigma[i]:.17g}",
                    observations.side,
                ]
            )


def _charge_at_times(
    model: PNP,
    cfg: PNPConfig,
    theta_pair: jnp.ndarray,
    t: jnp.ndarray,
    *,
    side: str,
    quadrature_nx: int,
) -> jnp.ndarray:
    side = _validate_side(side)
    if quadrature_nx < 3:
        raise ValueError("quadrature_nx must be at least 3")
    x_grid = jnp.linspace(cfg.x_min, cfg.x_max, quadrature_nx)
    length = cfg.x_max - cfg.x_min
    if side == "left":
        weight = (cfg.x_max - x_grid) / length
        wall_charge = -cfg.epsilon * (cfg.v_right - cfg.v_left) / length
    else:
        weight = (x_grid - cfg.x_min) / length
        wall_charge = cfg.epsilon * (cfg.v_right - cfg.v_left) / length

    def charge_one(tt):
        values = jax.vmap(lambda xx: model(xx, tt, theta_pair))(x_grid)
        charge_density = cfg.zp * values[:, 0] + cfg.zn * values[:, 1]
        induced_charge = jnp.trapezoid(weight * charge_density, x_grid)
        return wall_charge - induced_charge

    return jax.vmap(charge_one)(t)


def _make_current_batch_predictor(
    model: PNP,
    cfg: PNPConfig,
    observations: CurrentObservations,
    *,
    quadrature_nx: int,
):
    all_times_np, inverse = np.unique(
        np.concatenate([observations.t_start, observations.t_end]),
        return_inverse=True,
    )
    n_start = observations.t_start.size
    start_idx = jnp.asarray(inverse[:n_start])
    end_idx = jnp.asarray(inverse[n_start:])
    all_times = jnp.asarray(all_times_np)
    dt = jnp.asarray(observations.t_end - observations.t_start)
    side = observations.side

    def predict_one(theta_pair: jnp.ndarray) -> jnp.ndarray:
        charge = _charge_at_times(
            model,
            cfg,
            theta_pair,
            all_times,
            side=side,
            quadrature_nx=quadrature_nx,
        )
        q_start = charge[start_idx]
        q_end = charge[end_idx]
        return (q_end - q_start) / dt

    return jax.jit(jax.vmap(predict_one))


def predict_current_observations(
    model: PNP,
    cfg: PNPConfig,
    observations: CurrentObservations,
    dp: float,
    dn: float,
    *,
    quadrature_nx: int = 101,
) -> np.ndarray:
    predictor = _make_current_batch_predictor(
        model,
        cfg,
        observations,
        quadrature_nx=quadrature_nx,
    )
    theta = jnp.asarray([[dp, dn]], dtype=jnp.float32)
    pred = predictor(theta)[0]
    pred.block_until_ready()
    return np.asarray(pred, dtype=np.float64)


def evaluate_current_posterior_grid(
    model: PNP,
    cfg: PNPConfig,
    observations: CurrentObservations,
    *,
    dp_values: Sequence[float],
    dn_values: Sequence[float],
    chunk_size: int = 512,
    quadrature_nx: int = 101,
) -> dict[str, np.ndarray | float]:
    """Evaluate a uniform-prior Gaussian posterior from current observations."""

    dp_values = np.asarray(dp_values, dtype=np.float64)
    dn_values = np.asarray(dn_values, dtype=np.float64)
    if dp_values.ndim != 1 or dn_values.ndim != 1:
        raise ValueError("dp_values and dn_values must be one-dimensional")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    dp_mesh, dn_mesh = np.meshgrid(dp_values, dn_values, indexing="ij")
    theta_pairs = np.column_stack([dp_mesh.ravel(), dn_mesh.ravel()]).astype(np.float32)

    predictor = _make_current_batch_predictor(
        model,
        cfg,
        observations,
        quadrature_nx=quadrature_nx,
    )
    observed = np.asarray(observations.observed_current, dtype=np.float64)
    sigma = np.asarray(observations.sigma, dtype=np.float64)
    log_norm_const = np.log(sigma) + 0.5 * np.log(2.0 * np.pi)

    log_likelihood_chunks = []
    sse_chunks = []
    for start in range(0, theta_pairs.shape[0], chunk_size):
        chunk = jnp.asarray(theta_pairs[start : start + chunk_size])
        pred = np.asarray(predictor(chunk), dtype=np.float64)
        residual = (pred - observed[None, :]) / sigma[None, :]
        sse = np.sum(residual**2, axis=1)
        log_likelihood = -0.5 * sse - float(np.sum(log_norm_const))
        sse_chunks.append(sse)
        log_likelihood_chunks.append(log_likelihood)

    log_likelihood_flat = np.concatenate(log_likelihood_chunks)
    sse_flat = np.concatenate(sse_chunks)
    max_log = float(np.max(log_likelihood_flat))
    posterior_mass_flat = np.exp(log_likelihood_flat - max_log)
    posterior_mass_flat /= float(np.sum(posterior_mass_flat))

    shape = (dp_values.size, dn_values.size)
    return {
        "dp_values": dp_values,
        "dn_values": dn_values,
        "log_likelihood": log_likelihood_flat.reshape(shape),
        "sum_squared_standardized_error": sse_flat.reshape(shape),
        "posterior_mass": posterior_mass_flat.reshape(shape),
        "posterior_sum": float(np.sum(posterior_mass_flat)),
    }


def summarize_current_posterior_grid(
    grid: dict[str, np.ndarray | float],
    observations: CurrentObservations,
    *,
    credible_mass: float = 0.95,
) -> dict[str, object]:
    return summarize_posterior_grid(
        grid,
        true_dp=observations.true_dp,
        true_dn=observations.true_dn,
        credible_mass=credible_mass,
    )
