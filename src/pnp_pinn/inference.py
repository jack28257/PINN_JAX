"""Synthetic inverse-problem utilities for the 1D PNP PINN."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .model import PNP
from .reference_solver import ReferenceSolution


FIELD_INDEX = {"cp": 0, "cn": 1, "phi": 2}


@dataclass(frozen=True)
class ProbeObservations:
    """Noisy observations sampled from a reference PNP solution."""

    x: np.ndarray
    t: np.ndarray
    field: np.ndarray
    field_index: np.ndarray
    clean_value: np.ndarray
    observed_value: np.ndarray
    sigma: np.ndarray
    true_dp: float
    true_dn: float
    seed: int

    @property
    def n(self) -> int:
        return int(self.observed_value.size)


def parse_float_list(values: str | Iterable[float]) -> list[float]:
    if isinstance(values, str):
        return [float(item.strip()) for item in values.split(",") if item.strip()]
    return [float(value) for value in values]


def parse_field_list(values: str | Iterable[str]) -> list[str]:
    if isinstance(values, str):
        fields = [item.strip() for item in values.split(",") if item.strip()]
    else:
        fields = [str(value).strip() for value in values]
    unknown = sorted(set(fields) - set(FIELD_INDEX))
    if unknown:
        raise ValueError(f"Unknown observation field(s): {unknown}. Use any of {sorted(FIELD_INDEX)}.")
    return fields


def reference_case_name(dp: float, dn: float) -> str:
    return f"Dp{dp:g}_Dn{dn:g}".replace(".", "p")


def load_reference_npz(path: str | Path) -> ReferenceSolution:
    data = np.load(path, allow_pickle=False)
    return ReferenceSolution(
        x=np.asarray(data["x"], dtype=np.float64),
        t=np.asarray(data["t"], dtype=np.float64),
        cp=np.asarray(data["cp"], dtype=np.float64),
        cn=np.asarray(data["cn"], dtype=np.float64),
        phi=np.asarray(data["phi"], dtype=np.float64),
        dp=float(data["dp"]),
        dn=float(data["dn"]),
        nx=int(data["nx"]),
        method=str(data["method"]),
        rtol=float(data["rtol"]),
        atol=float(data["atol"]),
        success=bool(data["success"]),
        message=str(data["message"]),
        nfev=int(data["nfev"]),
    )


def interpolate_reference_value(solution: ReferenceSolution, field: str, x: float, t: float) -> float:
    """Bilinearly interpolate one field from a reference solution."""

    if field not in FIELD_INDEX:
        raise ValueError(f"Unknown field {field!r}")
    values = np.asarray(getattr(solution, field), dtype=np.float64)
    values_at_x = np.array([np.interp(x, solution.x, row) for row in values], dtype=np.float64)
    return float(np.interp(t, solution.t, values_at_x))


def make_probe_observations(
    solution: ReferenceSolution,
    *,
    x_points: Sequence[float],
    t_points: Sequence[float],
    fields: Sequence[str],
    sigma: float,
    seed: int,
) -> ProbeObservations:
    """Sample noisy probe-field observations from a reference solution."""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    fields = parse_field_list(fields)
    rng = np.random.default_rng(seed)

    rows: list[tuple[float, float, str, int, float, float, float]] = []
    for tt in t_points:
        for xx in x_points:
            for field in fields:
                clean = interpolate_reference_value(solution, field, float(xx), float(tt))
                noisy = clean + float(rng.normal(loc=0.0, scale=sigma))
                rows.append((float(xx), float(tt), field, FIELD_INDEX[field], clean, noisy, float(sigma)))

    return ProbeObservations(
        x=np.array([row[0] for row in rows], dtype=np.float64),
        t=np.array([row[1] for row in rows], dtype=np.float64),
        field=np.array([row[2] for row in rows], dtype=object),
        field_index=np.array([row[3] for row in rows], dtype=np.int32),
        clean_value=np.array([row[4] for row in rows], dtype=np.float64),
        observed_value=np.array([row[5] for row in rows], dtype=np.float64),
        sigma=np.array([row[6] for row in rows], dtype=np.float64),
        true_dp=float(solution.dp),
        true_dn=float(solution.dn),
        seed=int(seed),
    )


def save_observations_csv(observations: ProbeObservations, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "t", "x", "field", "clean_value", "observed_value", "sigma"])
        for i in range(observations.n):
            writer.writerow(
                [
                    i,
                    f"{observations.t[i]:.17g}",
                    f"{observations.x[i]:.17g}",
                    str(observations.field[i]),
                    f"{observations.clean_value[i]:.17g}",
                    f"{observations.observed_value[i]:.17g}",
                    f"{observations.sigma[i]:.17g}",
                ]
            )


def _make_batch_predictor(model: PNP, observations: ProbeObservations):
    x = jnp.asarray(observations.x)
    t = jnp.asarray(observations.t)
    field_index = jnp.asarray(observations.field_index)

    def predict_one(theta_pair: jnp.ndarray) -> jnp.ndarray:
        def eval_point(xx, tt):
            return model(xx, tt, theta_pair)

        values = jax.vmap(eval_point)(x, t)
        return values[jnp.arange(field_index.size), field_index]

    return jax.jit(jax.vmap(predict_one))


def predict_observations(
    model: PNP,
    observations: ProbeObservations,
    dp: float,
    dn: float,
) -> np.ndarray:
    predictor = _make_batch_predictor(model, observations)
    theta = jnp.asarray([[dp, dn]], dtype=jnp.float32)
    pred = predictor(theta)[0]
    pred.block_until_ready()
    return np.asarray(pred, dtype=np.float64)


def evaluate_posterior_grid(
    model: PNP,
    observations: ProbeObservations,
    *,
    dp_values: Sequence[float],
    dn_values: Sequence[float],
    chunk_size: int = 512,
) -> dict[str, np.ndarray | float]:
    """Evaluate a uniform-prior Gaussian posterior on a rectangular grid."""

    dp_values = np.asarray(dp_values, dtype=np.float64)
    dn_values = np.asarray(dn_values, dtype=np.float64)
    if dp_values.ndim != 1 or dn_values.ndim != 1:
        raise ValueError("dp_values and dn_values must be one-dimensional")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    dp_mesh, dn_mesh = np.meshgrid(dp_values, dn_values, indexing="ij")
    theta_pairs = np.column_stack([dp_mesh.ravel(), dn_mesh.ravel()]).astype(np.float32)

    predictor = _make_batch_predictor(model, observations)
    observed = np.asarray(observations.observed_value, dtype=np.float64)
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


def _weighted_quantile_grid(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return float(values[min(np.searchsorted(cdf, q, side="left"), values.size - 1)])


def summarize_posterior_grid(
    grid: dict[str, np.ndarray | float],
    *,
    true_dp: float,
    true_dn: float,
    credible_mass: float = 0.95,
) -> dict[str, object]:
    dp_values = np.asarray(grid["dp_values"], dtype=np.float64)
    dn_values = np.asarray(grid["dn_values"], dtype=np.float64)
    posterior_mass = np.asarray(grid["posterior_mass"], dtype=np.float64)
    dp_marginal = posterior_mass.sum(axis=1)
    dn_marginal = posterior_mass.sum(axis=0)

    map_i, map_j = np.unravel_index(int(np.argmax(posterior_mass)), posterior_mass.shape)
    dp_mean = float(np.sum(dp_values * dp_marginal))
    dn_mean = float(np.sum(dn_values * dn_marginal))
    dp_sd = float(np.sqrt(np.sum((dp_values - dp_mean) ** 2 * dp_marginal)))
    dn_sd = float(np.sqrt(np.sum((dn_values - dn_mean) ** 2 * dn_marginal)))

    alpha = (1.0 - credible_mass) / 2.0
    dp_ci = (
        _weighted_quantile_grid(dp_values, dp_marginal, alpha),
        _weighted_quantile_grid(dp_values, dp_marginal, 1.0 - alpha),
    )
    dn_ci = (
        _weighted_quantile_grid(dn_values, dn_marginal, alpha),
        _weighted_quantile_grid(dn_values, dn_marginal, 1.0 - alpha),
    )
    covered = bool(dp_ci[0] <= true_dp <= dp_ci[1] and dn_ci[0] <= true_dn <= dn_ci[1])

    return {
        "true": {"Dp": float(true_dp), "Dn": float(true_dn)},
        "map": {"Dp": float(dp_values[map_i]), "Dn": float(dn_values[map_j])},
        "mean": {"Dp": dp_mean, "Dn": dn_mean},
        "sd": {"Dp": dp_sd, "Dn": dn_sd},
        "credible_mass": float(credible_mass),
        "credible_interval": {
            "Dp": {"lower": dp_ci[0], "upper": dp_ci[1]},
            "Dn": {"lower": dn_ci[0], "upper": dn_ci[1]},
        },
        "true_value_inside_joint_marginal_intervals": covered,
        "posterior_sum": float(grid["posterior_sum"]),
        "max_posterior_mass": float(np.max(posterior_mass)),
    }


def save_posterior_grid_csv(grid: dict[str, np.ndarray | float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dp_values = np.asarray(grid["dp_values"], dtype=np.float64)
    dn_values = np.asarray(grid["dn_values"], dtype=np.float64)
    log_likelihood = np.asarray(grid["log_likelihood"], dtype=np.float64)
    posterior_mass = np.asarray(grid["posterior_mass"], dtype=np.float64)
    sse = np.asarray(grid["sum_squared_standardized_error"], dtype=np.float64)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Dp", "Dn", "log_likelihood", "posterior_mass", "sum_squared_standardized_error"]
        )
        for i, dp in enumerate(dp_values):
            for j, dn in enumerate(dn_values):
                writer.writerow(
                    [
                        f"{dp:.17g}",
                        f"{dn:.17g}",
                        f"{log_likelihood[i, j]:.17g}",
                        f"{posterior_mass[i, j]:.17g}",
                        f"{sse[i, j]:.17g}",
                    ]
                )
