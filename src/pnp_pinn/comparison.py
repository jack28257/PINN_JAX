"""PINN-vs-reference comparison utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .model import PNP
from .reference_solver import ReferenceSolution, max_abs_error, relative_l2


def evaluate_pinn_on_reference_grid(model: PNP, solution: ReferenceSolution) -> dict[str, np.ndarray]:
    theta = {"Dp": jnp.array(solution.dp), "Dn": jnp.array(solution.dn)}
    x_grid = jnp.asarray(solution.x)
    t_grid = jnp.asarray(solution.t)

    def eval_at_t(tt):
        return jax.vmap(lambda xx: model(xx, tt, theta))(x_grid)

    values = jax.vmap(eval_at_t)(t_grid)
    values.block_until_ready()
    values_np = np.asarray(values)
    return {
        "cp": values_np[:, :, 0],
        "cn": values_np[:, :, 1],
        "phi": values_np[:, :, 2],
    }


def compare_pinn_to_reference(model: PNP, solution: ReferenceSolution) -> dict[str, float]:
    pred = evaluate_pinn_on_reference_grid(model, solution)
    return {
        "cp_relative_l2": relative_l2(pred["cp"], solution.cp),
        "cn_relative_l2": relative_l2(pred["cn"], solution.cn),
        "phi_relative_l2": relative_l2(pred["phi"], solution.phi),
        "cp_max_abs": max_abs_error(pred["cp"], solution.cp),
        "cn_max_abs": max_abs_error(pred["cn"], solution.cn),
        "phi_max_abs": max_abs_error(pred["phi"], solution.phi),
        "cp_min_pinn": float(np.min(pred["cp"])),
        "cn_min_pinn": float(np.min(pred["cn"])),
        "cp_min_reference": float(np.min(solution.cp)),
        "cn_min_reference": float(np.min(solution.cn)),
    }
