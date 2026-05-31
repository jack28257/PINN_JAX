"""Loss construction and NTK-style group balancing."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .config import PNPConfig
from .model import PNP
from .types import Array, ConstraintBatch, GroupWeights, ResidualWeights, TrainingBatch


def tree_l2_norm(tree) -> Array:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    if not leaves:
        return jnp.array(0.0)
    return sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)


def make_residual_weights(cfg: PNPConfig) -> ResidualWeights:
    bc_weights = jnp.asarray(cfg.bc_resid_weights)
    if cfg.hard_phi_bc:
        bc_weights = bc_weights.at[2].set(0.0)
    return ResidualWeights(
        dom=jnp.asarray(cfg.dom_resid_weights),
        bl=jnp.asarray(cfg.bl_resid_weights),
        bc=bc_weights,
    )


def residual_mse(
    model: PNP,
    batch: ConstraintBatch,
    resid_name: str,
    component_weights: Array,
) -> Array:
    resid_fn = getattr(model, resid_name)
    residuals = jax.vmap(resid_fn)(batch.x, batch.t, batch.theta)
    return jnp.mean(jnp.sum(component_weights * jnp.square(residuals), axis=-1))


def loss_terms(model: PNP, batch: TrainingBatch, weights: ResidualWeights) -> dict[str, Array]:
    return {
        "dom": residual_mse(model, batch.dom, "dom_resid", weights.dom),
        "bl": residual_mse(model, batch.bl, "dom_resid", weights.bl),
        "bc": residual_mse(model, batch.bc, "bc_resid", weights.bc),
    }


def total_loss(
    model: PNP,
    batch: TrainingBatch,
    group_weights: GroupWeights,
    residual_weights: ResidualWeights,
) -> tuple[Array, dict[str, Array]]:
    terms = loss_terms(model, batch, residual_weights)
    loss = (
        group_weights.dom * terms["dom"]
        + group_weights.bl * terms["bl"]
        + group_weights.bc * terms["bc"]
    )
    return loss, terms


def compute_ntk_weights(
    model: PNP,
    batch: TrainingBatch,
    residual_weights: ResidualWeights,
    eps: float = 1e-12,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> GroupWeights:
    """Gradient-trace approximation to NTK-style loss balancing.

    The clamp is important. Without it, a group with a large gradient trace can
    receive a near-zero outer weight, making the printed total loss look tiny
    while a physical constraint, often the boundary condition, remains bad.
    """

    dom_grads = eqx.filter_grad(
        lambda m: residual_mse(m, batch.dom, "dom_resid", residual_weights.dom)
    )(model)
    bl_grads = eqx.filter_grad(
        lambda m: residual_mse(m, batch.bl, "dom_resid", residual_weights.bl)
    )(model)
    bc_grads = eqx.filter_grad(
        lambda m: residual_mse(m, batch.bc, "bc_resid", residual_weights.bc)
    )(model)
    traces = jnp.array(
        [
            tree_l2_norm(dom_grads),
            tree_l2_norm(bl_grads),
            tree_l2_norm(bc_grads),
        ]
    )
    active = traces > eps
    n_active = jnp.maximum(jnp.sum(active), 1)
    active_mean = jnp.sum(jnp.where(active, traces, 0.0)) / n_active

    raw = jnp.where(active, active_mean / (traces + eps), 1.0)
    raw = jnp.where(active, jnp.clip(raw, min_weight, max_weight), raw)
    active_raw_mean = jnp.sum(jnp.where(active, raw, 0.0)) / n_active
    raw = jnp.where(active, raw / (active_raw_mean + eps), raw)
    raw = jnp.where(active, jnp.clip(raw, min_weight, max_weight), raw)
    return GroupWeights(dom=raw[0], bl=raw[1], bc=raw[2])
