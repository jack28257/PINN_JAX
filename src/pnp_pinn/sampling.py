"""Sampling utilities for domain, boundary-layer, and boundary batches."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import PNPConfig, TrainConfig
from .types import Array, ConstraintBatch, TrainingBatch


def sample_theta(key: Array, batch_size: int, cfg: PNPConfig) -> dict[str, Array]:
    key_dp, key_dn = jax.random.split(key)
    return {
        "Dp": jax.random.uniform(key_dp, (batch_size,), minval=cfg.dp_min, maxval=cfg.dp_max),
        "Dn": jax.random.uniform(key_dn, (batch_size,), minval=cfg.dn_min, maxval=cfg.dn_max),
    }


def sample_dom(key: Array, batch_size: int, cfg: PNPConfig) -> ConstraintBatch:
    key_x, key_t, key_theta = jax.random.split(key, 3)
    x = jax.random.uniform(key_x, (batch_size,), minval=cfg.x_min, maxval=cfg.x_max)
    t = jax.random.uniform(key_t, (batch_size,), minval=0.0, maxval=cfg.t_max)
    return ConstraintBatch(x=x, t=t, theta=sample_theta(key_theta, batch_size, cfg))


def sample_bl(key: Array, batch_size: int, cfg: PNPConfig) -> ConstraintBatch:
    key_side, key_x, key_t, key_theta = jax.random.split(key, 4)
    side = jax.random.bernoulli(key_side, shape=(batch_size,))
    dist_from_wall = jax.random.uniform(
        key_x,
        (batch_size,),
        minval=0.0,
        maxval=cfg.boundary_layer_width,
    )
    x = jnp.where(side, cfg.x_max - dist_from_wall, cfg.x_min + dist_from_wall)
    t = jax.random.uniform(key_t, (batch_size,), minval=0.0, maxval=cfg.t_max)
    return ConstraintBatch(x=x, t=t, theta=sample_theta(key_theta, batch_size, cfg))


def sample_bc(key: Array, batch_size: int, cfg: PNPConfig) -> ConstraintBatch:
    key_side, key_t, key_theta = jax.random.split(key, 3)
    side = jax.random.bernoulli(key_side, shape=(batch_size,))
    x = jnp.where(side, cfg.x_max, cfg.x_min)
    t = jax.random.uniform(key_t, (batch_size,), minval=0.0, maxval=cfg.t_max)
    return ConstraintBatch(x=x, t=t, theta=sample_theta(key_theta, batch_size, cfg))


def sample_batch(key: Array, cfg: PNPConfig, train_cfg: TrainConfig) -> TrainingBatch:
    key_dom, key_bl, key_bc = jax.random.split(key, 3)
    return TrainingBatch(
        dom=sample_dom(key_dom, train_cfg.dom_batch_size, cfg),
        bl=sample_bl(key_bl, train_cfg.bl_batch_size, cfg),
        bc=sample_bc(key_bc, train_cfg.bc_batch_size, cfg),
    )
