"""Benchmark compile time and steady-state step time for the local JAX PINN."""

from __future__ import annotations

import argparse
import time

from _path_setup import PROJECT_ROOT

import equinox as eqx
import jax
import jax.numpy as jnp

from pnp_pinn.config import PNPConfig, TrainConfig
from pnp_pinn.losses import make_residual_weights
from pnp_pinn.model import PNP
from pnp_pinn.sampling import sample_batch
from pnp_pinn.types import GroupWeights
from pnp_pinn.train import make_optimizer, make_train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timed-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pnp_cfg = PNPConfig(width_size=args.width, depth=args.depth)
    train_cfg = TrainConfig(
        max_steps=args.timed_steps,
        dom_batch_size=args.batch_size,
        bl_batch_size=args.batch_size,
        bc_batch_size=max(args.batch_size // 2, 1),
        use_ntk_weights=False,
        checkpoint_path=None,
    )
    key = jax.random.PRNGKey(train_cfg.seed)
    key, model_key, batch_key = jax.random.split(key, 3)
    model = PNP(pnp_cfg, model_key)
    optimizer = make_optimizer(train_cfg)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_train_step(optimizer)
    residual_weights = make_residual_weights(pnp_cfg)
    group_weights = GroupWeights(
        dom=jnp.array(1.0),
        bl=jnp.array(1.0),
        bc=jnp.array(1.0),
    )
    batch = sample_batch(batch_key, pnp_cfg, train_cfg)

    compile_start = time.perf_counter()
    model, opt_state, loss, _ = train_step(model, opt_state, batch, group_weights, residual_weights)
    loss.block_until_ready()
    compile_seconds = time.perf_counter() - compile_start

    timed_start = time.perf_counter()
    for _ in range(args.timed_steps):
        key, batch_key = jax.random.split(key)
        batch = sample_batch(batch_key, pnp_cfg, train_cfg)
        model, opt_state, loss, _ = train_step(model, opt_state, batch, group_weights, residual_weights)
        loss.block_until_ready()
    timed_seconds = time.perf_counter() - timed_start

    print("Project:", PROJECT_ROOT)
    print("JAX backend:", jax.default_backend())
    print(f"Compile plus first step: {compile_seconds:.3f} s")
    print(f"Mean steady step: {timed_seconds / args.timed_steps:.3f} s")


if __name__ == "__main__":
    main()
