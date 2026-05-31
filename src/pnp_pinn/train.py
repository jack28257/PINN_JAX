"""Training loop with exact local resume support."""

from __future__ import annotations

from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .checkpointing import (
    load_history,
    load_pytree,
    load_run_configs,
    print_saved_artifacts,
    save_history,
    save_model,
    save_pytree,
    save_run_metadata,
)
from .config import PNPConfig, TrainConfig
from .losses import compute_ntk_weights, make_residual_weights, total_loss
from .model import PNP
from .sampling import sample_batch
from .types import Array, GroupWeights, TrainingBatch, TrainingState


def format_duration(seconds: float) -> str:
    """Format seconds as a compact human-readable duration."""

    if seconds != seconds or seconds == float("inf"):
        return "unknown"
    seconds = int(max(seconds, 0))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def make_optimizer(train_cfg: TrainConfig) -> optax.GradientTransformation:
    schedule = optax.exponential_decay(
        init_value=train_cfg.learning_rate,
        transition_steps=train_cfg.lr_decay_steps,
        decay_rate=train_cfg.lr_decay_rate,
        staircase=False,
    )
    return optax.chain(
        optax.clip_by_global_norm(train_cfg.max_grad_norm),
        optax.adam(schedule),
    )


def initial_training_state(
    pnp_cfg: PNPConfig,
    train_cfg: TrainConfig,
    optimizer: optax.GradientTransformation,
    *,
    initial_model: PNP | None = None,
    step: int = 0,
) -> TrainingState:
    key = jax.random.PRNGKey(train_cfg.seed)
    key, model_key = jax.random.split(key)
    model = initial_model if initial_model is not None else PNP(pnp_cfg, model_key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    group_weights = GroupWeights(
        dom=jnp.array(1.0),
        bl=jnp.array(1.0),
        bc=jnp.array(1.0),
    )
    return TrainingState(
        model=model,
        opt_state=opt_state,
        key=key,
        group_weights=group_weights,
        step=jnp.array(step, dtype=jnp.int32),
    )


def make_training_state_template(pnp_cfg: PNPConfig, train_cfg: TrainConfig) -> TrainingState:
    optimizer = make_optimizer(train_cfg)
    return initial_training_state(pnp_cfg, train_cfg, optimizer)


def save_training_state(state: TrainingState, path: str | Path) -> None:
    save_pytree(state, path)


def load_training_state(
    path: str | Path,
    pnp_cfg: PNPConfig,
    train_cfg: TrainConfig,
) -> TrainingState:
    template = make_training_state_template(pnp_cfg, train_cfg)
    return load_pytree(path, template)


def make_train_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def train_step(
        model: PNP,
        opt_state: optax.OptState,
        batch: TrainingBatch,
        group_weights: GroupWeights,
        residual_weights,
    ) -> tuple[PNP, optax.OptState, Array, dict[str, Array]]:
        (loss, terms), grads = eqx.filter_value_and_grad(total_loss, has_aux=True)(
            model,
            batch,
            group_weights,
            residual_weights,
        )
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, terms

    return train_step


def train(
    pnp_cfg: PNPConfig,
    train_cfg: TrainConfig,
    *,
    return_history: bool = False,
    resume_state: TrainingState | None = None,
    history: list[dict[str, float]] | None = None,
    initial_model: PNP | None = None,
    step_offset: int = 0,
) -> PNP | tuple[PNP, list[dict[str, float]]]:
    optimizer = make_optimizer(train_cfg)
    train_step = make_train_step(optimizer)
    residual_weights = make_residual_weights(pnp_cfg)

    if resume_state is None:
        state = initial_training_state(
            pnp_cfg,
            train_cfg,
            optimizer,
            initial_model=initial_model,
            step=step_offset,
        )
    else:
        state = resume_state

    model = state.model
    opt_state = state.opt_state
    key = state.key
    group_weights = state.group_weights
    start_step = int(state.step)

    history = list(history or [])
    history = [row for row in history if int(row["step"]) <= start_step]

    run_dir = Path(train_cfg.checkpoint_path).parent if train_cfg.checkpoint_path else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_history(history, run_dir)
        save_run_metadata(pnp_cfg, train_cfg, run_dir, status="started", latest_step=start_step)
        save_training_state(state, run_dir / "training_state_latest.eqx")

    last_step = start_step
    interrupted = False
    wall_start = time.perf_counter()
    last_log_wall = wall_start
    last_log_step = start_step
    total_steps_this_call = max(train_cfg.max_steps - start_step, 0)

    try:
        for step in range(start_step + 1, train_cfg.max_steps + 1):
            last_step = step
            key, batch_key = jax.random.split(key)
            batch = sample_batch(batch_key, pnp_cfg, train_cfg)

            if train_cfg.use_ntk_weights and (
                step == start_step + 1 or step % train_cfg.ntk_update_every == 0
            ):
                group_weights = compute_ntk_weights(model, batch, residual_weights)

            model, opt_state, loss, terms = train_step(
                model,
                opt_state,
                batch,
                group_weights,
                residual_weights,
            )
            state = TrainingState(
                model=model,
                opt_state=opt_state,
                key=key,
                group_weights=group_weights,
                step=jnp.array(step, dtype=jnp.int32),
            )

            if (
                step == start_step + 1
                or step % train_cfg.print_every == 0
                or step == train_cfg.max_steps
            ):
                now = time.perf_counter()
                elapsed_seconds = now - wall_start
                completed_steps_this_call = step - start_step
                remaining_steps = max(train_cfg.max_steps - step, 0)
                avg_steps_per_second = (
                    completed_steps_this_call / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
                )
                recent_seconds = now - last_log_wall
                recent_steps = step - last_log_step
                current_steps_per_second = (
                    recent_steps / recent_seconds if recent_seconds > 0.0 else 0.0
                )
                projected_total_seconds = (
                    total_steps_this_call / avg_steps_per_second
                    if avg_steps_per_second > 0.0
                    else float("inf")
                )
                remaining_seconds = (
                    remaining_steps / avg_steps_per_second
                    if avg_steps_per_second > 0.0
                    else float("inf")
                )
                progress = (
                    100.0 * completed_steps_this_call / total_steps_this_call
                    if total_steps_this_call > 0
                    else 100.0
                )
                row = {
                    "step": float(step),
                    "loss": float(loss),
                    "dom": float(terms["dom"]),
                    "bl": float(terms["bl"]),
                    "bc": float(terms["bc"]),
                    "w_dom": float(group_weights.dom),
                    "w_bl": float(group_weights.bl),
                    "w_bc": float(group_weights.bc),
                    "elapsed_seconds": elapsed_seconds,
                    "current_steps_per_second": current_steps_per_second,
                    "average_steps_per_second": avg_steps_per_second,
                    "projected_total_seconds": projected_total_seconds,
                    "remaining_seconds": remaining_seconds,
                }
                history.append(row)
                print(
                    f"step={step:8d}/{train_cfg.max_steps:<8d} "
                    f"({progress:5.1f}%) "
                    f"loss={float(loss):.3e} "
                    f"dom={float(terms['dom']):.3e} "
                    f"bl={float(terms['bl']):.3e} "
                    f"bc={float(terms['bc']):.3e} "
                    f"weights=({float(group_weights.dom):.3f}, "
                    f"{float(group_weights.bl):.3f}, "
                    f"{float(group_weights.bc):.3f}) "
                    f"speed={current_steps_per_second:.3f} steps/s "
                    f"elapsed={format_duration(elapsed_seconds)} "
                    f"remaining={format_duration(remaining_seconds)} "
                    f"projected_total={format_duration(projected_total_seconds)}"
                )
                last_log_wall = now
                last_log_step = step

                if run_dir is not None:
                    save_history(history, run_dir)
                    save_run_metadata(
                        pnp_cfg,
                        train_cfg,
                        run_dir,
                        status="running",
                        latest_step=step,
                    )

            if (
                train_cfg.checkpoint_path is not None
                and train_cfg.checkpoint_every > 0
                and step % train_cfg.checkpoint_every == 0
            ):
                save_model(model, train_cfg.checkpoint_path)
                if run_dir is not None:
                    save_training_state(state, run_dir / "training_state_latest.eqx")
                    save_history(history, run_dir)
                    save_run_metadata(
                        pnp_cfg,
                        train_cfg,
                        run_dir,
                        status="checkpoint",
                        latest_step=step,
                    )

    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted; saving the latest full training state before returning.")

    finally:
        final_state = TrainingState(
            model=model,
            opt_state=opt_state,
            key=key,
            group_weights=group_weights,
            step=jnp.array(last_step, dtype=jnp.int32),
        )
        if train_cfg.checkpoint_path is not None:
            save_model(model, train_cfg.checkpoint_path)
        if run_dir is not None:
            final_model_name = "model_interrupted.eqx" if interrupted else "model_final.eqx"
            final_state_name = (
                "training_state_interrupted.eqx" if interrupted else "training_state_final.eqx"
            )
            save_model(model, run_dir / final_model_name)
            save_training_state(final_state, run_dir / "training_state_latest.eqx")
            save_training_state(final_state, run_dir / final_state_name)
            save_history(history, run_dir)
            save_run_metadata(
                pnp_cfg,
                train_cfg,
                run_dir,
                status="interrupted" if interrupted else "completed",
                latest_step=last_step,
            )

    if return_history:
        return model, history
    return model


def latest_run_dir(label: str | None = None, base_dir: str | Path = "checkpoints/runs") -> Path:
    base_dir = Path(base_dir)
    runs = list(base_dir.glob(f"{label}_*")) if label is not None else [
        path for path in base_dir.iterdir() if path.is_dir()
    ]
    if not runs:
        raise FileNotFoundError(f"No saved runs found under {base_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def load_resume_state(
    run_dir: str | Path | None = None,
    *,
    state_name: str = "training_state_latest.eqx",
) -> tuple[TrainingState, PNPConfig, TrainConfig, list[dict[str, float]], Path]:
    if run_dir is None:
        run_dir = latest_run_dir()
    run_dir = Path(run_dir)

    pnp_cfg, train_cfg = load_run_configs(run_dir)
    state_path = run_dir / state_name
    if not state_path.exists():
        raise FileNotFoundError(f"Training state not found: {state_path}")

    state = load_training_state(state_path, pnp_cfg, train_cfg)
    history = load_history(run_dir)
    print(f"Loaded training state at step {int(state.step)} from {state_path}")
    return state, pnp_cfg, train_cfg, history, run_dir


def resume_saved_run(
    run_dir: str | Path | None = None,
    *,
    state_name: str = "training_state_latest.eqx",
    max_steps: int | None = None,
) -> tuple[PNP, list[dict[str, float]], Path]:
    from dataclasses import replace

    state, pnp_cfg, train_cfg, history, run_dir = load_resume_state(
        run_dir,
        state_name=state_name,
    )
    if max_steps is not None:
        train_cfg = replace(train_cfg, max_steps=max_steps)
    model, history = train(
        pnp_cfg,
        train_cfg,
        return_history=True,
        resume_state=state,
        history=history,
    )
    print_saved_artifacts(run_dir)
    return model, history, run_dir
