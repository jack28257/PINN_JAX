"""Resume a saved local run from its full training state."""

from __future__ import annotations

import argparse

from _path_setup import PROJECT_ROOT
from pnp_pinn import resume_saved_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory. If omitted, uses the newest run under checkpoints/runs/.",
    )
    parser.add_argument("--state-name", default="training_state_latest.eqx")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Final global step to train to, not additional steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if run_dir is not None and not run_dir.startswith("/"):
        run_dir = str(PROJECT_ROOT / run_dir)
    resume_saved_run(run_dir, state_name=args.state_name, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
