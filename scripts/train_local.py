"""Start a fresh local training run."""

from __future__ import annotations

import argparse
from dataclasses import replace

from _path_setup import PROJECT_ROOT
from pnp_pinn import (
    high_quality_configs,
    high_quality_hard_phi_configs,
    local_good_configs,
    local_good_hard_phi_configs,
    make_run_dir,
    print_saved_artifacts,
    train,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["good", "high"], default="high")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--hard-phi",
        action="store_true",
        help="Use the hard-constrained phi BC trial.",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "checkpoints" / "runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hard_phi:
        label = args.label or (
            "pnp_1d_hard_phi_high_quality" if args.preset == "high" else "pnp_1d_hard_phi_good"
        )
        config_factory = high_quality_hard_phi_configs
        if args.preset == "good":
            config_factory = local_good_hard_phi_configs
    else:
        label = args.label or "pnp_1d_high_quality"
        config_factory = high_quality_configs if args.preset == "high" else local_good_configs

    run_dir = make_run_dir(label, args.runs_dir)
    pnp_cfg, train_cfg = config_factory(run_dir)
    if args.max_steps is not None:
        train_cfg = replace(train_cfg, max_steps=args.max_steps)

    print("Run directory:", run_dir)
    print(pnp_cfg)
    print(train_cfg)
    train(pnp_cfg, train_cfg, return_history=True)
    print_saved_artifacts(run_dir)


if __name__ == "__main__":
    main()
