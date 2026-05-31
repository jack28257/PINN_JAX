"""Run Phase A PINN-only diagnostics for a completed training run."""

from __future__ import annotations

import argparse

from _path_setup import PROJECT_ROOT
from pnp_pinn.diagnostics import run_phase_a_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory, e.g. checkpoints/runs/pnp_1d_hard_phi_high_quality_YYYYMMDD_HHMMSS",
    )
    parser.add_argument("--checkpoint-name", default="model_final.eqx")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-domain", type=int, default=2048)
    parser.add_argument("--n-boundary-layer", type=int, default=1024)
    parser.add_argument("--n-boundary", type=int, default=1024)
    parser.add_argument("--n-initial", type=int, default=2048)
    parser.add_argument("--physical-nx", type=int, default=128)
    parser.add_argument("--physical-nt", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir.startswith("/"):
        run_dir = str(PROJECT_ROOT / run_dir)

    summary = run_phase_a_diagnostics(
        run_dir,
        checkpoint_name=args.checkpoint_name,
        seed=args.seed,
        n_domain=args.n_domain,
        n_boundary_layer=args.n_boundary_layer,
        n_boundary=args.n_boundary,
        n_initial=args.n_initial,
        physical_nx=args.physical_nx,
        physical_nt=args.physical_nt,
    )
    diagnostics_dir = PROJECT_ROOT / args.run_dir / "diagnostics" if not args.run_dir.startswith("/") else None
    print("Phase A diagnostics complete.")
    print("Run:", summary["run_dir"])
    if diagnostics_dir is not None:
        print("Diagnostics:", diagnostics_dir)
    else:
        print("Diagnostics:", f"{summary['run_dir']}/diagnostics")
    print("Final loss:", summary["training"].get("final_loss"))
    print("Domain held-out RMS:", summary["held_out_residuals"]["domain_total_rms"])
    print("Boundary held-out RMS:", summary["held_out_residuals"]["boundary_total_rms"])
    print("Boundary phi max abs:", summary["boundary_conditions"]["combined"]["phi_error"]["max_abs"])
    print("cp min:", summary["physical_sanity"]["summary"]["cp_min_global"])
    print("cn min:", summary["physical_sanity"]["summary"]["cn_min_global"])


if __name__ == "__main__":
    main()
