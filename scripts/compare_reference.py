"""Run Phase B diagnostics: finite-difference reference comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from _path_setup import PROJECT_ROOT
from pnp_pinn.checkpointing import load_model, load_run_configs, save_json
from pnp_pinn.comparison import compare_pinn_to_reference
from pnp_pinn.reference_solver import (
    compare_reference_solutions,
    parse_parameter_cases,
    save_reference_npz,
    solve_reference_pnp,
)


DEFAULT_CASES = "1.25,1.25;0.5,0.5;2.0,2.0;0.5,2.0;2.0,0.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory, e.g. checkpoints/runs/pnp_1d_hard_phi_high_quality_YYYYMMDD_HHMMSS",
    )
    parser.add_argument("--checkpoint-name", default="model_final.eqx")
    parser.add_argument(
        "--cases",
        default=DEFAULT_CASES,
        help="Semicolon-separated Dp,Dn pairs, e.g. '1.25,1.25;0.5,2.0'",
    )
    parser.add_argument(
        "--nx-values",
        default="101,201",
        help="Comma-separated reference grids. Last one is used for PINN comparison.",
    )
    parser.add_argument("--nt", type=int, default=41)
    parser.add_argument("--method", default="BDF", choices=["BDF", "Radau", "LSODA"])
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    return parser.parse_args()


def _case_name(dp: float, dn: float) -> str:
    return f"Dp{dp:g}_Dn{dn:g}".replace(".", "p")


def _make_report(summary: dict) -> str:
    lines = [
        "# Phase B Reference Comparison",
        "",
        f"Run directory: `{summary['run_dir']}`",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Reference method: `{summary['method']}`",
        f"Reference nx values: `{summary['nx_values']}`",
        f"Reference nt: `{summary['nt']}`",
        "",
        "## PINN vs Finest Reference",
        "",
        "| Dp | Dn | cp rel L2 | cn rel L2 | phi rel L2 | cp max abs | cn max abs | phi max abs |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        cmp = case["pinn_vs_reference"]
        lines.append(
            f"| {case['Dp']:.3g} | {case['Dn']:.3g} | "
            f"{cmp['cp_relative_l2']:.3e} | {cmp['cn_relative_l2']:.3e} | "
            f"{cmp['phi_relative_l2']:.3e} | {cmp['cp_max_abs']:.3e} | "
            f"{cmp['cn_max_abs']:.3e} | {cmp['phi_max_abs']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## Reference Grid Convergence",
            "",
            "These compare each coarser reference grid against the next finer grid, interpolated onto the coarser grid.",
            "",
            "| Dp | Dn | comparison | cp rel L2 | cn rel L2 | phi rel L2 |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for case in summary["cases"]:
        for conv in case["reference_grid_convergence"]:
            lines.append(
                f"| {case['Dp']:.3g} | {case['Dn']:.3g} | {conv['comparison']} | "
                f"{conv['cp_relative_l2']:.3e} | {conv['cn_relative_l2']:.3e} | "
                f"{conv['phi_relative_l2']:.3e} |"
            )

    lines.extend(
        [
            "",
            "## Solver Status",
            "",
            "| Dp | Dn | nx | success | nfev | seconds | message |",
            "|---:|---:|---:|:---:|---:|---:|---|",
        ]
    )
    for case in summary["cases"]:
        for solve in case["reference_solves"]:
            msg = str(solve["message"]).replace("|", "/")
            lines.append(
                f"| {case['Dp']:.3g} | {case['Dn']:.3g} | {solve['nx']} | "
                f"{solve['success']} | {solve['nfev']} | {solve['seconds']:.2f} | {msg} |"
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `phase_b_reference_summary.json`: full numeric comparison",
            "- `phase_b_reference_report.md`: this report",
            "- `reference_*.npz`: saved finite-difference reference solutions",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    diagnostics_dir = run_dir / "diagnostics"
    reference_dir = diagnostics_dir / "reference_solutions"
    reference_dir.mkdir(parents=True, exist_ok=True)

    pnp_cfg, _ = load_run_configs(run_dir)
    checkpoint = run_dir / args.checkpoint_name
    model = load_model(checkpoint, pnp_cfg)
    cases = parse_parameter_cases(args.cases)
    nx_values = [int(item.strip()) for item in args.nx_values.split(",") if item.strip()]
    if len(nx_values) < 1:
        raise ValueError("At least one nx value is required")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "method": args.method,
        "rtol": args.rtol,
        "atol": args.atol,
        "nt": args.nt,
        "nx_values": nx_values,
        "cases": [],
    }

    for dp, dn in cases:
        print(f"Solving reference case Dp={dp}, Dn={dn}")
        solutions = []
        solve_summaries = []
        for nx in nx_values:
            start = time.perf_counter()
            sol = solve_reference_pnp(
                pnp_cfg,
                dp=dp,
                dn=dn,
                nx=nx,
                nt=args.nt,
                method=args.method,
                rtol=args.rtol,
                atol=args.atol,
            )
            seconds = time.perf_counter() - start
            solutions.append(sol)
            solve_summaries.append(
                {
                    "nx": nx,
                    "success": sol.success,
                    "message": sol.message,
                    "nfev": sol.nfev,
                    "seconds": seconds,
                }
            )
            out_path = reference_dir / f"reference_{_case_name(dp, dn)}_nx{nx}.npz"
            save_reference_npz(out_path, sol)
            print(f"  nx={nx}: success={sol.success}, nfev={sol.nfev}, seconds={seconds:.2f}")

        convergence = []
        for coarse, fine in zip(solutions[:-1], solutions[1:]):
            conv = compare_reference_solutions(coarse, fine)
            conv["comparison"] = f"nx{coarse.nx}_vs_nx{fine.nx}"
            convergence.append(conv)

        finest = solutions[-1]
        pinn_cmp = compare_pinn_to_reference(model, finest)
        summary["cases"].append(
            {
                "Dp": dp,
                "Dn": dn,
                "reference_solves": solve_summaries,
                "reference_grid_convergence": convergence,
                "pinn_vs_reference": pinn_cmp,
            }
        )

    save_json(summary, diagnostics_dir / "phase_b_reference_summary.json")
    report = _make_report(summary)
    (diagnostics_dir / "phase_b_reference_report.md").write_text(report)
    print("Phase B reference comparison complete.")
    print("Report:", diagnostics_dir / "phase_b_reference_report.md")


if __name__ == "__main__":
    main()
