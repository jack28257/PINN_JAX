"""Run Phase C: posterior inference from synthetic charging-current data."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from _path_setup import PROJECT_ROOT

import matplotlib.pyplot as plt
import numpy as np

from pnp_pinn.checkpointing import load_model, load_run_configs, save_json
from pnp_pinn.current_observable import (
    evaluate_current_posterior_grid,
    interval_current_from_charge,
    make_current_observations,
    predict_current_observations,
    reference_electrode_charge,
    save_current_observations_csv,
    summarize_current_posterior_grid,
)
from pnp_pinn.inference import (
    load_reference_npz,
    reference_case_name,
    save_posterior_grid_csv,
)
from pnp_pinn.reference_solver import save_reference_npz, solve_reference_pnp


DEFAULT_RUN_DIR = PROJECT_ROOT / "checkpoints" / "runs" / "pnp_1d_hard_phi_high_quality_20260527_074537"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint-name", default="model_final.eqx")
    parser.add_argument("--out-name", default="phase_c_current_probe")
    parser.add_argument("--dp-true", type=float, default=1.25)
    parser.add_argument("--dn-true", type=float, default=1.25)
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument(
        "--relative-sigma",
        type=float,
        default=0.02,
        help="Noise standard deviation as a fraction of max absolute clean current.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Absolute current noise standard deviation. Overrides --relative-sigma.",
    )
    parser.add_argument(
        "--skip-initial-intervals",
        type=int,
        default=1,
        help="Drop this many earliest current intervals from the likelihood.",
    )
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--quadrature-nx",
        type=int,
        default=51,
        help="Spatial quadrature grid used to compute PINN electrode charge.",
    )
    parser.add_argument("--reference-nx", type=int, default=401)
    parser.add_argument("--reference-nt", type=int, default=81)
    parser.add_argument("--reference-method", default="BDF", choices=["BDF", "Radau", "LSODA"])
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--recompute-reference", action="store_true")
    return parser.parse_args()


def _resolve_run_dir(path: str) -> Path:
    run_dir = Path(path)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    return run_dir


def _reference_path(run_dir: Path, dp: float, dn: float, nx: int, nt: int) -> Path:
    return (
        run_dir
        / "diagnostics"
        / "reference_solutions"
        / f"reference_{reference_case_name(dp, dn)}_nx{nx}_nt{nt}.npz"
    )


def _load_or_solve_reference(args: argparse.Namespace, run_dir: Path, pnp_cfg):
    ref_path = _reference_path(run_dir, args.dp_true, args.dn_true, args.reference_nx, args.reference_nt)
    if ref_path.exists() and not args.recompute_reference:
        print(f"Using cached reference solution: {ref_path}")
        return load_reference_npz(ref_path), ref_path, True, 0.0

    print(
        f"Solving reference solution Dp={args.dp_true:g}, Dn={args.dn_true:g}, "
        f"nx={args.reference_nx}, nt={args.reference_nt}"
    )
    start = time.perf_counter()
    solution = solve_reference_pnp(
        pnp_cfg,
        dp=args.dp_true,
        dn=args.dn_true,
        nx=args.reference_nx,
        nt=args.reference_nt,
        method=args.reference_method,
        rtol=args.rtol,
        atol=args.atol,
    )
    seconds = time.perf_counter() - start
    save_reference_npz(ref_path, solution)
    return solution, ref_path, False, seconds


def _current_noise_sigma(args: argparse.Namespace, solution, pnp_cfg) -> tuple[float, float]:
    charge = reference_electrode_charge(solution, pnp_cfg, side=args.side)
    *_, clean_current = interval_current_from_charge(
        solution.t,
        charge,
        skip_initial_intervals=args.skip_initial_intervals,
    )
    current_scale = max(float(np.max(np.abs(clean_current))), 1e-12)
    if args.sigma is not None:
        if args.sigma <= 0.0:
            raise ValueError("--sigma must be positive")
        return float(args.sigma), current_scale
    if args.relative_sigma <= 0.0:
        raise ValueError("--relative-sigma must be positive")
    return float(args.relative_sigma * current_scale), current_scale


def _plot_posterior_contour(grid: dict, summary: dict, out_path: Path) -> None:
    dp_values = np.asarray(grid["dp_values"])
    dn_values = np.asarray(grid["dn_values"])
    posterior = np.asarray(grid["posterior_mass"])
    relative = posterior / np.max(posterior)

    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=180)
    x_mesh, y_mesh = np.meshgrid(dp_values, dn_values, indexing="xy")
    contour = ax.contourf(x_mesh, y_mesh, relative.T, levels=24, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="relative posterior mass")
    ax.scatter(
        summary["true"]["Dp"],
        summary["true"]["Dn"],
        marker="*",
        s=180,
        color="#f97316",
        edgecolor="white",
        linewidth=0.8,
        label="true",
        zorder=4,
    )
    ax.scatter(
        summary["map"]["Dp"],
        summary["map"]["Dn"],
        marker="x",
        s=80,
        color="white",
        linewidth=2.0,
        label="MAP",
        zorder=5,
    )
    ax.set_xlabel(r"$D_+$")
    ax.set_ylabel(r"$D_-$")
    ax.set_title("Posterior from synthetic charging-current data")
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(dp_values.min(), dp_values.max())
    ax.set_ylim(dn_values.min(), dn_values.max())
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_marginals(grid: dict, summary: dict, out_path: Path) -> None:
    dp_values = np.asarray(grid["dp_values"])
    dn_values = np.asarray(grid["dn_values"])
    posterior = np.asarray(grid["posterior_mass"])
    dp_marginal = posterior.sum(axis=1)
    dn_marginal = posterior.sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=180)
    for ax, values, marginal, name in [
        (axes[0], dp_values, dp_marginal, "Dp"),
        (axes[1], dn_values, dn_marginal, "Dn"),
    ]:
        ax.plot(values, marginal, color="#3430b5", linewidth=2.0)
        ax.axvline(summary["true"][name], color="#f97316", linestyle="--", linewidth=1.8, label="true")
        ax.axvline(summary["map"][name], color="#047d8b", linestyle=":", linewidth=1.8, label="MAP")
        ci = summary["credible_interval"][name]
        ax.axvspan(ci["lower"], ci["upper"], color="#3430b5", alpha=0.12, label="95% interval")
        ax.set_xlabel(fr"${name[0]}_{'+' if name == 'Dp' else '-'}$")
        ax.set_ylabel("posterior mass")
        ax.set_title(f"Marginal posterior for {name}")
        ax.set_xlim(values.min(), values.max())
    axes[0].legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_current_fit(observations, map_prediction: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=180)
    ax.plot(
        observations.t_mid,
        observations.clean_current,
        color="#222222",
        linewidth=1.8,
        label="benchmark clean current",
    )
    ax.scatter(
        observations.t_mid,
        observations.observed_current,
        color="#f97316",
        s=24,
        label="noisy observed current",
        zorder=3,
    )
    ax.plot(
        observations.t_mid,
        map_prediction,
        color="#3430b5",
        linewidth=1.8,
        label="PINN current at MAP",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("interval-averaged charging current")
    ax.set_title("Synthetic current observations and PINN fit")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _make_report(summary: dict) -> str:
    posterior = summary["posterior_summary"]
    true = posterior["true"]
    map_est = posterior["map"]
    mean = posterior["mean"]
    ci = posterior["credible_interval"]
    covered = posterior["true_value_inside_joint_marginal_intervals"]

    lines = [
        "# Phase C Current-Based Inverse Probe",
        "",
        "This diagnostic generates synthetic charging-current data from the independent",
        "finite-difference/BDF benchmark solver, then uses the trained PINN as a fast",
        "forward surrogate to infer the diffusion parameters.",
        "",
        "The current is the interval-averaged electrode charging current derived from",
        "the electrode charge response. It is not the boundary ionic flux; the active",
        "PNP setup uses blocking no-flux ion boundaries.",
        "",
        "## Setup",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- True parameters: `Dp={true['Dp']:.6g}`, `Dn={true['Dn']:.6g}`",
        f"- Electrode side: `{summary['side']}`",
        f"- Current observations: `{summary['n_observations']}`",
        f"- Skipped initial intervals: `{summary['skip_initial_intervals']}`",
        f"- Reference charge quadrature nx: `{summary['reference_charge_quadrature_nx']}`",
        f"- PINN charge quadrature nx: `{summary['pinn_charge_quadrature_nx']}`",
        f"- Current scale: `{summary['current_scale']:.6g}`",
        f"- Noise sigma: `{summary['sigma']:.6g}`",
        f"- Relative sigma: `{summary['relative_sigma']:.6g}`",
        f"- Posterior grid: `{summary['grid_size']} x {summary['grid_size']}`",
        "",
        "## Posterior Summary",
        "",
        f"- MAP estimate: `Dp={map_est['Dp']:.6g}`, `Dn={map_est['Dn']:.6g}`",
        f"- Posterior mean: `Dp={mean['Dp']:.6g}`, `Dn={mean['Dn']:.6g}`",
        (
            f"- 95% marginal interval for Dp: "
            f"`[{ci['Dp']['lower']:.6g}, {ci['Dp']['upper']:.6g}]`"
        ),
        (
            f"- 95% marginal interval for Dn: "
            f"`[{ci['Dn']['lower']:.6g}, {ci['Dn']['upper']:.6g}]`"
        ),
        f"- True value inside both marginal intervals: `{covered}`",
        f"- Posterior mass sum check: `{posterior['posterior_sum']:.12f}`",
        "",
        "## Files",
        "",
        "- `current_observations.csv`: synthetic clean and noisy current observations",
        "- `posterior_grid.csv`: log likelihood and posterior mass for each grid point",
        "- `summary.json`: machine-readable diagnostic summary",
        "- `posterior_contour.png`: 2D posterior over Dp and Dn",
        "- `marginals.png`: marginal posteriors",
        "- `current_fit.png`: noisy current data compared with PINN current at MAP",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2")

    run_dir = _resolve_run_dir(args.run_dir)
    out_dir = run_dir / "diagnostics" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pnp_cfg, _ = load_run_configs(run_dir)
    checkpoint = run_dir / args.checkpoint_name
    model = load_model(checkpoint, pnp_cfg)
    reference, reference_path, reference_cached, reference_seconds = _load_or_solve_reference(
        args,
        run_dir,
        pnp_cfg,
    )

    sigma, current_scale = _current_noise_sigma(args, reference, pnp_cfg)
    observations = make_current_observations(
        reference,
        pnp_cfg,
        side=args.side,
        sigma=sigma,
        seed=args.seed,
        skip_initial_intervals=args.skip_initial_intervals,
    )
    save_current_observations_csv(observations, out_dir / "current_observations.csv")

    dp_values = np.linspace(pnp_cfg.dp_min, pnp_cfg.dp_max, args.grid_size)
    dn_values = np.linspace(pnp_cfg.dn_min, pnp_cfg.dn_max, args.grid_size)
    print(
        f"Evaluating current posterior grid {args.grid_size} x {args.grid_size} "
        f"for {observations.n} observations"
    )
    start = time.perf_counter()
    grid = evaluate_current_posterior_grid(
        model,
        pnp_cfg,
        observations,
        dp_values=dp_values,
        dn_values=dn_values,
        chunk_size=args.chunk_size,
        quadrature_nx=args.quadrature_nx,
    )
    posterior_seconds = time.perf_counter() - start
    save_posterior_grid_csv(grid, out_dir / "posterior_grid.csv")

    posterior_summary = summarize_current_posterior_grid(grid, observations)
    map_prediction = predict_current_observations(
        model,
        pnp_cfg,
        observations,
        posterior_summary["map"]["Dp"],
        posterior_summary["map"]["Dn"],
        quadrature_nx=args.quadrature_nx,
    )

    _plot_posterior_contour(grid, posterior_summary, out_dir / "posterior_contour.png")
    _plot_marginals(grid, posterior_summary, out_dir / "marginals.png")
    _plot_current_fit(observations, map_prediction, out_dir / "current_fit.png")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "output_dir": str(out_dir),
        "reference_path": str(reference_path),
        "reference_cached": reference_cached,
        "reference_seconds": reference_seconds,
        "reference_success": reference.success,
        "reference_message": reference.message,
        "reference_nfev": reference.nfev,
        "side": observations.side,
        "reference_charge_quadrature_nx": observations.charge_quadrature_nx,
        "pinn_charge_quadrature_nx": args.quadrature_nx,
        "skip_initial_intervals": args.skip_initial_intervals,
        "n_observations": observations.n,
        "sigma": sigma,
        "relative_sigma": sigma / current_scale,
        "current_scale": current_scale,
        "seed": args.seed,
        "grid_size": args.grid_size,
        "chunk_size": args.chunk_size,
        "posterior_seconds": posterior_seconds,
        "posterior_summary": posterior_summary,
        "files": {
            "current_observations_csv": str(out_dir / "current_observations.csv"),
            "posterior_grid_csv": str(out_dir / "posterior_grid.csv"),
            "summary_json": str(out_dir / "summary.json"),
            "report_md": str(out_dir / "REPORT.md"),
            "posterior_contour_png": str(out_dir / "posterior_contour.png"),
            "marginals_png": str(out_dir / "marginals.png"),
            "current_fit_png": str(out_dir / "current_fit.png"),
        },
    }
    save_json(summary, out_dir / "summary.json")
    (out_dir / "REPORT.md").write_text(_make_report(summary))

    print("Current inverse probe complete.")
    print("Output:", out_dir)
    print(
        "MAP:",
        f"Dp={posterior_summary['map']['Dp']:.6g},",
        f"Dn={posterior_summary['map']['Dn']:.6g}",
    )
    print(
        "Mean:",
        f"Dp={posterior_summary['mean']['Dp']:.6g},",
        f"Dn={posterior_summary['mean']['Dn']:.6g}",
    )
    print(
        "True covered by marginal intervals:",
        posterior_summary["true_value_inside_joint_marginal_intervals"],
    )


if __name__ == "__main__":
    main()
