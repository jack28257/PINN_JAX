"""Generate diagnostic and illustration figures for a completed PNP PINN run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _path_setup import PROJECT_ROOT
from pnp_pinn.checkpointing import load_model, load_run_configs
from pnp_pinn.comparison import evaluate_pinn_on_reference_grid
from pnp_pinn.reference_solver import ReferenceSolution


DEFAULT_RUN_DIR = PROJECT_ROOT / "checkpoints" / "runs" / "pnp_1d_hard_phi_high_quality_20260527_074537"
DEFAULT_BASELINE_DIR = (
    PROJECT_ROOT
    / "archive"
    / "past_runs"
    / "20260527_before_fixed_voltage_hard_phi"
    / "pnp_1d_high_quality_20260526_145252"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--baseline-run-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--case", default="1.25,1.25", help="Dp,Dn case used for profile/heatmap plots.")
    parser.add_argument("--nx", type=int, default=401)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def case_name(dp: float, dn: float) -> str:
    return f"Dp{dp:g}_Dn{dn:g}".replace(".", "p")


def load_reference_solution(path: Path) -> ReferenceSolution:
    with np.load(path) as data:
        return ReferenceSolution(
            x=np.asarray(data["x"]),
            t=np.asarray(data["t"]),
            cp=np.asarray(data["cp"]),
            cn=np.asarray(data["cn"]),
            phi=np.asarray(data["phi"]),
            dp=float(data["dp"]),
            dn=float(data["dn"]),
            nx=int(data["nx"]),
            method=str(data["method"]),
            rtol=float(data["rtol"]),
            atol=float(data["atol"]),
            success=bool(data["success"]),
            message=str(data["message"]),
            nfev=int(data["nfev"]),
        )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(run_dir: Path, out_dir: Path) -> Path:
    history = np.genfromtxt(run_dir / "history.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)

    ax = axes[0]
    for name, label in [
        ("loss", "total"),
        ("dom", "domain"),
        ("bl", "boundary layer"),
        ("bc", "boundary"),
    ]:
        ax.semilogy(history["step"], history[name], label=label, linewidth=1.6)
    ax.set_ylabel("loss")
    ax.set_title("Training History")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=4, fontsize=9)

    ax = axes[1]
    ax.plot(history["step"], history["current_steps_per_second"], label="current", linewidth=1.4)
    ax.plot(history["step"], history["average_steps_per_second"], label="average", linewidth=1.4)
    ax.set_xlabel("training step")
    ax.set_ylabel("steps / second")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    path = out_dir / "training_history.png"
    save_figure(fig, path)
    return path


def plot_phase_a_summary(run_dir: Path, out_dir: Path) -> Path:
    summary = load_json(run_dir / "diagnostics" / "phase_a_summary.json")
    held = summary["held_out_residuals"]
    bc = summary["boundary_conditions"]["combined"]
    physical = summary["physical_sanity"]["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    labels = ["domain", "boundary layer", "boundary"]
    values = [
        held["domain_total_rms"],
        held["boundary_layer_total_rms"],
        held["boundary_total_rms"],
    ]
    axes[0].bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMS")
    axes[0].set_title("Held-Out Residual RMS")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", which="both", alpha=0.25)

    labels = ["Jp RMS", "Jn RMS", "phi RMS", "phi max"]
    values = [
        bc["flux_positive"]["rms"],
        bc["flux_negative"]["rms"],
        bc["phi_error"]["rms"],
        bc["phi_error"]["max_abs"],
    ]
    axes[1].bar(labels, values, color=["#4C78A8", "#72B7B2", "#E45756", "#B279A2"])
    axes[1].set_yscale("symlog", linthresh=1e-8)
    axes[1].set_title("Boundary Diagnostics")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", which="both", alpha=0.25)

    labels = ["min cp", "min cn", "max |phi|", "max cp drift", "max cn drift"]
    values = [
        physical["cp_min_global"],
        physical["cn_min_global"],
        physical["max_abs_phi_global"],
        physical["cp_mass_relative_drift_max"],
        physical["cn_mass_relative_drift_max"],
    ]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#B279A2"]
    axes[2].bar(labels, values, color=colors)
    axes[2].set_yscale("log")
    axes[2].set_title("Physical Sanity")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].grid(True, axis="y", which="both", alpha=0.25)

    path = out_dir / "phase_a_summary.png"
    save_figure(fig, path)
    return path


def phase_b_case_labels(summary: dict) -> list[str]:
    return [f"{case['Dp']:g},{case['Dn']:g}" for case in summary["cases"]]


def plot_phase_b_comparison(run_dir: Path, baseline_run_dir: Path | None, out_dir: Path) -> Path:
    summary = load_json(run_dir / "diagnostics" / "phase_b_reference_summary.json")
    baseline = None
    if baseline_run_dir is not None and (baseline_run_dir / "diagnostics" / "phase_b_reference_summary.json").exists():
        baseline = load_json(baseline_run_dir / "diagnostics" / "phase_b_reference_summary.json")

    labels = phase_b_case_labels(summary)
    x = np.arange(len(labels))
    width = 0.23

    cp = np.array([case["pinn_vs_reference"]["cp_relative_l2"] for case in summary["cases"]])
    cn = np.array([case["pinn_vs_reference"]["cn_relative_l2"] for case in summary["cases"]])
    phi = np.array([case["pinn_vs_reference"]["phi_relative_l2"] for case in summary["cases"]])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].bar(x - width, cp, width, label="cp")
    axes[0].bar(x, cn, width, label="cn")
    axes[0].bar(x + width, phi, width, label="phi")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Dp,Dn")
    axes[0].set_ylabel("relative L2 error")
    axes[0].set_title("PINN vs Reference")
    axes[0].grid(True, axis="y", which="both", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x, phi, width=0.35, label="hard phi", color="#4C78A8")
    if baseline is not None:
        base_phi = np.array(
            [case["pinn_vs_reference"]["phi_relative_l2"] for case in baseline["cases"]]
        )
        axes[1].bar(x + 0.38, base_phi, width=0.35, label="soft phi baseline", color="#E45756")
        axes[1].set_xticks(x + 0.19)
    else:
        axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Dp,Dn")
    axes[1].set_ylabel("phi relative L2 error")
    axes[1].set_title("Electric Potential Improvement")
    axes[1].grid(True, axis="y", which="both", alpha=0.25)
    axes[1].legend()

    path = out_dir / "phase_b_reference_errors.png"
    save_figure(fig, path)
    return path


def reference_file(run_dir: Path, dp: float, dn: float, nx: int) -> Path:
    return run_dir / "diagnostics" / "reference_solutions" / f"reference_{case_name(dp, dn)}_nx{nx}.npz"


def evaluate_model_for_reference(run_dir: Path, solution: ReferenceSolution) -> dict[str, np.ndarray]:
    pnp_cfg, _ = load_run_configs(run_dir)
    model = load_model(run_dir / "model_final.eqx", pnp_cfg)
    return evaluate_pinn_on_reference_grid(model, solution)


def plot_profiles(run_dir: Path, solution: ReferenceSolution, pred: dict[str, np.ndarray], out_dir: Path) -> Path:
    t_indices = [0, len(solution.t) // 2, len(solution.t) - 1]
    fields = [("cp", "c_p"), ("cn", "c_n"), ("phi", "phi")]
    fig, axes = plt.subplots(3, len(t_indices), figsize=(13.5, 9.5), sharex=True)

    for row, (field, label) in enumerate(fields):
        for col, idx in enumerate(t_indices):
            ax = axes[row, col]
            ax.plot(solution.x, getattr(solution, field)[idx], color="#222222", linewidth=2.0, label="reference")
            ax.plot(solution.x, pred[field][idx], color="#E45756", linestyle="--", linewidth=1.8, label="PINN")
            if row == 0:
                ax.set_title(f"t = {solution.t[idx]:.3f}")
            if col == 0:
                ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            if row == 0 and col == len(t_indices) - 1:
                ax.legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("x")

    fig.suptitle(f"PINN vs Reference Profiles, Dp={solution.dp:g}, Dn={solution.dn:g}", y=1.01)
    path = out_dir / f"profiles_{case_name(solution.dp, solution.dn)}.png"
    save_figure(fig, path)
    return path


def plot_heatmaps(run_dir: Path, solution: ReferenceSolution, pred: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    paths = []
    extent = [solution.x[0], solution.x[-1], solution.t[0], solution.t[-1]]
    fields = [("cp", "c_p"), ("cn", "c_n"), ("phi", "phi")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, (field, label) in zip(axes, fields):
        im = ax.imshow(pred[field], origin="lower", aspect="auto", extent=extent, cmap="viridis")
        ax.set_title(f"PINN {label}")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Predicted Solution, Dp={solution.dp:g}, Dn={solution.dn:g}")
    path = out_dir / f"solution_heatmaps_{case_name(solution.dp, solution.dn)}.png"
    save_figure(fig, path)
    paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, (field, label) in zip(axes, fields):
        err = np.abs(pred[field] - getattr(solution, field))
        im = ax.imshow(err, origin="lower", aspect="auto", extent=extent, cmap="magma")
        ax.set_title(f"|PINN - ref| {label}")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Absolute Error, Dp={solution.dp:g}, Dn={solution.dn:g}")
    path = out_dir / f"error_heatmaps_{case_name(solution.dp, solution.dn)}.png"
    save_figure(fig, path)
    paths.append(path)
    return paths


def plot_mass_drift(solution: ReferenceSolution, pred: dict[str, np.ndarray], out_dir: Path) -> Path:
    cp_mass = np.trapezoid(pred["cp"], solution.x, axis=1)
    cn_mass = np.trapezoid(pred["cn"], solution.x, axis=1)
    cp_ref = np.trapezoid(solution.cp, solution.x, axis=1)
    cn_ref = np.trapezoid(solution.cn, solution.x, axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(solution.t, cp_mass - cp_mass[0], label="PINN cp", linewidth=1.8)
    ax.plot(solution.t, cn_mass - cn_mass[0], label="PINN cn", linewidth=1.8)
    ax.plot(solution.t, cp_ref - cp_ref[0], "--", label="reference cp", linewidth=1.4)
    ax.plot(solution.t, cn_ref - cn_ref[0], "--", label="reference cn", linewidth=1.4)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("mass change")
    ax.set_title(f"Mass Conservation, Dp={solution.dp:g}, Dn={solution.dn:g}")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    path = out_dir / f"mass_drift_{case_name(solution.dp, solution.dn)}.png"
    save_figure(fig, path)
    return path


def write_index(paths: list[Path], run_dir: Path, out_dir: Path) -> Path:
    descriptions = {
        "training_history.png": "Training loss components and training speed.",
        "phase_a_summary.png": "Held-out residuals, boundary diagnostics, and physical sanity checks.",
        "phase_b_reference_errors.png": "PINN-vs-reference errors across parameter cases, including soft-phi baseline comparison.",
        "profiles_Dp1p25_Dn1p25.png": "Line profiles against the finite-difference reference at early, middle, and final times.",
        "solution_heatmaps_Dp1p25_Dn1p25.png": "Predicted cp, cn, and phi over the x-t grid.",
        "error_heatmaps_Dp1p25_Dn1p25.png": "Absolute error against the finite-difference reference over the x-t grid.",
        "mass_drift_Dp1p25_Dn1p25.png": "PINN and reference mass conservation over time.",
    }
    lines = [
        "# Diagnostic Figures",
        "",
        f"Run: `{run_dir}`",
        "",
    ]
    for path in paths:
        description = descriptions.get(path.name, "Generated diagnostic figure.")
        lines.append(f"- `{path.name}`: {description}")
    lines.append("")
    index_path = out_dir / "FIGURE_INDEX.md"
    index_path.write_text("\n".join(lines))
    return index_path


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    baseline_run_dir = resolve_path(args.baseline_run_dir) if args.baseline_run_dir else None
    dp, dn = [float(item.strip()) for item in args.case.split(",")]
    out_dir = run_dir / "diagnostics" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    paths.append(plot_training_history(run_dir, out_dir))
    paths.append(plot_phase_a_summary(run_dir, out_dir))
    paths.append(plot_phase_b_comparison(run_dir, baseline_run_dir, out_dir))

    ref_path = reference_file(run_dir, dp, dn, args.nx)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference solution not found: {ref_path}")
    solution = load_reference_solution(ref_path)
    pred = evaluate_model_for_reference(run_dir, solution)
    paths.append(plot_profiles(run_dir, solution, pred, out_dir))
    paths.extend(plot_heatmaps(run_dir, solution, pred, out_dir))
    paths.append(plot_mass_drift(solution, pred, out_dir))
    index_path = write_index(paths, run_dir, out_dir)

    print("Figures written to:", out_dir)
    for path in paths:
        print("  -", path.name)
    print("Index:", index_path)


if __name__ == "__main__":
    main()
