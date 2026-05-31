# Project Status

## Current Goal

Use a trained parameterized PINN as a fast forward surrogate for a 1D
Poisson-Nernst-Planck inverse problem. The inverse problem uses synthetic
electrode charging-current observations to infer the diffusion parameters
`Dp` and `Dn`.

## Main Run

```text
checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537
```

This run completed `500000` Adam steps on CPU. The final model enforces the
concentration initial condition and fixed wall voltages exactly by construction.

## Repository Scope

This repository is a compact project snapshot. It tracks the source code,
current documentation, the final model/state for the main run, and the
diagnostics needed to understand the current-based inverse result. Local
archives, presentation material, checkpoint backups, and unrelated generated
artifacts are intentionally not tracked.

For the shortest high-level progress update, start with
[progress/LATEST_PROGRESS.md](progress/LATEST_PROGRESS.md).

## Current Inference Result

Current-based inverse probe:

```text
checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537/diagnostics/phase_c_current_probe
```

Summary:

```text
true:           Dp=1.25, Dn=1.25
MAP:            Dp=1.25, Dn=1.2125
posterior mean: Dp=1.23531, Dn=1.23677
95% Dp interval: [1.0625, 1.4]
95% Dn interval: [1.0625, 1.4375]
true covered:   yes
```

## Useful Commands

Run a short training sanity check:

```bash
python3 scripts/train_local.py --preset good --hard-phi --max-steps 10
```

Regenerate the current-based inverse probe:

```bash
python3 scripts/run_current_probe.py \
  --run-dir checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537
```

## Main Files

```text
README.md
docs/reports/pnp_pinn_project_technical_report.pdf
src/pnp_pinn/
scripts/
checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537/RUN_SUMMARY.md
checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537/diagnostics/phase_c_current_probe/REPORT.md
```

## Limitations

- The PNP model is one-dimensional and nondimensional.
- The electrode interface is idealized as blocking and fixed-voltage.
- The current data are synthetic.
- The noise model is independent Gaussian noise.
- The posterior is computed on a grid, not by MCMC.
