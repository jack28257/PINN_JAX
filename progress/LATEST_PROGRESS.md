# Latest Progress

Last updated: May 31, 2026

## One-Sentence Summary

The project now has a complete controlled pipeline: train a parameterized 1D
PNP PINN, validate it against an independent finite-difference/BDF reference
solver, generate synthetic charging-current observations, and infer the
diffusion parameters `Dp` and `Dn` through a grid posterior.

## Current State

- The main PINN run completed `500000` Adam steps on CPU.
- The model enforces the concentration initial condition and fixed wall
  voltages exactly by construction.
- Forward validation against the reference solver shows strong concentration
  agreement and acceptable potential agreement for the current inverse test.
- The inverse problem now uses current observations rather than direct
  concentration-field observations.
- The repository has been cleaned into a compact supervisor-facing snapshot.

## Latest Inference Result

Synthetic current observations were generated from the independent reference
solver at true parameters:

```text
Dp = 1.25
Dn = 1.25
```

The PINN-based posterior summary is:

```text
MAP:            Dp=1.25, Dn=1.2125
posterior mean: Dp=1.23531, Dn=1.23677
95% Dp interval: [1.0625, 1.4]
95% Dn interval: [1.0625, 1.4375]
true covered:   yes
```

## Interpretation

The current-only inverse problem is working as a proof of concept. The result
does not claim perfect point recovery. The useful result is that noisy
charging-current data produce a posterior distribution that covers the true
diffusion parameters and shows the uncertainty/correlation structure between
`Dp` and `Dn`.

## Main Limitations

- The model is one-dimensional and nondimensional.
- The electrode is idealized as blocking and fixed-voltage.
- The current observations are synthetic.
- The likelihood assumes independent Gaussian noise.
- The posterior is evaluated on a rectangular grid, not by MCMC.

## Immediate Next Steps

- Prepare a 13-minute presentation for the 15-minute slot.
- Use the current-fit plot, posterior contour, and marginal posterior plots as
  the main result figures.
- Clearly explain why current is used instead of direct concentration
  observations.
- Keep the presentation focused on the inverse-problem pipeline rather than on
  every training detail.

## Where To Look

```text
README.md
PROJECT_STATUS.md
docs/reports/pnp_pinn_project_technical_report.pdf
checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537/diagnostics/phase_c_current_probe/REPORT.md
```
