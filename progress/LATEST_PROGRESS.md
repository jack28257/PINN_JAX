# Latest Progress

Last updated: May 31, 2026

## High-Level Summary

The current project demonstrates a controlled inverse-problem pipeline for a
1D Poisson-Nernst-Planck (PNP) model. We use an independent reference solver to
generate synthetic charging-current observations, then use the trained
parameterized PINN as a fast forward surrogate to infer the diffusion
parameters `Dp` and `Dn`.

## How The Inference Is Done

1. Choose true diffusion parameters:

```text
Dp = 1.25
Dn = 1.25
```

2. Solve the PNP forward problem with the independent finite-difference/BDF
   reference solver.

3. Convert the reference solution into a left-electrode charging-current time
   series. The observable is interval-averaged current:

```text
I_k = (Q(t_{k+1}) - Q(t_k)) / (t_{k+1} - t_k)
```

Here `Q(t)` is the electrode charge response. This is not the boundary ionic
flux; the active setup uses blocking no-flux ion boundaries.

4. Add independent Gaussian measurement noise to the clean current:

```text
sigma = 0.02 * max(abs(clean current)) = 0.0418868
```

5. For each candidate `(Dp, Dn)` on a `41 x 41` grid, use the trained PINN to
   predict the same current observable.

6. Evaluate a Gaussian likelihood:

```text
observed current = PINN-predicted current(Dp, Dn) + Gaussian noise
```

7. Use a uniform prior over the parameter box and normalize the likelihood over
   the grid to obtain the posterior distribution of `Dp` and `Dn`.

## Settings And Assumptions

- Model: nondimensional 1D binary-electrolyte PNP.
- Domain: `x in [0, 1]`, `t in [0, 0.2]`.
- Parameter range: `Dp, Dn in [0.5, 2.0]`.
- Charges: `z+ = 1`, `z- = -1`.
- Permittivity: `epsilon = 1`.
- Initial concentrations: `cp(x,0) = 1`, `cn(x,0) = 1`.
- Voltage boundary: `phi(0,t) = -0.5`, `phi(1,t) = 0.5`.
- Ion boundary: no-flux / blocking electrodes.
- Current observable: electrode charging current, not Faradaic current and not
  ionic flux through the wall.
- Data source: synthetic current generated from the reference solver.
- Noise model: independent Gaussian current noise with relative level `2%`.
- Prior: uniform over `[0.5, 2.0] x [0.5, 2.0]`.
- Posterior method: rectangular grid posterior, not MCMC.

## Simple Result

The latest current-based inverse probe used `79` noisy current observations
from the left electrode.

```text
true:           Dp=1.25, Dn=1.25
MAP:            Dp=1.25, Dn=1.2125
posterior mean: Dp=1.23531, Dn=1.23677
95% Dp interval: [1.0625, 1.4]
95% Dn interval: [1.0625, 1.4375]
true covered:   yes
```

The main takeaway is that current-only synthetic observations can recover a
reasonable posterior region for `Dp` and `Dn`. The result should be read as a
proof of concept for the inference pipeline, not as perfect point estimation.

## Where To Look

```text
README.md
PROJECT_STATUS.md
docs/reports/pnp_pinn_project_technical_report.pdf
checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537/diagnostics/phase_c_current_probe/REPORT.md
```
