# Phase C Current-Based Inverse Probe

This diagnostic generates synthetic charging-current data from the independent
finite-difference/BDF benchmark solver, then uses the trained PINN as a fast
forward surrogate to infer the diffusion parameters.

The current is the interval-averaged electrode charging current derived from
the electrode charge response. It is not the boundary ionic flux; the active
PNP setup uses blocking no-flux ion boundaries.

## Setup

- Run directory: `/Users/zhengjia/Desktop/UWaterloo Master/T3/JAX/checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537`
- Checkpoint: `/Users/zhengjia/Desktop/UWaterloo Master/T3/JAX/checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537/model_final.eqx`
- True parameters: `Dp=1.25`, `Dn=1.25`
- Electrode side: `left`
- Current observations: `79`
- Skipped initial intervals: `1`
- Reference charge quadrature nx: `401`
- PINN charge quadrature nx: `51`
- Current scale: `2.09434`
- Noise sigma: `0.0418868`
- Relative sigma: `0.02`
- Posterior grid: `41 x 41`

## Posterior Summary

- MAP estimate: `Dp=1.25`, `Dn=1.2125`
- Posterior mean: `Dp=1.23531`, `Dn=1.23677`
- 95% marginal interval for Dp: `[1.0625, 1.4]`
- 95% marginal interval for Dn: `[1.0625, 1.4375]`
- True value inside both marginal intervals: `True`
- Posterior mass sum check: `1.000000000000`

## Files

- `current_observations.csv`: synthetic clean and noisy current observations
- `posterior_grid.csv`: log likelihood and posterior mass for each grid point
- `summary.json`: machine-readable diagnostic summary
- `posterior_contour.png`: 2D posterior over Dp and Dn
- `marginals.png`: marginal posteriors
- `current_fit.png`: noisy current data compared with PINN current at MAP
