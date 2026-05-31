# Phase B Reference Comparison

Run directory: `/Users/zhengjia/Desktop/UWaterloo Master/T3/JAX/checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537`
Checkpoint: `/Users/zhengjia/Desktop/UWaterloo Master/T3/JAX/checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537/model_final.eqx`
Reference method: `BDF`
Reference nx values: `[101, 201, 401]`
Reference nt: `41`

## PINN vs Finest Reference

| Dp | Dn | cp rel L2 | cn rel L2 | phi rel L2 | cp max abs | cn max abs | phi max abs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.25 | 1.25 | 8.493e-04 | 5.090e-04 | 2.499e-02 | 2.385e-03 | 1.470e-03 | 6.675e-02 |
| 0.5 | 0.5 | 1.575e-03 | 1.186e-03 | 1.689e-02 | 5.604e-03 | 4.972e-03 | 4.864e-02 |
| 2 | 2 | 8.223e-04 | 6.199e-04 | 2.796e-02 | 2.595e-03 | 2.904e-03 | 7.333e-02 |
| 0.5 | 2 | 3.304e-03 | 2.580e-03 | 2.604e-02 | 7.682e-03 | 5.885e-03 | 6.764e-02 |
| 2 | 0.5 | 5.664e-04 | 1.080e-03 | 2.675e-02 | 2.601e-03 | 3.182e-03 | 7.239e-02 |

## Reference Grid Convergence

These compare each coarser reference grid against the next finer grid, interpolated onto the coarser grid.

| Dp | Dn | comparison | cp rel L2 | cn rel L2 | phi rel L2 |
|---:|---:|---|---:|---:|---:|
| 1.25 | 1.25 | nx101_vs_nx201 | 1.157e-03 | 1.157e-03 | 1.487e-04 |
| 1.25 | 1.25 | nx201_vs_nx401 | 5.759e-04 | 5.759e-04 | 7.472e-05 |
| 0.5 | 0.5 | nx101_vs_nx201 | 1.262e-03 | 1.262e-03 | 1.532e-04 |
| 0.5 | 0.5 | nx201_vs_nx401 | 6.300e-04 | 6.300e-04 | 7.732e-05 |
| 2 | 2 | nx101_vs_nx201 | 9.830e-04 | 9.830e-04 | 1.251e-04 |
| 2 | 2 | nx201_vs_nx401 | 4.891e-04 | 4.891e-04 | 6.278e-05 |
| 0.5 | 2 | nx101_vs_nx201 | 1.270e-03 | 9.796e-04 | 1.484e-04 |
| 0.5 | 2 | nx201_vs_nx401 | 6.343e-04 | 4.873e-04 | 7.460e-05 |
| 2 | 0.5 | nx101_vs_nx201 | 9.796e-04 | 1.270e-03 | 1.484e-04 |
| 2 | 0.5 | nx201_vs_nx401 | 4.873e-04 | 6.343e-04 | 7.460e-05 |

## Solver Status

| Dp | Dn | nx | success | nfev | seconds | message |
|---:|---:|---:|:---:|---:|---:|---|
| 1.25 | 1.25 | 101 | True | 202 | 0.02 | The solver successfully reached the end of the integration interval. |
| 1.25 | 1.25 | 201 | True | 214 | 0.04 | The solver successfully reached the end of the integration interval. |
| 1.25 | 1.25 | 401 | True | 226 | 0.16 | The solver successfully reached the end of the integration interval. |
| 0.5 | 0.5 | 101 | True | 174 | 0.02 | The solver successfully reached the end of the integration interval. |
| 0.5 | 0.5 | 201 | True | 186 | 0.04 | The solver successfully reached the end of the integration interval. |
| 0.5 | 0.5 | 401 | True | 198 | 0.14 | The solver successfully reached the end of the integration interval. |
| 2 | 2 | 101 | True | 218 | 0.02 | The solver successfully reached the end of the integration interval. |
| 2 | 2 | 201 | True | 232 | 0.05 | The solver successfully reached the end of the integration interval. |
| 2 | 2 | 401 | True | 242 | 0.16 | The solver successfully reached the end of the integration interval. |
| 0.5 | 2 | 101 | True | 208 | 0.02 | The solver successfully reached the end of the integration interval. |
| 0.5 | 2 | 201 | True | 222 | 0.04 | The solver successfully reached the end of the integration interval. |
| 0.5 | 2 | 401 | True | 232 | 0.15 | The solver successfully reached the end of the integration interval. |
| 2 | 0.5 | 101 | True | 208 | 0.02 | The solver successfully reached the end of the integration interval. |
| 2 | 0.5 | 201 | True | 222 | 0.05 | The solver successfully reached the end of the integration interval. |
| 2 | 0.5 | 401 | True | 232 | 0.16 | The solver successfully reached the end of the integration interval. |

## Files

- `phase_b_reference_summary.json`: full numeric comparison
- `phase_b_reference_report.md`: this report
- `reference_*.npz`: saved finite-difference reference solutions
