# PNP 1D Hard-Phi High-Quality Run

This is the fixed-voltage hard-constrained electric-potential run.

## Identity

- Run directory: `checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537`
- Model type: JAX/Equinox 1D parametric PNP PINN
- Boundary voltage style: hard constrained `phi(0)=-0.5`, `phi(1)=0.5`
- Initial condition style: hard constrained for `cp` and `cn`
- Device backend: CPU
- Status: completed
- Final step: `500000`

## Training Summary

- Final loss: `1.236948e-02`
- Final domain loss: `2.227460e-03`
- Final boundary-layer loss: `5.163684e-03`
- Final boundary loss: `7.020860e-03`
- Best logged loss: `3.645345e-03` at step `450000`
- Average speed: `17.0235` steps/s
- Elapsed training time: about `8h 9m`

## Phase A Summary

- Initial condition errors for `cp` and `cn`: exactly zero
- `phi` wall max error: exactly zero
- Concentrations stayed positive on the diagnostic grid
- Global `cp` min: `5.866663e-01`
- Global `cn` min: `5.894203e-01`
- Max `cp` mass drift: `1.612544e-03`
- Max `cn` mass drift: `3.250003e-03`
- Held-out domain total RMS: `1.030480e-01`
- Held-out boundary-layer total RMS: `4.290570e-02`
- Held-out boundary total RMS: `1.600171e-02`

## Phase B Reference Comparison

Compared against finite-difference/BDF references at `nx=401`, `nt=41`.

| Dp | Dn | cp rel L2 | cn rel L2 | phi rel L2 | phi max abs |
|---:|---:|---:|---:|---:|---:|
| 1.25 | 1.25 | `8.493e-04` | `5.090e-04` | `2.499e-02` | `6.675e-02` |
| 0.5 | 0.5 | `1.575e-03` | `1.186e-03` | `1.689e-02` | `4.864e-02` |
| 2.0 | 2.0 | `8.223e-04` | `6.199e-04` | `2.796e-02` | `7.333e-02` |
| 0.5 | 2.0 | `3.304e-03` | `2.580e-03` | `2.604e-02` | `6.764e-02` |
| 2.0 | 0.5 | `5.664e-04` | `1.080e-03` | `2.675e-02` | `7.239e-02` |

## Interpretation

This run substantially improves the electric-potential solution compared with
the archived soft-phi baseline. The previous run had `phi` relative L2 errors
around `0.11-0.14`; this run is around `0.017-0.028` on the tested reference
cases. Concentration errors remain small, typically around `1e-3`.

Phase A still shows some held-out residual outliers, especially in the domain
Poisson residual, so the model is not mathematically perfect. Phase B is the
more important result here: the predicted solution now matches the reference
solver much better, especially for `phi`.

## Saved Artifacts

- `model_final.eqx`: final model weights
- `training_state_final.eqx`: final model, optimizer state, random key, weights, and step
- `model_latest.eqx` and `training_state_latest.eqx`: latest checkpoint mirrors
- `history.csv` and `history.json`: logged training curve
- `run_metadata.json`: full configuration and backend metadata
- `diagnostics/phase_a_report.md`
- `diagnostics/phase_a_summary.json`
- `diagnostics/phase_b_reference_report.md`
- `diagnostics/phase_b_reference_summary.json`
- `diagnostics/reference_solutions/*.npz`
- `diagnostics/figures/*.png`: diagnostic and illustration figures
- `source_snapshot/`: source code and scripts used for this run
- `CHECKSUMS.sha256`: checksum manifest
- `ARTIFACTS.txt`: file listing
