# Parameterized PINN for 1D Poisson-Nernst-Planck Inference

This project trains and evaluates a local JAX/Equinox physics-informed neural
network (PINN) surrogate for a nondimensional one-dimensional
Poisson-Nernst-Planck (PNP) system. The trained model represents the
parameterized forward map

```text
(x, t, Dp, Dn) -> (cp, cn, phi)
```

where `cp` and `cn` are positive- and negative-ion concentrations, `phi` is the
electric potential, and `Dp`, `Dn` are the unknown diffusion parameters. The
main use case is amortized inverse inference: train the PINN once, then reuse it
for many fast likelihood evaluations over diffusion-parameter space.

## Physical Model

The active setup is a nondimensional binary-electrolyte PNP benchmark on

```text
x in [0, 1],  t in [0, 0.2],  Dp, Dn in [0.5, 2.0].
```

The fluxes are

```text
Jp = -Dp * cp_x - Dp * zp * cp * phi_x
Jn = -Dn * cn_x - Dn * zn * cn * phi_x
```

with `zp = 1`, `zn = -1`, and `epsilon = 1`. The PDE residuals are

```text
cp_t + Jp_x = 0
cn_t + Jn_x = 0
-epsilon * phi_xx - (zp * cp + zn * cn) = 0
```

The initial and boundary conditions are

```text
cp(x, 0) = 1
cn(x, 0) = 1
phi(0, t) = -0.5
phi(1, t) =  0.5
Jp(0, t) = Jp(1, t) = 0
Jn(0, t) = Jn(1, t) = 0
```

This is a controlled blocking-electrode benchmark. The no-flux ion boundary is
intentional: the current used for inference is electrode charging current, not
Faradaic ionic flux through the boundary.

## Implementation

The active implementation lives in `src/pnp_pinn`. It uses:

```text
JAX + Equinox
8-layer tanh MLP, width 256
input dimension 4:  x, t, Dp, Dn
output dimension 3: cp, cn, phi
Adam optimization with exponential learning-rate decay
NTK-style group weighting for residual groups
```

The model enforces two constraints structurally:

```text
cp = 1 + t * raw_cp
cn = 1 + t * raw_cn
phi = wall_phi(x) + 4 * xi * (1 - xi) * raw_phi
```

Therefore the concentration initial condition and fixed wall voltages are exact
by construction.

The local runtime defaults to CPU. On the local Apple M4 environment, CPU is the
stable backend for the nested automatic differentiation used by the PINN.

## Current Trained Run

The main completed run is:

```text
checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537
```

It completed `500000` training steps on CPU. The local run directory may contain
additional generated artifacts. This repository intentionally tracks only the
compact current-run subset needed to inspect and reproduce the present project:
the final model/state, history, metadata, core diagnostics, and current-based
inverse-problem outputs.

Key files:

```text
RUN_SUMMARY.md
run_metadata.json
history.csv
model_final.eqx
training_state_final.eqx
diagnostics/phase_a_report.md
diagnostics/phase_b_reference_report.md
diagnostics/phase_c_current_probe/REPORT.md
```

## Training

Start a fresh high-quality training run:

```bash
python3 scripts/train_local.py --preset high --hard-phi
```

Run a short sanity check:

```bash
python3 scripts/train_local.py --preset good --hard-phi --max-steps 10
```

Each run writes a timestamped folder under:

```text
checkpoints/runs/
```

The saved full training state stores the model, optimizer state, PRNG key,
adaptive group weights, and global step.

## Resume Training

Resume an existing run exactly:

```bash
python3 scripts/resume_local.py \
  --run-dir checkpoints/runs/<run_name> \
  --max-steps 600000
```

`--max-steps` is the final global step, not the number of extra steps.

## Reference Solver

The benchmark reference solver is independent of the PINN. It uses a
finite-difference method-of-lines discretization:

```text
Poisson equation: banded linear solve at each RHS evaluation
concentration dynamics: finite-volume-style flux differences
time integration: scipy.integrate.solve_ivp with BDF by default
```

Reference solutions are used for forward validation and for generating
synthetic current observations.

## Diagnostics

Phase A diagnostics check the trained PINN without using a reference solution:

```bash
python3 scripts/diagnose_run.py \
  --run-dir checkpoints/runs/<run_name>
```

Phase B diagnostics compare the PINN against finite-difference/BDF reference
solutions:

```bash
python3 scripts/compare_reference.py \
  --run-dir checkpoints/runs/<run_name>
```

Diagnostic figures are generated with:

```bash
python3 scripts/visualize_run.py \
  --run-dir checkpoints/runs/<run_name> \
  --case 1.25,1.25 \
  --nx 401
```

## Current-Based Inverse Problem

The current inverse diagnostic is the primary inference
experiment. It does not assume direct concentration-field observations.
Instead:

1. The reference solver generates a clean electrode charging-current time series.
2. Independent Gaussian measurement noise is added.
3. The trained PINN predicts the same current observable for candidate
   `(Dp, Dn)` values.
4. A Gaussian likelihood is evaluated on a rectangular parameter grid.
5. The grid posterior is normalized and summarized.

Run the current-based inverse probe:

```bash
python3 scripts/run_current_probe.py \
  --run-dir checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537
```

The default output directory is:

```text
checkpoints/runs/<run_name>/diagnostics/phase_c_current_probe/
```

Main outputs:

```text
REPORT.md
summary.json
current_observations.csv
posterior_grid.csv
current_fit.png
posterior_contour.png
marginals.png
```

## Project Layout

```text
.
|-- README.md
|-- PROJECT_STATUS.md
|-- pyproject.toml
|-- requirements.txt
|-- src/
|   `-- pnp_pinn/
|       |-- model.py              PINN architecture and pointwise physics
|       |-- train.py              Training and resume logic
|       |-- losses.py             PDE, boundary, and NTK-weighted losses
|       |-- reference_solver.py   Finite-difference/BDF benchmark solver
|       |-- current_observable.py Current observable and posterior utilities
|       |-- diagnostics.py        Phase A diagnostic checks
|       |-- comparison.py         Phase B reference-comparison helpers
|       `-- config.py             Physical and training configuration
|-- scripts/
|   |-- train_local.py            Fresh training run
|   |-- resume_local.py           Resume a saved training state
|   |-- diagnose_run.py           PINN-only diagnostics
|   |-- compare_reference.py      Reference-solver comparison
|   |-- visualize_run.py          Diagnostic figures
|   |-- run_current_probe.py      Current-based inverse problem
|   `-- benchmark_step_time.py    Local step-time benchmark
|-- checkpoints/
|   `-- runs/
|       `-- pnp_1d_hard_phi_high_quality_20260527_074537/
|           |-- RUN_SUMMARY.md
|           |-- model_final.eqx
|           |-- training_state_final.eqx
|           |-- history.csv
|           |-- run_metadata.json
|           `-- diagnostics/
|               |-- phase_a_report.md
|               |-- phase_b_reference_report.md
|               `-- phase_c_current_probe/
|                   |-- REPORT.md
|                   |-- summary.json
|                   |-- current_observations.csv
|                   |-- posterior_grid.csv
|                   |-- current_fit.png
|                   |-- posterior_contour.png
|                   `-- marginals.png
|-- docs/
|   `-- reports/
|       |-- pnp_pinn_project_technical_report.tex
|       `-- pnp_pinn_project_technical_report.pdf
`-- notebooks/
    `-- pnp_local_training.ipynb
```

Only the compact current-run artifact subset is tracked. Local archives,
checkpoint backups, presentation material, and unrelated generated outputs are
excluded by `.gitignore`.

## Main Python Entry Points

```text
scripts/train_local.py          Fresh training run
scripts/resume_local.py         Exact continuation from full training state
scripts/diagnose_run.py         Phase A PINN diagnostics
scripts/compare_reference.py    Phase B reference comparison
scripts/visualize_run.py        Diagnostic figure generation
scripts/run_current_probe.py    Current-based posterior inference
scripts/benchmark_step_time.py  Local step-time benchmark
```
