# Phase A PINN Diagnostics

Run directory: `/Users/zhengjia/Desktop/UWaterloo Master/T3/JAX/checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537`
Checkpoint: `/Users/zhengjia/Desktop/UWaterloo Master/T3/JAX/checkpoints/runs/pnp_1d_hard_phi_high_quality_20260527_074537/model_final.eqx`

## Training Summary

- Status: `completed`
- Final step: `500000`
- Final loss: `1.236948e-02`
- Final dom/bl/bc: `2.227460e-03`, `5.163684e-03`, `7.020860e-03`
- Best loss: `3.645345e-03` at step `450000`

## Held-Out Residual RMS

- Domain total RMS: `1.030480e-01`
- Boundary-layer total RMS: `4.290570e-02`
- Boundary total RMS: `1.600171e-02`

## Boundary Combined Max Errors

- Flux positive max abs: `2.478699e-01`
- Flux negative max abs: `2.679297e-01`
- Phi wall max abs: `0.000000e+00`
- Worst left phi error: `0.000000e+00` at `t=8.279832e-02`, `Dp=1.138`, `Dn=1.306`
- Worst right phi error: `0.000000e+00` at `t=8.279832e-02`, `Dp=1.138`, `Dn=1.306`

## Initial Condition Max Errors

- cp(x,0)-cp_init max abs: `0.000000e+00`
- cn(x,0)-cn_init max abs: `0.000000e+00`

## Physical Sanity

- Global cp min: `5.866663e-01`
- Global cn min: `5.894203e-01`
- Max |phi|: `5.000000e-01`
- Max cp mass drift: `1.612544e-03`
- Max cn mass drift: `3.250003e-03`

## Files

- `phase_a_summary.json`: full numeric diagnostics
- `phase_a_report.md`: this report
