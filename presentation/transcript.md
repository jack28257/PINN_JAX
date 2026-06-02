# Presentation Transcript and Speaker Notes

Deck: `pnp_pinn_current_inference_presentation.pptx`

Target slot: 15 minutes

Prepared talk length: approximately 13 minutes

## Core Message

A trained parameterized PINN can be reused as a fast forward surrogate inside a
measurable inverse problem: given noisy electrode-current observations, we infer
a posterior distribution over the diffusion parameters `Dp` and `Dn`.

Important wording discipline:

- Say that the real observable is current `I(t)`.
- Say that `Q(t)` is a simulation-side construction used to compute the same
  charging-current signal from the benchmark/PINN state.
- Do not imply that `Q(t)` is what would be directly measured in the lab.
- Do not present the current result as a final experimental model. Present it as
  a controlled proof of concept with explicit assumptions.

## Timing Plan

| Slide | Topic | Target time |
| --- | --- | ---: |
| 1 | Core thesis | 0:50 |
| 2 | Motivation: measurable current | 1:10 |
| 3 | PNP benchmark and assumptions | 1:20 |
| 4 | Trained PINN surrogate | 1:15 |
| 5 | Synthetic current data generation | 1:10 |
| 6 | Observable definition: current vs Q(t) | 1:15 |
| 7 | Inverse problem and likelihood | 1:25 |
| 8 | Current fit at MAP | 1:20 |
| 9 | Posterior result | 1:40 |
| 10 | Why non-trivial and next steps | 1:35 |

Total prepared talk: about 13 minutes.

## Slide 1 - Core Thesis

Timing: 0:00 to 0:50

Script:

The main message of this presentation is that the project has moved beyond only
training a PINN. The current goal is to use a trained PINN as a reusable forward
model inside an inverse problem.

In this setup, the unknown parameters are the positive- and negative-ion
diffusion parameters, `Dp` and `Dn`. The observation is not a hidden
concentration field. The observation is an electrode-level current signal. I
generate synthetic noisy current observations from an independent PNP benchmark
solver, then ask whether the trained PINN can recover a posterior distribution
over `Dp` and `Dn`.

The headline result is that, in this controlled test case, the posterior covers
the true parameter values. The inverse problem used 79 noisy current
observations and a 41 by 41 posterior grid.

Transition:

First I will explain why current is the right observable for this inverse
problem.

## Slide 2 - Motivation

Timing: 0:50 to 2:00

Script:

The motivation is that a realistic inverse problem should start from what an
experiment can actually measure.

During training and validation, it is useful to look at the full fields:
`cp(x,t)`, `cn(x,t)`, and `phi(x,t)`. Those fields are useful for checking
whether the PINN has learned the PNP dynamics. But they are not the observation
model I want to rely on for inference, because they would be hard to observe
directly throughout the full space-time domain in a real experiment.

So the inverse problem is framed around current `I(t)`. Current is an
electrode-level signal. It is not a local concentration probe, and it does not
require reconstructing the full concentration field. It is also a standard
electrical measurement in electrochemical systems.

In this project the current data are synthetic, but the important point is that
the synthetic observable is chosen to match something that could be measured in
principle.

Transition:

Now I will define the controlled PNP benchmark that produces the synthetic
current data.

## Slide 3 - PNP Benchmark and Assumptions

Timing: 2:00 to 3:20

Script:

The forward model is a nondimensional one-dimensional Poisson-Nernst-Planck
system for a binary electrolyte. The unknown diffusion parameters are `Dp` and
`Dn`, and the model outputs the positive-ion concentration, the negative-ion
concentration, and the electric potential.

The fluxes include both diffusion and electrostatic drift. The concentration
equations are conservation laws, and the potential is determined by the Poisson
equation.

The active benchmark uses `x` in `[0,1]`, `t` in `[0,0.2]`, and parameter values
`Dp`, `Dn` in `[0.5,2.0]`. The true synthetic case shown later is
`Dp = 1.25`, `Dn = 1.25`. The wall voltages are fixed at `phi(0) = -0.5` and
`phi(1) = 0.5`.

A key assumption is the blocking-electrode boundary condition: `Jp = 0` and
`Jn = 0` at the walls. This does not mean there is no measured current. It means
ions do not cross the electrode. The current in this setup is charging current
associated with the electrode and electric field, not ionic flux through the
boundary.

Transition:

With that benchmark fixed, the next step is the trained PINN surrogate.

## Slide 4 - Trained PINN Surrogate

Timing: 3:20 to 4:35

Script:

The PINN is parameterized by both space-time and diffusion parameters. Its input
is `(x, t, Dp, Dn)`, and its output is `(cp, cn, phi)`.

The implementation is in JAX and Equinox. The current trained run uses an
8-layer tanh MLP with width 256, trained for 500,000 Adam steps on the CPU
backend.

Two important constraints are enforced structurally. First, the concentration
initial conditions are exact by construction. Second, the fixed wall voltages
for `phi` are exact by construction. This matters because it removes two
important sources of boundary-condition error from the neural network output.

The diagnostics show that the trained surrogate is not perfect, especially for
the potential field, but it is accurate enough to support the controlled
current-based inverse test. The important amortization idea is that the expensive
training happens once, and then the same trained surrogate can be reused for many
candidate `Dp`, `Dn` values.

Transition:

Next I will explain how the current observations are generated.

## Slide 5 - Synthetic Data Generation

Timing: 4:35 to 5:45

Script:

The current observations are deliberately generated from an independent
benchmark solver, not from the PINN. This is important because it keeps the
inverse test honest. The PINN is being evaluated against data from a separate
numerical model.

The true parameters are set to `Dp = 1.25` and `Dn = 1.25`. The reference solver
then computes the PNP solution. From that simulated PNP state, I compute a clean
charging-current signal. After that, I add independent Gaussian noise with 2%
relative standard deviation.

The final observation used by the inverse problem is therefore a noisy
electrode-current time series. The current diagnostic uses 79 time intervals
after skipping the first transient interval.

The key point is that the training target and the inverse observation are not the
same thing. The PINN produces fields, but the likelihood is built from current.

Transition:

There is one subtle point here: how the current is computed in the synthetic
benchmark.

## Slide 6 - Observable Definition

Timing: 5:45 to 7:00

Script:

This slide is mainly to prevent a possible misunderstanding.

In a real experiment, the instrument measures current `I(t)` directly through
the external circuit. We do not need to measure a concentration field, and we do
not need to measure electrode charge `Q(t)` directly.

In the synthetic benchmark, however, the simulator gives us the PNP state. From
that state we construct `Q_L(t)` using a Gauss-law-based expression, and then
compute an interval-averaged current from finite differences of `Q_L(t)`.

So `Q(t)` is not the measured data. It is the simulation-side route used to
construct the same type of charging-current observable.

This also explains why we do not use boundary ionic flux as the current. The
boundary is blocking, so `Jp = Jn = 0` at the wall. The current signal here is an
electrode charging current, not ions entering or leaving the domain.

The reason for using the `Q(t)` construction in the synthetic/PINN pipeline is
practical: it avoids relying on unstable boundary derivatives while preserving
the electrode-level current meaning.

Transition:

Now that the observation is defined, I can describe the inverse problem.

## Slide 7 - Inverse Problem and Likelihood

Timing: 7:00 to 8:25

Script:

The inverse problem asks: given observed current data, what can we infer about
`Dp` and `Dn`?

For the current diagnostic, I use a simple and transparent grid posterior. The
candidate grid contains 41 by 41 parameter pairs over the parameter box. For
each candidate pair, the trained PINN predicts the current signal. That
prediction is compared to the noisy observed current.

The likelihood assumes independent Gaussian measurement errors. In other words,
for each observed time interval, the difference between the observed current and
the PINN-predicted current is modeled as a Gaussian error term. The standard
deviation is set to 2% of the current scale in this diagnostic.

The posterior is proportional to likelihood times prior. Here the prior over the
grid is uniform, so the posterior shape mainly comes from the current mismatch.
After evaluating all grid points, the probabilities are normalized.

This is intentionally simple. The grid posterior is easy to inspect and explain
for the presentation. Later it can be replaced by MCMC or another sampling
method if the parameter space becomes larger.

Transition:

The first check is whether the MAP point actually matches the observed current.

## Slide 8 - Current Fit at MAP

Timing: 8:25 to 9:45

Script:

This slide compares three current traces: the clean benchmark current, the noisy
synthetic observations, and the PINN-predicted current at the MAP parameter
point.

The important thing to notice is that the PINN prediction follows the observed
current at the level of the measurement noise. This is the direct evidence that
the inverse problem is fitting the actual likelihood signal, not hidden
concentration fields.

There are 79 current observations. The assumed relative Gaussian noise level is
2%. Evaluating the 41 by 41 posterior grid took about 7.3 seconds for this
diagnostic.

The MAP point is the parameter pair with the highest posterior probability. In
this run, the MAP is `Dp = 1.25`, `Dn = 1.2125`. The true value is
`Dp = 1.25`, `Dn = 1.25`, so the MAP is close but not exactly equal to the true
pair. That is acceptable because the data are noisy and the surrogate is not
perfect.

Transition:

The more important result is the full posterior, not just the MAP point.

## Slide 9 - Posterior Result

Timing: 9:45 to 11:25

Script:

This is the main inference result.

The posterior is concentrated near the true parameter pair, but it is not a
single point. It has a tilted shape, which indicates coupled uncertainty between
`Dp` and `Dn`. This is important because current is a compressed observation:
different combinations of `Dp` and `Dn` can produce similar current responses.

The true parameters are `Dp = 1.25`, `Dn = 1.25`. The MAP is `Dp = 1.25`,
`Dn = 1.2125`. The posterior mean is `Dp = 1.23531`, `Dn = 1.23677`.

The 95% marginal interval for `Dp` is `[1.0625, 1.4]`. The 95% marginal interval
for `Dn` is `[1.0625, 1.4375]`. Both intervals cover the true value.

So the correct interpretation is not: the inverse problem has exactly recovered
the true parameters. The correct interpretation is: the current-based inverse
pipeline produces a posterior region that contains the true parameters and makes
the remaining uncertainty visible.

That is the result I would present as usable evidence at this stage: a coherent
proof of concept, not a final calibrated experimental inference system.

Transition:

I will finish by explaining why this is non-trivial and what should come next.

## Slide 10 - Why Non-Trivial and Next Steps

Timing: 11:25 to 13:00

Script:

This problem is non-trivial for several reasons.

First, we do not observe the full PNP state. We observe only electrode current.
Second, current is a compressed signal, so it does not uniquely reveal every
detail of the internal concentration and potential fields. Third, `Dp` and `Dn`
can be statistically correlated in the posterior because they both affect the
same current response. Fourth, the forward model is a coupled PDE system, so
each likelihood evaluation is expensive if we use the benchmark solver directly.
Finally, the output should include uncertainty, not just one best-fit parameter
pair.

The PINN helps because it amortizes the forward model. Training is expensive,
but once trained, the surrogate can be reused for many likelihood evaluations.
That is where the computational advantage becomes meaningful.

The immediate next steps are to repeat the current-based inverse experiment
across more true parameter pairs, test sensitivity to noise level and time
sampling, and move from the simple grid posterior to MCMC if needed. Later, the
model can include richer electrode-interface physics.

The bottom line is that the current pipeline is coherent enough to present: the
assumptions are explicit, the observable is measurable, the reference data are
generated independently, and the posterior result answers the inverse question
with uncertainty.

## Q&A Prep

### What is the reference solver?

The reference solver is independent of the PINN. It uses a finite-difference
method-of-lines discretization, solves the Poisson equation with a banded linear
solve, and integrates the concentration dynamics in time with BDF.

### Why use current instead of concentration?

Concentration fields are useful for validation, but they are not a realistic
primary observation for the inverse problem. Current is an electrode-level signal
that can be measured in an electrochemical circuit.

### Are we measuring `Q(t)` in reality?

No. In a real experiment the measured signal is current `I(t)`. In the synthetic
benchmark, `Q(t)` is only a simulation-side construction used to compute the
same charging-current observable from the PNP state.

### Why is there current if the ion boundary flux is zero?

The boundary condition is a blocking-electrode condition. Ions do not cross the
electrode, so ionic flux through the boundary is zero. But the electrode can
still have charging current because charge redistributes and the electric field
changes.

### Is 2% Gaussian noise physically guaranteed?

No. It is a controlled synthetic noise model for this diagnostic. It is
reasonable as a first test because it is simple and interpretable, but the next
step should include sensitivity tests across different noise levels and possibly
different noise models.

### Is the inverse result already fully reliable?

It is reliable as a controlled proof of concept. It shows that the pipeline is
coherent and the posterior covers the true parameters for the tested case. It is
not yet a final experimental inference model, because it still needs robustness
tests across more true parameter pairs, noise levels, sampling choices, and
physics assumptions.

### Why use a PINN if training is slow?

The advantage is amortization. The benchmark solver may be better for one or a
small number of forward solves. The PINN becomes useful when many forward
evaluations are needed, such as grid posterior evaluation, MCMC, design sweeps,
or repeated inverse problems.

### What does MAP mean here?

MAP means maximum a posteriori estimate: the grid point with the highest
posterior probability. It is a single representative point. It should be read
together with the posterior mean and credible intervals, because the posterior
contains uncertainty and parameter coupling.

### What should be emphasized if time is short?

Emphasize four points: the observable is current, the reference data are
generated independently of the PINN, the likelihood is Gaussian on current
measurements, and the output is a posterior over `Dp` and `Dn`, not only a
single best-fit value.
