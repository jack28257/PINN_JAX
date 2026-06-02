# Presentation Transcript and Speaker Notes

Deck: `pnp_pinn_current_inference_presentation.pptx`

Format: academic presentation

Target slot: 15 minutes

Prepared talk length: approximately 13 minutes, plus the final Q&A slide

## Main Story

This project studies a controlled inverse problem for a one-dimensional
Poisson-Nernst-Planck system. The unknown parameters are the two diffusion
coefficients `Dp` and `Dn`. The observation is electrode current, not the hidden
concentration or potential fields. A physics-informed neural network is trained
as a parameterized forward surrogate. The trained surrogate is then reused to
evaluate a current-based Gaussian likelihood and form a posterior distribution
over `Dp` and `Dn`.

The most important conceptual sentence:

> The PINN is not directly estimating `Dp` and `Dn`; it learns the forward PNP
> solution map for many possible diffusion-parameter values, and the inverse
> problem uses that map to compare predicted current with observed current.

## Timing Plan

| Slide | Topic | Target time |
| --- | --- | ---: |
| 1 | Formal opening | 0:25 |
| 2 | Research question and storyline | 0:45 |
| 3 | What is observed | 0:55 |
| 4 | Forward PNP model | 1:05 |
| 5 | What the PINN learns | 1:05 |
| 6 | PINN training objective | 1:00 |
| 7 | Why amortization matters | 0:50 |
| 8 | Synthetic data generation | 0:55 |
| 9 | Current observable and Q(t) | 1:05 |
| 10 | Likelihood and posterior | 1:10 |
| 11 | Current fit | 1:00 |
| 12 | Posterior result | 1:20 |
| 13 | Interpretation and next steps | 1:15 |
| 14 | Any questions | hold |

Total prepared talk: about 13 minutes.

## Slide 1 - Formal Opening

Timing: 0:00 to 0:25

Script:

Good morning. Today I will present a current-based inverse inference pipeline
for a one-dimensional Poisson-Nernst-Planck system.

The goal is to infer diffusion parameters from electrode current. The method
uses a physics-informed neural network as a reusable forward surrogate, and then
uses that surrogate inside a Bayesian inverse problem.

Transition:

I will first state the central question and the structure of the talk.

## Slide 2 - Research Question and Storyline

Timing: 0:25 to 1:10

Script:

The central question is whether electrode current can identify the diffusion
parameters `Dp` and `Dn`.

The storyline is four steps. First, define the measurable signal: current.
Second, define the forward PNP physics. Third, train a PINN to approximate the
forward map over the parameter space. Finally, use the trained forward model to
build a posterior distribution over the unknown parameters.

This is important because the inverse problem should be posed around something
that would be observable in an experiment, not around hidden simulation fields.

Transition:

So I will start with the observation model.

## Slide 3 - What Is Observed

Timing: 1:10 to 2:05

Script:

During development, it is useful to inspect the full fields:
`cp(x,t)`, `cn(x,t)`, and `phi(x,t)`. But those are hidden state variables.
They are not the measurement model used for inference here.

The inverse problem uses electrode current `I(t)`. The data are noisy current
observations, and the output is a posterior distribution over `Dp` and `Dn`
conditioned on those observations.

This distinction matters because the PINN predicts fields, but the likelihood
is built from current. The field prediction is only an intermediate object.

Transition:

Next I will define the forward PNP system that produces these fields.

## Slide 4 - Forward PNP Model

Timing: 2:05 to 3:10

Script:

The benchmark is a nondimensional one-dimensional Poisson-Nernst-Planck system.
There are two ionic concentrations, `cp` and `cn`, and an electric potential
`phi`.

The fluxes contain diffusion and drift terms. The concentration equations are
conservation laws, and the potential satisfies the Poisson equation. The
unknown parameters are the two diffusion coefficients `Dp` and `Dn`.

The active setup uses `x` in `[0,1]`, `t` in `[0,0.2]`, and `Dp`, `Dn` in
`[0.5,2.0]`. The wall potentials are fixed, and the ion fluxes at the walls are
zero.

The no-flux condition is a blocking-electrode assumption. It means ions do not
cross the electrode. It does not mean the measured current is zero, because the
current considered here is charging current associated with the electrode and
electric field.

Transition:

With the forward model fixed, the next question is what the PINN actually
learns.

## Slide 5 - What the PINN Learns

Timing: 3:10 to 4:15

Script:

This is the key conceptual slide for the PINN.

The PINN is not trained to directly output `Dp` and `Dn`. Instead, it learns a
parameterized forward solution map. Its input is `(x, t, Dp, Dn)`, and its
output is `(cp, cn, phi)`.

This means one trained neural network represents a family of PNP solutions over
the parameter domain. If we give the network a candidate diffusion-parameter
pair, it returns the predicted state at the requested space-time point.

The inverse problem then uses those predicted fields to compute the predicted
current for that candidate parameter pair.

Transition:

Now I will explain how this forward surrogate is trained.

## Slide 6 - PINN Training Objective

Timing: 4:15 to 5:15

Script:

The PINN is trained by enforcing the physics of the PNP system, rather than by
supervised fitting to reference solution labels.

The loss has three roles. The PDE residual term enforces the differential
equations. The initial and boundary terms enforce the physical constraints. The
parameter-domain sampling term ensures that the network is trained across
candidate values of `Dp` and `Dn`, not only at one parameter pair.

In the current completed run, the model was trained for 500,000 Adam steps.
Reference solutions are still important, but their role is different: they are
used for validation and synthetic observations, not as pointwise PINN training
labels.

Transition:

This leads to the computational reason for using a PINN.

## Slide 7 - Why Amortization Matters

Timing: 5:15 to 6:05

Script:

The claim is not that training a PINN is faster than one benchmark solve.
Training is expensive.

The computational argument is amortization. If the inverse problem is solved
directly with the benchmark model, then each candidate `Dp`, `Dn` pair requires
a forward numerical PDE solve. A posterior grid or an MCMC chain may require
many such evaluations.

After the PINN is trained, the forward model is a neural-network evaluation.
The benefit appears when the same trained surrogate is reused many times.

Transition:

Now I will show how the synthetic current observations are generated.

## Slide 8 - Synthetic Data Generation

Timing: 6:05 to 7:00

Script:

The synthetic data are generated independently of the PINN.

The true parameter pair is `Dp = 1.25`, `Dn = 1.25`. An independent
finite-difference reference solver with BDF time integration generates the PNP
solution. From that reference solution, we compute the clean electrode current.
Then we add 2% Gaussian noise.

After skipping the first transient interval, the inverse problem uses 79 current
observations.

This separation is important: the data source is the independent benchmark
solver, while the model being tested in the inverse problem is the trained PINN
surrogate.

Transition:

Before forming the likelihood, I need to clarify how current is computed in the
blocking-electrode setup.

## Slide 9 - Current Observable and Q(t)

Timing: 7:00 to 8:05

Script:

In a real experiment, the observable would be current `I(t)` measured through an
external circuit.

In the synthetic benchmark, the simulator gives us the PNP state. From that
state, we construct the electrode charge response `Q_L(t)` and take finite
differences to obtain interval-averaged current.

The important point is that `Q(t)` is not being presented as the measured lab
quantity. It is the simulation-side construction used to compute the same
current observable from the PNP state.

This route also avoids unstable boundary derivatives when evaluating current
from the learned PINN field. It respects the blocking boundary condition,
because the current here is not boundary ionic flux.

Transition:

The next slide shows the likelihood model used to turn those current
observations into a posterior.

## Slide 10 - Likelihood and Posterior

Timing: 8:05 to 9:15

Script:

For each candidate pair `Dp`, `Dn`, the PINN predicts a current time series.
We compare that prediction to the observed current.

The noise model is independent Gaussian noise. The residual is the difference
between the PINN-predicted current and the observed current, normalized by the
noise standard deviation.

The likelihood is therefore proportional to the exponential of minus one half
times the sum of squared standardized residuals. In this diagnostic, the prior
is uniform over a 41 by 41 parameter grid, so the posterior is obtained by
evaluating and normalizing this likelihood over the grid.

The output is a distribution over `Dp` and `Dn`, not just a single fitted
parameter pair.

Transition:

The first result is the current fit at the MAP point.

## Slide 11 - Current Fit

Timing: 9:15 to 10:15

Script:

This plot compares three signals: the reference clean current, the noisy
current observations, and the PINN-predicted current at the MAP parameter pair.

The MAP estimate is `Dp = 1.25`, `Dn = 1.2125`. The PINN current at this point
matches the observed current at approximately the noise level.

This is an important check because the inverse problem is not evaluated on
hidden concentration fields. It is evaluated on the same current signal that
appears in the likelihood.

Transition:

The current fit is useful, but the posterior is the main inverse result.

## Slide 12 - Posterior Result

Timing: 10:15 to 11:35

Script:

The posterior is concentrated near the true parameter values. The true pair is
`Dp = 1.25`, `Dn = 1.25`, and the posterior mean is approximately
`Dp = 1.2353`, `Dn = 1.2368`.

The 95% marginal interval for `Dp` is `[1.0625, 1.4000]`, and for `Dn` it is
`[1.0625, 1.4375]`. Both intervals include the true values.

The shape of the posterior is also informative. It is not a circular
independent uncertainty region. It is tilted, which means the current signal
couples the two diffusion parameters. Several nearby combinations of `Dp` and
`Dn` can explain the current almost equally well.

So the result should be interpreted as a posterior uncertainty region, not as
exact deterministic recovery.

Transition:

I will close with what this result shows and what still needs to be tested.

## Slide 13 - Interpretation and Next Steps

Timing: 11:35 to 12:50

Script:

The current result is a controlled proof of concept for PINN-based inverse
inference.

It shows three things. First, the observation model is current-based and
experimentally meaningful. Second, the PINN is used as a reusable forward
surrogate, not as the source of the synthetic data. Third, the inference result
is a posterior distribution over `Dp` and `Dn`, so the uncertainty is visible.

The next steps are to repeat the experiment across more true parameter pairs,
test sensitivity to the noise level and time sampling, and eventually replace
the grid posterior with MCMC if the parameter space becomes larger.

The main conclusion is that the pipeline is now coherent: measurable current
data, an independent reference data source, a trained PINN forward surrogate,
and a posterior answer to the inverse problem.

Transition:

Thank you.

## Slide 14 - Any Questions

Timing: hold for Q&A

Script:

Any questions?

## Short Q&A Prep

### What is the PINN doing?

It approximates the parameterized forward PNP solution map:
`(x,t,Dp,Dn) -> (cp,cn,phi)`. It does not directly output the inferred
parameters.

### What is the reference solver?

An independent finite-difference method-of-lines solver with BDF time
integration. It is used for validation and synthetic data generation.

### What is actually measured?

The realistic observable is current `I(t)`. In the synthetic benchmark, `Q(t)`
is only used internally to compute the same current from the simulated PNP
state.

### Why can there be current with no ion flux through the boundary?

The boundary is blocking, so ions do not cross the electrode. The current here
is electrode charging current, not boundary ionic flux.

### Why use a PINN if training is expensive?

Because the trained PINN can be reused for many forward evaluations. That is
useful for posterior grids, MCMC, parameter sweeps, and repeated inverse
experiments.

### Is the result final?

No. It is a controlled proof of concept. The next step is robustness testing
across parameter pairs, noise levels, time sampling choices, and richer physics.
