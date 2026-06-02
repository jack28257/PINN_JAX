# Presentation Transcript and Speaker Notes

Deck PDF: `pnp_pinn_ssc_2026_presentation.pdf`

Editable source deck: `pnp_pinn_ssc_2026_presentation.pptx`

Venue: Society of Statistics Canada, 2026

Target slot: 15 minutes

Prepared talk length: approximately 13 minutes, followed by Q&A.

## Core Story

This project builds a current-based Bayesian inverse inference pipeline for a
one-dimensional Poisson-Nernst-Planck system. The unknowns are the diffusion
parameters `Dp` and `Dn`. The observable is electrode charging current, not the
hidden concentration or potential fields. A parametric physics-informed neural
network is trained once as a reusable forward surrogate. The trained surrogate
is then used to evaluate a Gaussian current likelihood over parameter space and
form a posterior distribution for `Dp` and `Dn`.

The most important conceptual sentence:

> The PINN is not the inverse estimator by itself; it is a reusable forward map
> that makes many likelihood evaluations cheaper after training.

## Timing Plan

| Slide | Topic | Target time |
| --- | --- | ---: |
| 1 | Title and project overview | 0:35 |
| 2 | PNP motivation | 0:55 |
| 3 | Repeated-solve bottleneck | 0:55 |
| 4 | What a PINN does | 1:00 |
| 5 | Parametric PINN | 1:00 |
| 6 | Forward PNP benchmark | 1:05 |
| 7 | Training setup | 0:55 |
| 8 | Inverse problem setup | 1:00 |
| 9 | Current observable | 1:00 |
| 10 | Likelihood and posterior | 1:10 |
| 11 | Current-fit result | 0:55 |
| 12 | Posterior result | 1:05 |
| 13 | Interpretation | 1:00 |
| 14 | Takeaways | 0:50 |
| 15 | Q&A | hold |

## Slide 1 - Title and Project Overview

Script:

Good morning. My talk is about current-based Bayesian inference for a
one-dimensional Poisson-Nernst-Planck system.

The project asks whether we can infer diffusion parameters from electrode
current. The computational idea is to train a parametric physics-informed neural
network once, use it as a forward surrogate, and then reuse it inside a
Bayesian inverse problem.

The output I want to emphasize is not just a point estimate. It is a posterior
distribution over `Dp` and `Dn`.

Transition:

I will first explain the physical model and why this is a repeated-forward-solve
problem.

## Slide 2 - PNP Motivation

Script:

The Poisson-Nernst-Planck system describes the interaction between charged
particle concentrations and an electric field.

In this project there are two concentration fields: positive ions and negative
ions. The electric potential couples them, so the transport of one species is
not independent of the other. The diffusion parameters `Dp` and `Dn` control how
the two species move.

For this presentation, the important point is not the full electrochemistry.
The important point is that this is a coupled PDE model with physically
meaningful parameters.

Transition:

The inverse problem becomes expensive because those parameters have to be
evaluated repeatedly.

## Slide 3 - Repeated-Solve Bottleneck

Script:

A conventional numerical solver is reliable, but it solves one forward problem
at a time. If I want to evaluate one candidate pair of diffusion parameters, I
run a PDE solve and compute the predicted current.

But Bayesian inference is a many-query setting. A grid posterior or an MCMC
algorithm may require many forward evaluations.

This is where amortization matters. Training the PINN is expensive once. The
benefit appears when we reuse the trained forward map for many candidate
parameter pairs.

Transition:

So the next step is to define what the PINN is actually learning.

## Slide 4 - What a PINN Does

Script:

A physics-informed neural network represents a solution function with a neural
network. Instead of fitting a large table of labeled solutions, it is trained so
that the output satisfies the governing equations and the initial or boundary
conditions.

At sampled collocation points, the PDE residual should be close to zero. The
training loss combines the PDE residual, the initial and boundary residuals, and
in our case coverage over the parameter domain.

This is why the method is physics-informed: the differential equation is part
of the training objective.

Transition:

For inverse inference, we need a parametric version of this idea.

## Slide 5 - Parametric PINN

Script:

A standard PINN usually represents one solution for one PDE setting. Here the
network input also includes `Dp` and `Dn`.

The trained model takes `(x, t, Dp, Dn)` as input and returns the state
variables `(cp, cn, phi)`. That means one trained network represents a family of
solutions across the parameter box.

This is what makes the inverse step possible. For any candidate parameter pair,
we can evaluate the forward state, compute the corresponding current, and
compare it with observations.

Transition:

Now I will describe the particular PNP benchmark used in this project.

## Slide 6 - Forward PNP Benchmark

Script:

The benchmark is nondimensional, one-dimensional, and controlled.

The Nernst-Planck equations describe the transport of positive and negative
ions. The Poisson equation links the electric potential to the charge density.
The active parameter range is `Dp, Dn` in `[0.5, 2.0]`.

The boundaries are fixed-voltage blocking electrodes. This means ions do not
cross the electrode boundary. However, charging current can still be measured in
the external circuit.

This is an idealized setup. It is useful because it makes the current-based
inverse experiment clear and reproducible.

Transition:

With the forward model fixed, the network can be trained as a surrogate.

## Slide 7 - Training Setup

Script:

Training builds the surrogate. It does not use the finite-difference reference
solver as a pointwise label generator.

The loss enforces three things: PDE residuals, initial and boundary constraints,
and coverage over the diffusion-parameter domain.

The completed run uses an 8-layer tanh MLP with width 256 in JAX and Equinox.
It was trained for 500,000 Adam steps. The reference solver is still important,
but its role is validation and synthetic observation generation.

Transition:

Now I will show how the trained surrogate is used in the inverse problem.

## Slide 8 - Inverse Problem Setup

Script:

The inverse problem is posed around current, not around hidden concentration
fields.

For the current diagnostic, the true parameters are `Dp = 1.25` and
`Dn = 1.25`. The independent reference solver generates a clean current time
series. Then 2% Gaussian noise is added, and the likelihood uses 79 current
intervals.

For each candidate `Dp, Dn`, the trained PINN predicts the state and then the
current observable. The posterior is formed by comparing predicted current with
observed current.

Transition:

Before writing the likelihood, I need to clarify what current means in this
blocking-electrode benchmark.

## Slide 9 - Current Observable

Script:

Experimentally, the measured quantity would be current `I(t)` through an
external circuit.

In the synthetic benchmark, the simulator gives the full PNP state. From that
state, we compute an electrode charge response `Q(t)` and then take finite
differences to get interval-averaged current.

This is not saying that `Q(t)` is directly measured in the lab. It is a stable
way to compute the current from the simulated or learned field, especially
because it avoids differentiating the learned potential exactly at the wall.

Transition:

Once current is defined, the statistical model is straightforward.

## Slide 10 - Likelihood and Posterior

Script:

The observation model assumes independent Gaussian errors on the current
measurements.

For each candidate diffusion pair, the PINN predicts the current. The residual
between predicted current and observed current is scaled by the noise standard
deviation. The Gaussian likelihood is proportional to the exponential of minus
one half of the summed squared standardized residuals.

For the current result, the prior is uniform over the parameter box, and the
posterior is normalized on a 41 by 41 grid.

Transition:

The next two slides show the result: first the current fit, then the posterior.

## Slide 11 - Current-Fit Result

Script:

At the MAP point, the PINN-predicted current follows the noisy current
observations closely.

The black curve is the clean reference current. The orange points are noisy
observations. The teal line is the current predicted from the PINN at the MAP
parameter pair.

The MAP is `Dp = 1.25`, `Dn = 1.2125`. This is a good current fit, but the fit
alone is not the final inferential answer.

Transition:

The posterior distribution gives the uncertainty and the coupling between the
two parameters.

## Slide 12 - Posterior Result

Script:

The posterior covers the true diffusion parameters and shows coupled
uncertainty.

The true value is `Dp = 1.25`, `Dn = 1.25`. The posterior mean is approximately
`Dp = 1.2353`, `Dn = 1.2368`. The 95% marginal interval for `Dp` is
`[1.0625, 1.4000]`, and for `Dn` it is `[1.0625, 1.4375]`.

The important visual feature is the elongated posterior region. It shows that
the current signal identifies a correlated combination of the two diffusion
parameters, not two completely independent quantities.

Transition:

That posterior shape is why this is not just a curve-fitting exercise.

## Slide 13 - Interpretation

Script:

This problem is indirect. The observed data are current only. The concentration
fields and electric potential are predicted by the model, but they are not
assumed to be observed.

The two diffusion parameters affect current through a coupled nonlinear PDE.
That is why the posterior is correlated. There is also surrogate error, because
the forward map is approximated by the trained PINN rather than solved exactly
for each candidate pair.

So the result should be read as a controlled proof of concept. The meaningful
claim is that current-only synthetic data can produce a reasonable posterior
region in this benchmark.

Transition:

I will close with the takeaways and the next steps.

## Slide 14 - Takeaways

Script:

There are three takeaways.

First, the current-based inverse pipeline is working in this controlled
benchmark. Current-only synthetic data recover a posterior region that contains
the true diffusion parameters.

Second, the PINN acts as an amortized forward surrogate. It is not directly the
inverse estimator; it provides repeated current predictions for the likelihood.

Third, the next steps are to stress-test more parameter pairs, noise levels, and
time sampling choices, and then replace the grid posterior with MCMC when the
parameter space becomes larger.

Transition:

That concludes the prepared part of the talk.

## Slide 15 - Q&A

Script:

Thank you. Any questions?
