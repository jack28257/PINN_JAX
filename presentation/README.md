# Presentation Materials

This folder contains the current presentation package for the 1D PNP PINN
current-based inverse inference project.

## Files

```text
pnp_pinn_current_inference_presentation.pptx  Editable PowerPoint deck
transcript.md                                Slide-by-slide speaking notes
```

## Talk Length

The deck is designed for a 15-minute presentation slot, with the prepared talk
timed to approximately 13 minutes so there is room for questions.

## Main Thesis

The project is no longer just about training a PINN. The current result is a
current-based inverse-inference pipeline: an independent PNP benchmark solver
generates synthetic electrode-current observations, and a trained PINN surrogate
is reused to infer a posterior distribution over `Dp` and `Dn`.
