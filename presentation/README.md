# Presentation Materials

This folder contains the current academic presentation package for the 1D PNP
PINN current-based inverse inference project.

## Files

```text
pnp_pinn_ssc_2026_presentation.pdf   Final presentation PDF
pnp_pinn_ssc_2026_presentation.pptx  Editable source deck
transcript.md                         Slide-by-slide speaking notes
```

## Talk Length

The deck has 15 slides and is designed for a 15-minute presentation slot. The
prepared talk is timed to approximately 13 minutes, followed by a final Q&A
slide.

## Main Thesis

The project uses a trained PINN as a parameterized forward surrogate for the 1D
PNP system. An independent benchmark solver generates synthetic electrode-current
observations, and the trained PINN is reused to infer a posterior distribution
over `Dp` and `Dn`.
