"""Shared type aliases and small PyTree containers for the PNP PINN."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax


Array = jax.Array
Theta = Mapping[str, Array] | Array


class ConstraintBatch(NamedTuple):
    x: Array
    t: Array
    theta: dict[str, Array]


class TrainingBatch(NamedTuple):
    dom: ConstraintBatch
    bl: ConstraintBatch
    bc: ConstraintBatch


class GroupWeights(NamedTuple):
    dom: Array
    bl: Array
    bc: Array


class ResidualWeights(NamedTuple):
    dom: Array
    bl: Array
    bc: Array


class FirstDerivatives(NamedTuple):
    u: Array
    ux: Array
    ut: Array


class SecondDerivatives(NamedTuple):
    u: Array
    ux: Array
    ut: Array
    uxx: Array


class TrainingState(NamedTuple):
    model: Any
    opt_state: Any
    key: Array
    group_weights: GroupWeights
    step: Array
