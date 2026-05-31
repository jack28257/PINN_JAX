"""Equinox model and pointwise physics quantities for the 1D PNP PINN."""

from __future__ import annotations

from typing import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp

from .config import PNPConfig
from .types import Array, FirstDerivatives, SecondDerivatives, Theta


def theta_to_pair(theta: Theta) -> tuple[Array, Array]:
    if isinstance(theta, Mapping):
        return theta["Dp"], theta["Dn"]
    return theta[0], theta[1]


def fluxes_from_derivatives(cfg: PNPConfig, theta: Theta, u: Array, ux: Array) -> Array:
    """Assemble [J_p, J_n] from u and first spatial derivatives."""

    dp, dn = theta_to_pair(theta)
    cp, cn, _ = u
    cp_x, cn_x, phi_x = ux
    flux_p = -dp * cp_x - dp * cfg.zp * cp * phi_x
    flux_n = -dn * cn_x - dn * cfg.zn * cn * phi_x
    return jnp.stack([flux_p, flux_n])


def flux_x_from_derivatives(cfg: PNPConfig, theta: Theta, u: Array, ux: Array, uxx: Array) -> Array:
    """Assemble dJ/dx without differentiating through fluxes again."""

    dp, dn = theta_to_pair(theta)
    cp, cn, _ = u
    cp_x, cn_x, phi_x = ux
    cp_xx, cn_xx, phi_xx = uxx

    flux_p_x = -dp * cp_xx - dp * cfg.zp * (cp_x * phi_x + cp * phi_xx)
    flux_n_x = -dn * cn_xx - dn * cfg.zn * (cn_x * phi_x + cn * phi_xx)
    return jnp.stack([flux_p_x, flux_n_x])


class PNP(eqx.Module):
    """Parametric PINN for u(x, t, theta) = (cp, cn, phi)."""

    learner: eqx.nn.MLP
    cfg: PNPConfig = eqx.field(static=True)

    def __init__(self, cfg: PNPConfig, key: Array):
        self.cfg = cfg
        self.learner = eqx.nn.MLP(
            in_size=4,
            out_size=3,
            width_size=cfg.width_size,
            depth=cfg.depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def _features(self, x: Array, t: Array, theta: Theta) -> Array:
        dp, dn = theta_to_pair(theta)

        if self.cfg.normalize_inputs:
            x = 2.0 * (x - self.cfg.x_min) / (self.cfg.x_max - self.cfg.x_min) - 1.0
            t = 2.0 * t / self.cfg.t_max - 1.0
            dp = 2.0 * (dp - self.cfg.dp_min) / (self.cfg.dp_max - self.cfg.dp_min) - 1.0
            dn = 2.0 * (dn - self.cfg.dn_min) / (self.cfg.dn_max - self.cfg.dn_min) - 1.0

        return jnp.stack([jnp.asarray(x), jnp.asarray(t), jnp.asarray(dp), jnp.asarray(dn)])

    def _wall_coordinate(self, x: Array) -> Array:
        return (x - self.cfg.x_min) / (self.cfg.x_max - self.cfg.x_min)

    def _wall_phi(self, x: Array) -> Array:
        xi = self._wall_coordinate(x)
        return (1.0 - xi) * self.cfg.v_left + xi * self.cfg.v_right

    def _hard_phi(self, x: Array, raw_phi: Array) -> Array:
        xi = self._wall_coordinate(x)
        interior_envelope = 4.0 * xi * (1.0 - xi)
        return self._wall_phi(x) + interior_envelope * raw_phi

    def __call__(self, x: Array, t: Array, theta: Theta) -> Array:
        raw_cp, raw_cn, raw_phi = self.learner(self._features(x, t, theta))

        if self.cfg.hard_ic:
            cp = self.cfg.cp_init + t * raw_cp
            cn = self.cfg.cn_init + t * raw_cn
        else:
            cp = raw_cp
            cn = raw_cn
        phi = self._hard_phi(x, raw_phi) if self.cfg.hard_phi_bc else raw_phi

        return jnp.stack([cp, cn, phi])

    def eval_with_first_derivatives(self, x: Array, t: Array, theta: Theta) -> FirstDerivatives:
        """Return u, du/dx, and du/dt for one point.

        This avoids separate grad calls for cp_x, cn_x, phi_x, cp_t, and cn_t.
        """

        xt = jnp.stack([jnp.asarray(x), jnp.asarray(t)])

        def u_of_xt(local_xt: Array) -> Array:
            return self(local_xt[0], local_xt[1], theta)

        u = u_of_xt(xt)
        jac = jax.jacfwd(u_of_xt)(xt)
        return FirstDerivatives(u=u, ux=jac[:, 0], ut=jac[:, 1])

    def eval_with_derivatives(self, x: Array, t: Array, theta: Theta) -> SecondDerivatives:
        """Return u, first derivatives, and second x derivatives for one point.

        The PNP residuals need cp_t, cn_t, cp_x, cn_x, phi_x, and x-second
        derivatives. Computing these through one shared derivative kernel avoids
        differentiating through fluxes and cuts out repeated network evaluations.
        """

        xt = jnp.stack([jnp.asarray(x), jnp.asarray(t)])

        def u_of_xt(local_xt: Array) -> Array:
            return self(local_xt[0], local_xt[1], theta)

        u = u_of_xt(xt)
        jac = jax.jacfwd(u_of_xt)(xt)
        hess = jax.jacfwd(jax.jacfwd(u_of_xt))(xt)
        return SecondDerivatives(u=u, ux=jac[:, 0], ut=jac[:, 1], uxx=hess[:, 0, 0])

    def fluxes(self, x: Array, t: Array, theta: Theta) -> Array:
        derivs = self.eval_with_first_derivatives(x, t, theta)
        return fluxes_from_derivatives(self.cfg, theta, derivs.u, derivs.ux)

    def current(self, x: Array, t: Array, theta: Theta) -> Array:
        flux_p, flux_n = self.fluxes(x, t, theta)
        return self.cfg.zp * flux_p + self.cfg.zn * flux_n

    def dom_resid(self, x: Array, t: Array, theta: Theta) -> Array:
        derivs = self.eval_with_derivatives(x, t, theta)
        cp, cn, _ = derivs.u
        cp_t, cn_t, _ = derivs.ut
        flux_p_x, flux_n_x = flux_x_from_derivatives(
            self.cfg,
            theta,
            derivs.u,
            derivs.ux,
            derivs.uxx,
        )
        phi_xx = derivs.uxx[2]

        charge_density = self.cfg.zp * cp + self.cfg.zn * cn
        nernst_planck_p = cp_t + flux_p_x
        nernst_planck_n = cn_t + flux_n_x
        poisson = -self.cfg.epsilon * phi_xx - charge_density
        return jnp.stack([nernst_planck_p, nernst_planck_n, poisson])

    def bc_resid(self, x: Array, t: Array, theta: Theta) -> Array:
        derivs = self.eval_with_first_derivatives(x, t, theta)
        flux_p, flux_n = fluxes_from_derivatives(self.cfg, theta, derivs.u, derivs.ux)
        phi = derivs.u[2]
        if self.cfg.hard_phi_bc:
            wall_phi = self._wall_phi(x)
        else:
            wall_phi = jnp.where(
                x <= 0.5 * (self.cfg.x_min + self.cfg.x_max),
                self.cfg.v_left,
                self.cfg.v_right,
            )
        return jnp.stack([flux_p, flux_n, phi - wall_phi])

    def ic_resid(self, x: Array, t: Array, theta: Theta) -> Array:
        cp, cn, _ = self(x, t, theta)
        return jnp.stack([cp - self.cfg.cp_init, cn - self.cfg.cn_init])
