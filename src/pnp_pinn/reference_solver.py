"""Finite-difference reference solver for the 1D PNP system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded

from .config import PNPConfig


@dataclass(frozen=True)
class ReferenceSolution:
    x: np.ndarray
    t: np.ndarray
    cp: np.ndarray
    cn: np.ndarray
    phi: np.ndarray
    dp: float
    dn: float
    nx: int
    method: str
    rtol: float
    atol: float
    success: bool
    message: str
    nfev: int


def _poisson_banded_matrix(nx: int, cfg: PNPConfig) -> np.ndarray:
    dx = (cfg.x_max - cfg.x_min) / (nx - 1)
    m = nx - 2
    ab = np.zeros((3, m), dtype=np.float64)
    main = 2.0 * cfg.epsilon / dx**2
    off = -cfg.epsilon / dx**2
    ab[0, 1:] = off
    ab[1, :] = main
    ab[2, :-1] = off
    return ab


def solve_phi_from_charge(cp: np.ndarray, cn: np.ndarray, cfg: PNPConfig, ab: np.ndarray) -> np.ndarray:
    """Solve -epsilon phi_xx = zp*cp + zn*cn with Dirichlet wall voltages."""

    nx = cp.size
    if nx < 3:
        raise ValueError("nx must be at least 3")

    dx = (cfg.x_max - cfg.x_min) / (nx - 1)
    charge = cfg.zp * cp + cfg.zn * cn
    rhs = charge[1:-1].astype(np.float64).copy()
    wall_scale = cfg.epsilon / dx**2
    rhs[0] += wall_scale * cfg.v_left
    rhs[-1] += wall_scale * cfg.v_right

    phi = np.empty(nx, dtype=np.float64)
    phi[0] = cfg.v_left
    phi[-1] = cfg.v_right
    phi[1:-1] = solve_banded((1, 1), ab, rhs)
    return phi


def _rhs_factory(dp: float, dn: float, cfg: PNPConfig, nx: int):
    dx = (cfg.x_max - cfg.x_min) / (nx - 1)
    ab = _poisson_banded_matrix(nx, cfg)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        cp = y[:nx]
        cn = y[nx:]
        phi = solve_phi_from_charge(cp, cn, cfg, ab)

        cp_face = 0.5 * (cp[:-1] + cp[1:])
        cn_face = 0.5 * (cn[:-1] + cn[1:])
        cp_x_face = (cp[1:] - cp[:-1]) / dx
        cn_x_face = (cn[1:] - cn[:-1]) / dx
        phi_x_face = (phi[1:] - phi[:-1]) / dx

        jp_internal = -dp * cp_x_face - dp * cfg.zp * cp_face * phi_x_face
        jn_internal = -dn * cn_x_face - dn * cfg.zn * cn_face * phi_x_face

        jp_faces = np.zeros(nx + 1, dtype=np.float64)
        jn_faces = np.zeros(nx + 1, dtype=np.float64)
        jp_faces[1:-1] = jp_internal
        jn_faces[1:-1] = jn_internal

        dcp_dt = -(jp_faces[1:] - jp_faces[:-1]) / dx
        dcn_dt = -(jn_faces[1:] - jn_faces[:-1]) / dx
        return np.concatenate([dcp_dt, dcn_dt])

    return rhs


def solve_reference_pnp(
    cfg: PNPConfig,
    *,
    dp: float,
    dn: float,
    nx: int = 201,
    nt: int = 41,
    method: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> ReferenceSolution:
    """Solve the 1D PNP system with finite differences and a stiff ODE solver."""

    x = np.linspace(cfg.x_min, cfg.x_max, nx, dtype=np.float64)
    t_eval = np.linspace(0.0, cfg.t_max, nt, dtype=np.float64)
    y0 = np.concatenate(
        [
            np.full(nx, cfg.cp_init, dtype=np.float64),
            np.full(nx, cfg.cn_init, dtype=np.float64),
        ]
    )
    rhs = _rhs_factory(dp, dn, cfg, nx)
    result = solve_ivp(
        rhs,
        (0.0, cfg.t_max),
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    cp = result.y[:nx, :].T
    cn = result.y[nx:, :].T
    ab = _poisson_banded_matrix(nx, cfg)
    phi = np.stack([solve_phi_from_charge(cp_i, cn_i, cfg, ab) for cp_i, cn_i in zip(cp, cn)])

    return ReferenceSolution(
        x=x,
        t=t_eval,
        cp=cp,
        cn=cn,
        phi=phi,
        dp=float(dp),
        dn=float(dn),
        nx=nx,
        method=method,
        rtol=rtol,
        atol=atol,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
    )


def interpolate_solution_to_x(solution: ReferenceSolution, x_target: np.ndarray) -> dict[str, np.ndarray]:
    """Interpolate a reference solution onto a different spatial grid."""

    return {
        "cp": np.stack([np.interp(x_target, solution.x, row) for row in solution.cp]),
        "cn": np.stack([np.interp(x_target, solution.x, row) for row in solution.cn]),
        "phi": np.stack([np.interp(x_target, solution.x, row) for row in solution.phi]),
    }


def relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    diff = np.asarray(a) - np.asarray(b)
    denom = max(float(np.sqrt(np.mean(np.asarray(b) ** 2))), eps)
    return float(np.sqrt(np.mean(diff**2)) / denom)


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def compare_reference_solutions(coarse: ReferenceSolution, fine: ReferenceSolution) -> dict[str, float]:
    """Compare a fine reference solution against a coarser one on the coarse grid."""

    fine_on_coarse = interpolate_solution_to_x(fine, coarse.x)
    return {
        "cp_relative_l2": relative_l2(coarse.cp, fine_on_coarse["cp"]),
        "cn_relative_l2": relative_l2(coarse.cn, fine_on_coarse["cn"]),
        "phi_relative_l2": relative_l2(coarse.phi, fine_on_coarse["phi"]),
        "cp_max_abs": max_abs_error(coarse.cp, fine_on_coarse["cp"]),
        "cn_max_abs": max_abs_error(coarse.cn, fine_on_coarse["cn"]),
        "phi_max_abs": max_abs_error(coarse.phi, fine_on_coarse["phi"]),
    }


def save_reference_npz(path: str | Path, solution: ReferenceSolution) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=solution.x,
        t=solution.t,
        cp=solution.cp,
        cn=solution.cn,
        phi=solution.phi,
        dp=solution.dp,
        dn=solution.dn,
        nx=solution.nx,
        method=solution.method,
        rtol=solution.rtol,
        atol=solution.atol,
        success=solution.success,
        message=solution.message,
        nfev=solution.nfev,
    )


def parse_parameter_cases(cases: str | Iterable[str]) -> list[tuple[float, float]]:
    if isinstance(cases, str):
        items = [item.strip() for item in cases.split(";") if item.strip()]
    else:
        items = list(cases)
    parsed = []
    for item in items:
        left, right = item.split(",")
        parsed.append((float(left), float(right)))
    return parsed
