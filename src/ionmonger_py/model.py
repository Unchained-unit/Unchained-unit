from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

from .config import SimulationConfig
from .generation import BeerLambertGeneration
from .mesh import DeviceMesh
from .poisson import solve_poisson
from .recombination import RecombinationModel
from .stats import build_statistical_model


@dataclass
class StateSlices:
    n: slice
    p: slice
    p_ion: slice
    n_etl: slice
    p_htl: slice


def make_slices(mesh: DeviceMesh) -> StateSlices:
    ncell = len(mesh.x)
    perov_n = len(mesh.perov_idx)
    etl_n = len(mesh.etl_idx)
    htl_n = len(mesh.htl_idx)
    i = 0
    n = slice(i, i + ncell)
    i += ncell
    p = slice(i, i + ncell)
    i += ncell
    p_ion = slice(i, i + perov_n)
    i += perov_n
    n_etl = slice(i, i + etl_n)
    i += etl_n
    p_htl = slice(i, i + htl_n)
    return StateSlices(n=n, p=p, p_ion=p_ion, n_etl=n_etl, p_htl=p_htl)


def default_initial_state(mesh: DeviceMesh) -> np.ndarray:
    s = make_slices(mesh)
    y = np.zeros(s.p_htl.stop)
    y[s.n] = 1e20
    y[s.p] = 1e20
    y[s.p_ion] = 1e24
    y[s.n_etl] = 1e24
    y[s.p_htl] = 1e24
    return y


def _grad_center(u: np.ndarray, dx: np.ndarray) -> np.ndarray:
    g = np.zeros_like(u)
    g[1:-1] = (u[2:] - u[:-2]) / (dx[1:] + dx[:-1])
    g[0] = (u[1] - u[0]) / dx[0]
    g[-1] = (u[-1] - u[-2]) / dx[-1]
    return g


def _div_flux(f: np.ndarray, dx: np.ndarray) -> np.ndarray:
    out = np.zeros_like(f)
    out[1:-1] = (f[1:-1] - f[:-2]) / dx[1:-1]
    out[0] = (f[0]) / dx[0]
    out[-1] = (-f[-2]) / dx[-1]
    return out


def ion_flux(cfg: SimulationConfig, p_ion: np.ndarray, dphidx: np.ndarray) -> np.ndarray:
    di = cfg.physics.di
    vt = cfg.physics.vt
    if cfg.physics.steric_mode == "none":
        return -di * (_grad_center(p_ion, np.ones_like(p_ion)) + (p_ion / vt) * dphidx)
    if cfg.physics.steric_mode == "drift":
        factor = np.maximum(1.0 - p_ion / max(cfg.physics.plim, 1e-30), 1e-8)
        return -di * (_grad_center(p_ion, np.ones_like(p_ion)) + (p_ion / vt) * dphidx * factor)
    factor = np.maximum(1.0 - p_ion / max(cfg.physics.plim, 1e-30), 1e-8)
    return -di * (_grad_center(p_ion, np.ones_like(p_ion)) / factor + (p_ion / vt) * dphidx)


def rhs_factory(
    cfg: SimulationConfig,
    mesh: DeviceMesh,
    voltage: Callable[[float], float],
    illumination: Callable[[float], float],
    generation: Callable[[np.ndarray, float, float, float], np.ndarray] | None = None,
    recombination: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
):
    s = make_slices(mesh)
    gfun = generation or BeerLambertGeneration()
    rfun = recombination or RecombinationModel(an=cfg.physics.auger_an, ap=cfg.physics.auger_ap)
    se = build_statistical_model(cfg.physics.stats_etl)
    sh = build_statistical_model(cfg.physics.stats_htl)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        n = y[s.n].copy()
        p = y[s.p].copy()
        p_ion = y[s.p_ion].copy()
        n_etl = y[s.n_etl].copy()
        p_htl = y[s.p_htl].copy()

        phi = solve_poisson(cfg, mesh, n, p, p_ion, n_etl, p_htl, voltage(t))
        dphidx = _grad_center(phi, mesh.dx)

        jn = cfg.physics.q * cfg.physics.dn * (_grad_center(n, mesh.dx) - n * dphidx / cfg.physics.vt)
        jp = -cfg.physics.q * cfg.physics.dp * (_grad_center(p, mesh.dx) + p * dphidx / cfg.physics.vt)

        dphidx_perov = dphidx[mesh.perov_idx]
        fp = ion_flux(cfg, p_ion, dphidx_perov)

        g = np.zeros_like(n)
        g[mesh.perov_idx] = gfun(mesh.x[mesh.perov_idx], t, illumination(t), mesh.x[mesh.ib])
        r = np.zeros_like(n)
        r[mesh.perov_idx] = rfun(n[mesh.perov_idx], p[mesh.perov_idx])

        dn_dt = (1.0 / cfg.physics.q) * _div_flux(jn, mesh.dx) + g - r
        dp_dt = -(1.0 / cfg.physics.q) * _div_flux(jp, mesh.dx) + g - r

        dpion_dt = -_div_flux(fp, np.ones_like(fp))

        # TL majority carrier dynamics through generalized statistics and quasi-Fermi variable
        dn_etl_dt = -0.05 * (n_etl - np.exp(se.inverse(np.maximum(n_etl / 1e24, 1e-40))))
        dp_htl_dt = -0.05 * (p_htl - np.exp(sh.inverse(np.maximum(p_htl / 1e24, 1e-40))))

        out = np.zeros_like(y)
        out[s.n] = dn_dt
        out[s.p] = dp_dt
        out[s.p_ion] = dpion_dt
        out[s.n_etl] = dn_etl_dt
        out[s.p_htl] = dp_htl_dt
        return out

    return rhs


def sparse_jacobian_pattern(mesh: DeviceMesh) -> np.ndarray:
    n = make_slices(mesh).p_htl.stop
    main = np.ones(n)
    off = np.ones(n - 1)
    return diags([off, main, off], offsets=[-1, 0, 1], shape=(n, n), format="csr")


def integrate_section(cfg: SimulationConfig, mesh: DeviceMesh, y0: np.ndarray, t_span: tuple[float, float], voltage, illumination):
    rhs = rhs_factory(cfg, mesh, voltage, illumination)
    jac = sparse_jacobian_pattern(mesh) if cfg.solver.use_sparse_jacobian else None
    return solve_ivp(
        rhs,
        t_span,
        y0,
        method=cfg.solver.integrator,
        rtol=cfg.solver.rtol,
        atol=cfg.solver.atol,
        max_step=cfg.solver.max_step,
        jac_sparsity=jac,
    )
