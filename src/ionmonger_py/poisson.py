from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .config import SimulationConfig
from .mesh import DeviceMesh


def solve_poisson(cfg: SimulationConfig, mesh: DeviceMesh, n: np.ndarray, p: np.ndarray, p_ion: np.ndarray, n_etl: np.ndarray, p_htl: np.ndarray, v_applied: float) -> np.ndarray:
    x, dx = mesh.x, mesh.dx
    N = len(x)
    dxi = np.r_[dx[0], 0.5 * (dx[:-1] + dx[1:]), dx[-1]]
    lower = np.zeros(N)
    diag = np.zeros(N)
    upper = np.zeros(N)

    eps = np.full(N, cfg.physics.eps_a)
    eps[mesh.etl_idx] = cfg.physics.eps_e
    eps[mesh.htl_idx] = cfg.physics.eps_h

    rho = np.zeros(N)
    rho[mesh.perov_idx] = cfg.physics.q * (cfg.physics.n0_hat - p_ion + n[mesh.perov_idx] - p[mesh.perov_idx])
    rho[mesh.etl_idx] = cfg.physics.q * (n_etl - 1e24)
    rho[mesh.htl_idx] = cfg.physics.q * (1e24 - p_htl)

    for i in range(1, N - 1):
        hm = dxi[i]
        hp = dxi[i + 1]
        lower[i] = eps[i] / hm
        upper[i] = eps[i] / hp
        diag[i] = -(lower[i] + upper[i])

    diag[0] = 1.0
    diag[-1] = 1.0
    b = -rho.copy()
    b[0] = (cfg.contacts.v_bi - v_applied) / 2.0
    b[-1] = -(cfg.contacts.v_bi - v_applied) / 2.0

    A = diags([lower[1:], diag, upper[:-1]], offsets=[-1, 0, 1], format="csr")
    return spsolve(A, b)
