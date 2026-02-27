from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


@dataclass
class DeviceMesh:
    x: np.ndarray
    dx: np.ndarray
    etl_idx: np.ndarray
    perov_idx: np.ndarray
    htl_idx: np.ndarray
    i0: int
    ib: int


def _stretched_segment(x0: float, x1: float, n: int, stretch: float, cluster_left: bool = True) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n + 1)
    if stretch > 1:
        if cluster_left:
            y = np.tanh(stretch * s) / np.tanh(stretch)
        else:
            y = 1.0 - np.tanh(stretch * (1.0 - s)) / np.tanh(stretch)
    else:
        y = s
    return x0 + (x1 - x0) * y


def build_mesh(cfg: SimulationConfig) -> DeviceMesh:
    g, m = cfg.geometry, cfg.mesh
    etl_edges = _stretched_segment(-g.b_etl, 0.0, m.n_etl, m.stretch_etl, cluster_left=False)
    perov_left = _stretched_segment(0.0, g.b_perov / 2, m.n_perov // 2, m.stretch_perov_left, cluster_left=True)
    perov_right = _stretched_segment(g.b_perov / 2, g.b_perov, m.n_perov - m.n_perov // 2, m.stretch_perov_right, cluster_left=False)
    perov_edges = np.concatenate([perov_left[:-1], perov_right])
    htl_edges = _stretched_segment(g.b_perov, g.b_perov + g.b_htl, m.n_htl, m.stretch_htl, cluster_left=True)

    edges = np.concatenate([etl_edges[:-1], perov_edges[:-1], htl_edges])
    x = 0.5 * (edges[:-1] + edges[1:])
    dx = np.diff(edges)

    etl_idx = np.where(x < 0.0)[0]
    perov_idx = np.where((x >= 0.0) & (x <= g.b_perov))[0]
    htl_idx = np.where(x > g.b_perov)[0]

    i0 = perov_idx[0]
    ib = perov_idx[-1]
    return DeviceMesh(x=x, dx=dx, etl_idx=etl_idx, perov_idx=perov_idx, htl_idx=htl_idx, i0=i0, ib=ib)
