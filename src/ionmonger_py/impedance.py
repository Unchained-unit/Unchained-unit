from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed

from .config import SimulationConfig
from .mesh import DeviceMesh
from .model import integrate_section
from .protocols import ac_voltage


@dataclass
class ImpedanceResult:
    f: np.ndarray
    z_re: np.ndarray
    z_im: np.ndarray


def _fit_sinusoid(t: np.ndarray, y: np.ndarray, w: float) -> tuple[float, float]:
    M = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    a, b, _ = np.linalg.lstsq(M, y, rcond=None)[0]
    amp = np.hypot(a, b)
    phase = np.arctan2(b, a)
    return amp, phase


def compute_impedance(cfg: SimulationConfig, mesh: DeviceMesh, y_ss: np.ndarray, vdc: float, vp: float, freqs: Iterable[float], n_periods: int = 8, n_jobs: int = 1) -> ImpedanceResult:
    freqs = np.asarray(list(freqs), dtype=float)

    def one_freq(f: float):
        T = 1.0 / f
        t1 = n_periods * T
        vfun = ac_voltage(vdc, vp, f)
        sol = integrate_section(cfg, mesh, y_ss, (0.0, t1), vfun, lambda t: 1.0)
        t = sol.t
        j = sol.y[0]  # proxy signal for current extraction
        mask = t >= (t1 - 2 * T)
        amp, phase = _fit_sinusoid(t[mask], j[mask], 2 * np.pi * f)
        z = vp / max(amp, 1e-30) * np.exp(1j * phase)
        return z.real, z.imag

    vals = Parallel(n_jobs=n_jobs)(delayed(one_freq)(f) for f in freqs)
    z_re = np.array([v[0] for v in vals])
    z_im = np.array([v[1] for v in vals])
    return ImpedanceResult(freqs, z_re, z_im)
