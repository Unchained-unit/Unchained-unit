from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import expit


@dataclass
class StatisticalModel:
    name: str
    s: Callable[[np.ndarray], np.ndarray]

    @cached_property
    def lookup(self) -> "LookupTable":
        return LookupTable.build(self.s)

    def forward(self, xi: np.ndarray) -> np.ndarray:
        return self.lookup.forward(xi)

    def inverse(self, c: np.ndarray) -> np.ndarray:
        return self.lookup.inverse(c)


@dataclass
class LookupTable:
    xi_grid: np.ndarray
    s_grid: np.ndarray
    s_interp: PchipInterpolator
    inv_interp: PchipInterpolator

    @classmethod
    def build(cls, s_func: Callable[[np.ndarray], np.ndarray], xi_min: float = -20, xi_max: float = 20, n: int = 4000) -> "LookupTable":
        xi = np.linspace(xi_min, xi_max, n)
        s = np.maximum(s_func(xi), 1e-40)
        order = np.argsort(s)
        s_sorted = s[order]
        xi_sorted = xi[order]
        uniq, idx = np.unique(s_sorted, return_index=True)
        return cls(xi, s, PchipInterpolator(xi, s, extrapolate=True), PchipInterpolator(uniq, xi_sorted[idx], extrapolate=True))

    def forward(self, xi: np.ndarray) -> np.ndarray:
        return np.asarray(self.s_interp(xi))

    def inverse(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(self.inv_interp(np.maximum(c, 1e-40)))


def _fermi_dirac_like(xi: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(xi))


def _gauss_fermi_like(xi: np.ndarray, s: float = 5.0) -> np.ndarray:
    return np.exp(xi + 0.5 * (1.0 / max(s, 1e-6)) ** 2)


def _blakemore(xi: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(xi))


def build_statistical_model(kind: str) -> StatisticalModel:
    if kind == "boltzmann":
        return StatisticalModel(kind, lambda x: np.exp(np.clip(x, -100, 100)))
    if kind == "parabolic_fd":
        return StatisticalModel(kind, _fermi_dirac_like)
    if kind == "gaussian_fd":
        return StatisticalModel(kind, lambda x: _gauss_fermi_like(x, s=5.0))
    if kind == "blakemore_fd":
        return StatisticalModel(kind, _blakemore)
    raise ValueError(f"Unknown statistical model: {kind}")
