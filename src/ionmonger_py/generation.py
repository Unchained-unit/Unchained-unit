from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BeerLambertGeneration:
    alpha: float = 1e7
    flux0: float = 1e21
    direction: float = 1.0

    def __call__(self, x: np.ndarray, t: float, illumination: float = 1.0, b: float = 500e-9) -> np.ndarray:
        return illumination * self.flux0 * self.alpha * np.exp(-self.alpha * (x - (0 if self.direction > 0 else b)))


@dataclass
class SpectralGeneration:
    energies: np.ndarray
    photon_flux: np.ndarray
    absorption: np.ndarray

    def __call__(self, x: np.ndarray, t: float, illumination: float = 1.0, b: float = 500e-9) -> np.ndarray:
        integrand = self.photon_flux[:, None] * self.absorption[:, None] * np.exp(-self.absorption[:, None] * x[None, :])
        return illumination * np.trapz(integrand, self.energies, axis=0)
