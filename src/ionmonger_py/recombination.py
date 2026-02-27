from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RecombinationModel:
    beta: float = 1e-16
    taun: float = 1e-6
    taup: float = 1e-6
    ni: float = 1e10
    an: float = 0.0
    ap: float = 0.0

    def __call__(self, n: np.ndarray, p: np.ndarray) -> np.ndarray:
        np_minus_ni2 = n * p - self.ni**2
        r_bim = self.beta * np_minus_ni2
        denom = self.taup * (n + self.ni) + self.taun * (p + self.ni) + 1e-30
        r_srh = np_minus_ni2 / denom
        r_aug = (self.an * n + self.ap * p) * np_minus_ni2
        return r_bim + r_srh + r_aug
