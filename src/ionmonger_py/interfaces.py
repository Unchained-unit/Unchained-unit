from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InterfaceRates:
    beta_e: float = 1e-12
    beta_h: float = 1e-12


def enforce_potential_continuity(phi: np.ndarray, i0: int, ib: int) -> None:
    phi[i0 - 1] = phi[i0]
    phi[ib + 1] = phi[ib]


def interfacial_recombination(n_left: float, p_right: float, rates: InterfaceRates) -> float:
    return rates.beta_e * n_left * p_right / (n_left + p_right + 1e-30)


def majority_bc_ohmic(value_ref: float, workfunction: float | None, vt: float) -> float:
    if workfunction is None:
        return value_ref
    return np.exp(workfunction / vt)
