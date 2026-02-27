from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class ProtocolSection:
    t_span: tuple[float, float]
    voltage: Callable[[float], float]
    illumination: Callable[[float], float] = lambda t: 1.0
    label: str = "section"


def preconditioning(v: float, duration: float, illum: float = 1.0) -> ProtocolSection:
    return ProtocolSection((0.0, duration), lambda t: v, lambda t: illum, label="preconditioning")


def linear_jv(v0: float, v1: float, duration: float, illum: float = 1.0) -> ProtocolSection:
    return ProtocolSection((0.0, duration), lambda t: v0 + (v1 - v0) * (t / duration), lambda t: illum, label="jv_sweep")


def piecewise_protocol(sections: Sequence[ProtocolSection]) -> list[ProtocolSection]:
    out = []
    t_shift = 0.0
    for sec in sections:
        t0, t1 = sec.t_span
        dt = t1 - t0
        out.append(
            ProtocolSection(
                t_span=(t_shift, t_shift + dt),
                voltage=lambda t, sec=sec, ts=t_shift: sec.voltage(t - ts),
                illumination=lambda t, sec=sec, ts=t_shift: sec.illumination(t - ts),
                label=sec.label,
            )
        )
        t_shift += dt
    return out


def ac_voltage(vdc: float, vp: float, freq: float) -> Callable[[float], float]:
    w = 2 * np.pi * freq
    return lambda t: vdc + vp * np.sin(w * t)
