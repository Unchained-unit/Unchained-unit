from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import SimulationConfig
from .mesh import DeviceMesh, build_mesh
from .model import default_initial_state, integrate_section
from .protocols import ProtocolSection, piecewise_protocol


@dataclass
class Solution:
    t: np.ndarray
    y: np.ndarray
    mesh: DeviceMesh
    sections: list[str]


def simulate(protocol: Sequence[ProtocolSection], params: SimulationConfig) -> Solution:
    mesh = build_mesh(params)
    sections = piecewise_protocol(protocol)
    y0 = default_initial_state(mesh)
    t_all, y_all, labels = [], [], []

    for i, sec in enumerate(sections):
        sol = integrate_section(params, mesh, y0, sec.t_span, sec.voltage, sec.illumination)
        if i == 0:
            t_all.append(sol.t)
            y_all.append(sol.y)
        else:
            t_all.append(sol.t[1:])
            y_all.append(sol.y[:, 1:])
        labels.append(sec.label)
        y0 = sol.y[:, -1]

    return Solution(np.concatenate(t_all), np.hstack(y_all), mesh, labels)
