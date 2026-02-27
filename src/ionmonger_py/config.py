from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PositiveFloat


class GeometryConfig(BaseModel):
    b_etl: PositiveFloat = 30e-9
    b_perov: PositiveFloat = 500e-9
    b_htl: PositiveFloat = 30e-9


class MeshConfig(BaseModel):
    n_etl: int = 40
    n_perov: int = 200
    n_htl: int = 40
    stretch_etl: float = 4.0
    stretch_perov_left: float = 6.0
    stretch_perov_right: float = 6.0
    stretch_htl: float = 4.0


class PhysicsConfig(BaseModel):
    vt: PositiveFloat = 0.02585
    q: PositiveFloat = 1.602176634e-19
    eps0: PositiveFloat = 8.8541878128e-12
    eps_a: PositiveFloat = 24.1 * 8.8541878128e-12
    eps_e: PositiveFloat = 10.0 * 8.8541878128e-12
    eps_h: PositiveFloat = 3.0 * 8.8541878128e-12
    dn: PositiveFloat = 1e-6
    dp: PositiveFloat = 1e-6
    di: float = 1e-16
    n0_hat: float = 1e24
    plim: float = 2.5e25
    steric_mode: Literal["none", "drift", "diffusion"] = "none"
    stats_etl: Literal["boltzmann", "parabolic_fd", "gaussian_fd", "blakemore_fd"] = "boltzmann"
    stats_htl: Literal["boltzmann", "parabolic_fd", "gaussian_fd", "blakemore_fd"] = "boltzmann"
    auger_an: float = 0.0
    auger_ap: float = 0.0
    rs: float = 0.0
    rp: float = 1e99
    area: PositiveFloat = 1e-4


class ContactsConfig(BaseModel):
    v_bi: float = 1.0
    ect: float | None = None
    ean: float | None = None


class SolverConfig(BaseModel):
    mode: Literal["ode", "dae"] = "ode"
    integrator: Literal["BDF", "Radau"] = "BDF"
    rtol: float = 1e-5
    atol: float = 1e-8
    steady_abs_tol: float = 1e-8
    steady_rel_tol: float = 1e-5
    max_step: float = 1e-2
    use_sparse_jacobian: bool = True
    jacobian_mode: Literal["finite_diff", "sparse_approx"] = "sparse_approx"


class SimulationConfig(BaseModel):
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    contacts: ContactsConfig = Field(default_factory=ContactsConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)


def load_config(path: str | Path) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SimulationConfig.model_validate(data)
