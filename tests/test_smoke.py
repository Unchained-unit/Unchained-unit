import numpy as np

from ionmonger_py import SimulationConfig, simulate
from ionmonger_py.impedance import compute_impedance
from ionmonger_py.model import default_initial_state
from ionmonger_py.mesh import build_mesh
from ionmonger_py.protocols import preconditioning


def test_simulate_smoke():
    cfg = SimulationConfig()
    sol = simulate([preconditioning(v=0.8, duration=1e-3)], cfg)
    assert sol.t.size > 2
    assert sol.y.shape[1] == sol.t.size


def test_impedance_smoke():
    cfg = SimulationConfig()
    mesh = build_mesh(cfg)
    y0 = default_initial_state(mesh)
    res = compute_impedance(cfg, mesh, y0, vdc=0.9, vp=0.01, freqs=[10.0], n_periods=2, n_jobs=1)
    assert np.isfinite(res.z_re[0])
