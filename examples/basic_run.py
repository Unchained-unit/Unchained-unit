from ionmonger_py import SimulationConfig, simulate
from ionmonger_py.io import save_solution_hdf5
from ionmonger_py.protocols import linear_jv, preconditioning

cfg = SimulationConfig()
sol = simulate([
    preconditioning(v=0.9, duration=0.05),
    linear_jv(v0=1.1, v1=0.2, duration=0.1),
], cfg)

save_solution_hdf5(
    "run.h5",
    {
        "t": sol.t,
        "y": sol.y,
        "x": sol.mesh.x,
        "sections": sol.sections,
        "config": cfg,
    },
)
print("saved run.h5")
