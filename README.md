# ionmonger-py

Python port scaffold of IonMonger 2.0 for 1D planar PSC drift-diffusion + ion migration.

## Features implemented

- 3-layer mesh (ETL / perovskite / HTL) with nonuniform stretching.
- Method-of-lines ODE workflow with Poisson elimination (option A scaffold).
- Configurable ion flux modes: PNP, steric-drift, steric-diffusion.
- Transport-layer statistical model abstraction with lookup + PCHIP inverse.
- Beer-Lambert and spectral generation hooks.
- Bimolecular + SRH + Auger recombination model.
- Protocol engine for preconditioning/JV sections.
- Impedance workflow with sinusoidal fitting and optional joblib parallelism.
- HDF5 output helpers.

## Quickstart

```python
from ionmonger_py import SimulationConfig, simulate
from ionmonger_py.protocols import preconditioning, linear_jv

cfg = SimulationConfig()
sol = simulate([
    preconditioning(v=0.9, duration=0.1),
    linear_jv(v0=1.2, v1=0.0, duration=0.2)
], cfg)
```
