from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _to_jsonable(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def save_solution_hdf5(path: str | Path, data: dict[str, Any]) -> None:
    with h5py.File(path, "w") as h5:
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                h5.create_dataset(k, data=v)
            elif np.isscalar(v):
                h5.attrs[k] = v
            else:
                h5.attrs[k] = json.dumps(_to_jsonable(v))


def load_solution_hdf5(path: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as h5:
        for k in h5.keys():
            out[k] = h5[k][...]
        for k, v in h5.attrs.items():
            out[k] = v
    return out
