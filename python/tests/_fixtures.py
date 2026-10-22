"""Lector mínimo de la fixture CSV generada por el validador Fortran."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FIXTURE = Path(__file__).parent / "fixtures" / "fortran_rk4.csv"


@dataclass
class FortranCase:
    """Un caso de referencia: parámetros, resultados globales y muestras escritas."""

    name: str
    dim: int
    mass: float
    rho: float
    cd: float
    area: float
    g: float
    ground: float
    dt: float
    t_max: float
    stride: int
    hit: bool
    t_end: float
    apex_t: float
    max_height: float
    n_samples: int
    p0: NDArray[np.float64]
    v0: NDArray[np.float64]
    indices: list[int] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    position: list[list[float]] = field(default_factory=list)
    velocity: list[list[float]] = field(default_factory=list)


def load_cases(path: Path = FIXTURE) -> list[FortranCase]:
    """Lee todos los casos de la fixture."""
    cases: dict[str, FortranCase] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        if raw.startswith("#case,"):
            f = raw.split(",")
            dim = int(f[2])
            case = FortranCase(
                name=f[1],
                dim=dim,
                mass=float(f[3]),
                rho=float(f[4]),
                cd=float(f[5]),
                area=float(f[6]),
                g=float(f[7]),
                ground=float(f[8]),
                dt=float(f[9]),
                t_max=float(f[10]),
                stride=int(f[11]),
                hit=bool(int(f[12])),
                t_end=float(f[13]),
                apex_t=float(f[14]),
                max_height=float(f[15]),
                n_samples=int(f[16]),
                p0=np.array([float(x) for x in f[17 : 17 + dim]]),
                v0=np.array([float(x) for x in f[20 : 20 + dim]]),
            )
            cases[case.name] = case
        elif raw.startswith("#"):
            continue
        else:
            f = raw.split(",")
            case = cases[f[0]]
            nums = [float(x) for x in f[2:]]
            case.indices.append(int(f[1]))
            case.t.append(nums[0])
            case.position.append(nums[1 : 1 + case.dim])
            case.velocity.append(nums[4 : 4 + case.dim])
    return list(cases.values())
