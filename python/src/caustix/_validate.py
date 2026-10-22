"""Validación de entradas en el borde de la API pública (uso interno)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

_MAX_DIM = 3
_MIN_PROJECTILE_DIM = 2


def finite_scalar(name: str, value: Any) -> float:
    """Convierte ``value`` a ``float`` finito o lanza ``ValueError`` con nombre y valor."""
    if isinstance(value, (bool, np.bool_, str, complex, np.complexfloating)):
        raise ValueError(f"{name} debe ser un número real finito; recibido {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser un número real finito; recibido {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} debe ser finito; recibido {value!r}")
    return result


def positive(name: str, value: Any) -> float:
    """Devuelve ``value`` como ``float`` si es finito y estrictamente positivo."""
    result = finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} debe ser > 0; recibido {value!r}")
    return result


def non_negative(name: str, value: Any) -> float:
    """Devuelve ``value`` como ``float`` si es finito y no negativo."""
    result = finite_scalar(name, value)
    if result < 0.0:
        raise ValueError(f"{name} debe ser >= 0; recibido {value!r}")
    return result


def real_array(name: str, value: Any, what: str) -> NDArray[np.float64]:
    """Copia ``float64`` de ``value``; rechaza complejos, booleanos y texto (``ValueError``)."""
    try:
        raw = np.asarray(value)
        if raw.dtype.kind in "cUSb":
            raise TypeError(raw.dtype)
        return np.array(raw, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser {what}; recibido {value!r}") from None


def as_vector(name: str, value: Any, dim: int | None = None) -> NDArray[np.float64]:
    """Convierte ``value`` en una copia ``float64`` de forma ``(dim,)``, con ``dim`` en 1..3."""
    array = real_array(name, value, "un vector numérico real")
    if array.ndim != 1 or not 1 <= array.shape[0] <= _MAX_DIM:
        raise ValueError(
            f"{name} debe ser un vector de dimensión 1 a {_MAX_DIM}; recibido {value!r}"
        )
    if dim is not None and array.shape[0] != dim:
        raise ValueError(f"{name} debe tener dimensión {dim}; recibido {value!r}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} debe ser finito; recibido {value!r}")
    return array


def check_same_dim(**vectors: NDArray[np.float64]) -> int:
    """Comprueba que todos los vectores nombrados tengan la misma dimensión y la devuelve."""
    dims = {name: vec.shape[0] for name, vec in vectors.items()}
    if len(set(dims.values())) > 1:
        detail = ", ".join(f"{name}={d}" for name, d in dims.items())
        raise ValueError(f"dimensiones incoherentes entre parámetros: {detail}")
    return int(next(iter(dims.values())))


def launch_state(
    position0: Any, velocity0: Any, ground: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Valida el estado inicial de un tiro: vectores 2D/3D coherentes sobre el suelo.

    Returns:
        ``(position0, velocity0, ground)`` como arrays ``float64`` (copias) y ``float``.
    """
    p0 = as_vector("position0", position0)
    if p0.shape[0] < _MIN_PROJECTILE_DIM:
        raise ValueError(f"position0 debe tener dimensión 2 o 3; recibido {position0!r}")
    v0 = as_vector("velocity0", velocity0, dim=p0.shape[0])
    floor = finite_scalar("ground", ground)
    if p0[-1] < floor:
        raise ValueError(f"position0 debe estar sobre el suelo ({floor!r}); recibido {position0!r}")
    return p0, v0, floor
