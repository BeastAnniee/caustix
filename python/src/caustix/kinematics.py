"""Cinemática analítica: MRU y MRUA."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from caustix._validate import as_vector, check_same_dim, real_array
from caustix.results import FloatArray, Motion


def _as_time(value: Any) -> FloatArray:
    t = real_array("t", value, "un número o un array numérico real")
    if t.ndim > 1 or t.size == 0:
        raise ValueError(f"t debe ser un escalar o un array 1D no vacío; recibido {value!r}")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"t debe ser finito; recibido {value!r}")
    return t


def _checked(name: str, motion: Motion) -> Motion:
    """Lanza ``OverflowError`` si el resultado de entradas finitas desborda a ``inf``/NaN."""
    if not (np.all(np.isfinite(motion.position)) and np.all(np.isfinite(motion.velocity))):
        raise OverflowError(
            f"{name}: el resultado desborda el rango de float64; reduzca t o los datos"
        )
    return motion


def mru(x0: ArrayLike, v: ArrayLike, t: ArrayLike) -> Motion:
    """Movimiento rectilíneo uniforme.

    Ecuaciones: ``x(t) = x0 + v·t`` y ``v(t) = v``.

    Args:
        x0: Posición inicial en m, vector de dimensión 1 a 3.
        v: Velocidad constante en m/s, misma dimensión que ``x0``.
        t: Instante o instantes en s (escalar o array 1D).

    Returns:
        ``Motion`` con ``position`` y ``velocity`` de forma ``(dim,)`` si ``t`` es
        escalar o ``(N, dim)`` si es un array.

    Raises:
        ValueError: Si alguna entrada no es finita o las dimensiones son incoherentes.
        OverflowError: Si, con entradas finitas, el resultado desborda a ``inf``.

    Example:
        >>> m = mru([0.0, 0.0], [3.0, 4.0], 2.0)
        >>> m.position.tolist()
        [6.0, 8.0]
    """
    x0_ = as_vector("x0", x0)
    v_ = as_vector("v", v)
    check_same_dim(x0=x0_, v=v_)
    t_ = _as_time(t)
    t_col = t_[..., None]
    with np.errstate(over="ignore", invalid="ignore"):
        position = x0_ + v_ * t_col
    velocity = np.broadcast_to(v_, position.shape)
    return _checked("mru", Motion(t=t_, position=position, velocity=velocity))


def mrua(x0: ArrayLike, v0: ArrayLike, a: ArrayLike, t: ArrayLike) -> Motion:
    """Movimiento rectilíneo uniformemente acelerado.

    Ecuaciones: ``x(t) = x0 + v0·t + ½·a·t²`` y ``v(t) = v0 + a·t``.

    Args:
        x0: Posición inicial en m, vector de dimensión 1 a 3.
        v0: Velocidad inicial en m/s, misma dimensión que ``x0``.
        a: Aceleración constante en m/s², misma dimensión que ``x0``.
        t: Instante o instantes en s (escalar o array 1D).

    Returns:
        ``Motion`` con ``position`` y ``velocity`` de forma ``(dim,)`` si ``t`` es
        escalar o ``(N, dim)`` si es un array.

    Raises:
        ValueError: Si alguna entrada no es finita o las dimensiones son incoherentes.
        OverflowError: Si, con entradas finitas, el resultado desborda a ``inf``.

    Example:
        >>> m = mrua([0.0], [0.0], [2.0], 3.0)
        >>> m.position.tolist(), m.velocity.tolist()
        ([9.0], [6.0])
    """
    x0_ = as_vector("x0", x0)
    v0_ = as_vector("v0", v0)
    a_ = as_vector("a", a)
    check_same_dim(x0=x0_, v0=v0_, a=a_)
    t_ = _as_time(t)
    t_col = t_[..., None]
    with np.errstate(over="ignore", invalid="ignore"):
        position = x0_ + v0_ * t_col + 0.5 * a_ * t_col**2
        velocity = v0_ + a_ * t_col
    return _checked("mrua", Motion(t=t_, position=position, velocity=velocity))
