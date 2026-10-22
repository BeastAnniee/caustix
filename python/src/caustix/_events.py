"""Localización de impacto y ápice por interpolación de Hermite (uso interno)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from caustix._rk4 import Accel, RK4Result
from caustix.results import FloatArray

_BISECTION_ITERATIONS = 60


def _hermite(
    p0: FloatArray, m0: FloatArray, p1: FloatArray, m1: FloatArray, h: float, s: float
) -> FloatArray:
    """Interpolación cúbica de Hermite en ``s`` ∈ [0, 1] con derivadas ``m0`` y ``m1``."""
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    result: FloatArray = h00 * p0 + h10 * h * m0 + h01 * p1 + h11 * h * m1
    return result


def _bisect(f: Callable[[float], float], lo: float = 0.0, hi: float = 1.0) -> float:
    """Raíz de ``f`` en [lo, hi] sabiendo que ``f(lo)`` y ``f(hi)`` tienen signo opuesto."""
    sign_lo = f(lo) > 0.0
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0.0) == sign_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _descent_start(
    height: Callable[[float], float],
    z0: float,
    vz0: float,
    z1: float,
    vz1: float,
    h: float,
    ground: float,
) -> float:
    """Inicio del tramo descendente de la cúbica ``height`` (0 si ya parte sobre el suelo).

    Si el paso parte exactamente del suelo, ``height(0) == 0`` y la bisección sobre [0, 1]
    convergería a una raíz espuria; se acota entonces el tramo al máximo de la cúbica.
    """
    if height(0.0) > 0.0:
        return 0.0
    # d/ds de la cúbica de Hermite: a·s² + b·s + c con a = 3(c3), b = 2(c2), c = c1.
    c1 = h * vz0
    c2 = -3.0 * z0 - 2.0 * h * vz0 + 3.0 * z1 - h * vz1
    c3 = 2.0 * z0 + h * vz0 - 2.0 * z1 + h * vz1
    roots = np.roots([3.0 * c3, 2.0 * c2, c1]) if c3 != 0.0 else np.roots([2.0 * c2, c1])
    candidates = [float(r.real) for r in roots if abs(r.imag) < 1e-12 and 0.0 < r.real < 1.0]
    s_max = max(candidates, key=height, default=0.0)
    if height(s_max) <= 0.0:
        raise ValueError(
            "el paso de integración es demasiado grueso para localizar el impacto: "
            f"reduce dt (el lanzamiento parte de z={ground} y el primer paso cae bajo el suelo)"
        )
    return s_max


def locate_impact(
    result: RK4Result, accel: Accel, *, ground: float
) -> tuple[float, FloatArray, FloatArray]:
    """Instante, posición y velocidad del impacto dentro del último paso.

    La posición se interpola con Hermite (posición y velocidad en los extremos) y la
    velocidad con Hermite (velocidad y aceleración); la raíz de ``z(s) = ground`` se
    busca por bisección acotada. La coordenada vertical devuelta es exactamente ``ground``.

    Raises:
        ValueError: Si el último paso no cruza el suelo.
    """
    if not result.crossed:
        raise ValueError("el último paso no presenta un cruce del suelo")
    t0, t1 = float(result.t[-2]), float(result.t[-1])
    h = t1 - t0
    x0, x1 = result.position[-2], result.position[-1]
    v0, v1 = result.velocity[-2], result.velocity[-1]
    a0, a1 = accel(v0), accel(v1)

    def height(s: float) -> float:
        return float(_hermite(x0, v0, x1, v1, h, s)[-1]) - ground

    if height(1.0) == 0.0:
        # Aterrizaje exacto sobre la muestra final del paso: es el impacto, sin bisección.
        s_star = 1.0
    else:
        s_star = _bisect(height, _descent_start(height, x0[-1], v0[-1], x1[-1], v1[-1], h, ground))
    position = _hermite(x0, v0, x1, v1, h, s_star)
    position[-1] = ground
    velocity = _hermite(v0, a0, v1, a1, h, s_star)
    return (t1 if s_star == 1.0 else t0 + s_star * h), position, velocity


def locate_apex(result: RK4Result, accel: Accel) -> tuple[float, float]:
    """Instante y altura (coordenada vertical) del ápice.

    Si la velocidad vertical cambia de signo entre dos muestras, el ápice se localiza por
    bisección sobre la velocidad vertical interpolada (Hermite con la aceleración) y su
    altura con la posición interpolada. Si la trayectoria empieza sin subir, el ápice es
    la primera muestra; si sigue subiendo hasta el final, la última.
    """
    vz = result.velocity[:, -1]
    if vz[0] <= 0.0:
        return float(result.t[0]), float(result.position[0, -1])
    falling = np.nonzero(vz[1:] <= 0.0)[0]
    if falling.size == 0:
        return float(result.t[-1]), float(result.position[-1, -1])
    i = int(falling[0])
    t0, t1 = float(result.t[i]), float(result.t[i + 1])
    h = t1 - t0
    x0, x1 = result.position[i], result.position[i + 1]
    v0, v1 = result.velocity[i], result.velocity[i + 1]
    a0, a1 = accel(v0), accel(v1)

    def vertical_velocity(s: float) -> float:
        return float(_hermite(v0, a0, v1, a1, h, s)[-1])

    s_star = _bisect(vertical_velocity)
    z_apex = float(_hermite(x0, v0, x1, v1, h, s_star)[-1])
    return t0 + s_star * h, z_apex
