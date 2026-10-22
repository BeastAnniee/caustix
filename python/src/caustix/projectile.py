"""Tiro parabólico sin arrastre (solución analítica) y utilidades de lanzamiento."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from caustix._validate import finite_scalar, launch_state, non_negative, positive
from caustix.results import FloatArray, Trajectory, build_trajectory

G_STANDARD = 9.80665
_TIME_EPS = 1e-12
_MAX_SAMPLES = 1_000_000


def launch_velocity(
    speed: float, elevation: float, azimuth: float = 0.0, *, dim: int = 2
) -> FloatArray:
    """Vector velocidad inicial a partir de rapidez y ángulos.

    Ecuaciones: en 2D ``v = s·(cos ε, sin ε)``; en 3D
    ``v = s·(cos ε·cos α, cos ε·sin α, sin ε)``. El último eje es el vertical.

    Args:
        speed: Rapidez en m/s, finita y >= 0.
        elevation: Ángulo sobre la horizontal en **radianes**.
        azimuth: Ángulo en el plano horizontal en **radianes** (solo en 3D).
        dim: Dimensión del resultado, 2 o 3.

    Returns:
        Array de forma ``(dim,)`` en m/s.

    Raises:
        ValueError: Si ``speed`` es negativa, algún valor no es finito o ``dim`` no es 2 ni 3.

    Example:
        >>> launch_velocity(10.0, 0.0).tolist()
        [10.0, 0.0]
    """
    if dim not in (2, 3) or isinstance(dim, bool):
        raise ValueError(f"dim debe ser 2 o 3; recibido {dim!r}")
    s = non_negative("speed", speed)
    eps = finite_scalar("elevation", elevation)
    alpha = finite_scalar("azimuth", azimuth)
    horizontal = s * math.cos(eps)
    vertical = s * math.sin(eps)
    if dim == 2:
        return np.array([horizontal, vertical], dtype=np.float64)
    return np.array(
        [horizontal * math.cos(alpha), horizontal * math.sin(alpha), vertical], dtype=np.float64
    )


def projectile(
    position0: ArrayLike,
    velocity0: ArrayLike,
    *,
    dt: float,
    g: float = G_STANDARD,
    ground: float = 0.0,
) -> Trajectory:
    """Tiro parabólico sin arrastre por su solución analítica cerrada.

    Ecuaciones (eje vertical ``z`` = último eje): ``x(t) = x0 + v0·t + ½·g_vec·t²`` con
    ``g_vec = (0, …, -g)``; tiempo de vuelo
    ``t_v = (v_z + √(v_z² + 2·g·(z0 - suelo))) / g``; altura máxima
    ``z0 + v_z²/(2g) - suelo`` si ``v_z > 0``. La trayectoria se muestrea cada ``dt`` y
    su última muestra es el impacto, exactamente en el suelo.

    Args:
        position0: Posición inicial en m, dimensión 2 o 3.
        velocity0: Velocidad inicial en m/s, misma dimensión.
        dt: Paso de muestreo en s, finito y > 0.
        g: Gravedad en m/s², finita y > 0 (por defecto 9,80665).
        ground: Nivel vertical del suelo en m (por defecto 0).

    Returns:
        ``Trajectory`` con ``hit_ground=True``. Si el lanzamiento parte del suelo con
        velocidad vertical <= 0 tiene una sola muestra y ``time_of_flight = 0``.

    Raises:
        ValueError: Si una entrada no es finita, las dimensiones son incoherentes o < 2,
            ``dt`` o ``g`` no son > 0, la posición inicial está por debajo del suelo, o la
            trayectoria requeriría más de 1 000 000 de muestras (se comprueba antes de reservar
            memoria).

    Example:
        >>> tr = projectile([0.0, 0.0], [3.0, 4.0], dt=0.1)
        >>> bool(tr.hit_ground), round(tr.time_of_flight, 4)
        (True, 0.8158)
    """
    p0, v0, floor = launch_state(position0, velocity0, ground)
    step = positive("dt", dt)
    grav = positive("g", g)
    z0 = float(p0[-1])

    vz = float(v0[-1])
    height = z0 - floor
    if height == 0.0 and vz <= 0.0:
        t_flight = 0.0
    else:
        # hypot evita el desbordamiento de vz*vz con velocidades enormes pero finitas.
        disc = math.hypot(vz, math.sqrt(2.0 * grav * height))
        t_flight = (vz + disc) / grav if vz >= 0.0 else 2.0 * height / (disc - vz)

    samples = t_flight / step
    if not math.isfinite(samples) or samples > _MAX_SAMPLES:
        raise ValueError(
            f"la trayectoria requeriría demasiadas muestras (límite {_MAX_SAMPLES}): "
            f"tiempo de vuelo {t_flight!r} s con dt={dt!r}; aumenta dt"
        )
    n = int(math.floor(samples))
    t = np.arange(n + 1, dtype=np.float64) * step
    t = (
        np.concatenate([t[:1], t[1:][t[1:] < t_flight - _TIME_EPS * step]])
        if t_flight > 0.0
        else t[:1]
    )
    if t_flight > 0.0:
        t = np.append(t, t_flight)

    gvec = np.zeros_like(p0)
    gvec[-1] = -grav
    col = t[:, None]
    position = p0 + v0 * col + 0.5 * gvec * col**2
    velocity = v0 + gvec * col
    if t_flight > 0.0:
        position[-1, -1] = floor
    apex = z0 + vz * vz / (2.0 * grav) if vz > 0.0 else z0
    return build_trajectory(
        t, position, velocity, g=grav, ground=floor, hit_ground=True, max_height=apex - floor
    )
