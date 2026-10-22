"""Tiro parabólico con resistencia cuadrática del aire (RK4 de paso fijo)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from caustix._events import locate_apex, locate_impact
from caustix._rk4 import check_step_budget, integrate
from caustix._validate import launch_state, non_negative, positive
from caustix.projectile import G_STANDARD
from caustix.results import FloatArray, Trajectory, build_trajectory

DEFAULT_T_MAX = 600.0
DEFAULT_MAX_STEPS = 1_000_000


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} debe ser un entero >= 1; recibido {value!r}")
    return int(value)


def projectile_with_drag(
    position0: ArrayLike,
    velocity0: ArrayLike,
    *,
    mass: float,
    rho: float,
    cd: float,
    area: float,
    dt: float,
    g: float = G_STANDARD,
    ground: float = 0.0,
    t_max: float = DEFAULT_T_MAX,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Trajectory:
    """Tiro parabólico con arrastre cuadrático, integrado con RK4 de paso fijo.

    Ecuación: ``m·dv/dt = m·g_vec − ½·ρ·C_d·A·|v|·v`` con ``g_vec = (0, …, −g)`` y el eje
    vertical como último eje; equivale a ``a = g_vec − k·|v|·v`` con ``k = ρ·C_d·A/(2m)``.
    El impacto y el ápice se localizan por interpolación cúbica de Hermite dentro del paso
    correspondiente (error O(dt⁴)), de modo que la última muestra queda exactamente en el
    suelo. El último paso se acorta para terminar en ``t_max`` si no hay impacto.

    Args:
        position0: Posición inicial en m, dimensión 2 o 3.
        velocity0: Velocidad inicial en m/s, misma dimensión.
        mass: Masa en kg, finita y > 0.
        rho: Densidad del aire en kg/m³, finita y >= 0 (obligatoria; no hay valor por defecto).
        cd: Coeficiente de arrastre adimensional, finito y >= 0.
        area: Área frontal en m², finita y >= 0.
        dt: Paso de integración en s, finito y > 0.
        g: Gravedad en m/s², finita y >= 0 (por defecto 9,80665).
        ground: Nivel vertical del suelo en m (por defecto 0).
        t_max: Tiempo máximo de simulación en s (por defecto 600).
        max_steps: Máximo de pasos permitidos (por defecto 1 000 000).

    Returns:
        ``Trajectory``; ``hit_ground`` es ``False`` si se alcanzó ``t_max`` sin impacto.

    Raises:
        ValueError: Si una entrada no es finita o está fuera de rango físico, las
            dimensiones son incoherentes, la posición está bajo el suelo o ``t_max/dt``
            supera ``max_steps`` (se comprueba antes de reservar memoria).
        IntegrationError: Si el estado deja de ser finito o ``dt`` supera el límite de
            estabilidad de RK4 (``2·k·max(|v|, v_t)·dt > 2,78``, con ``k = ρ·C_d·A/(2m)`` y
            ``v_t = √(g/k)``).

    Example:
        >>> tr = projectile_with_drag(
        ...     [0.0, 100.0], [0.0, 0.0], mass=1.0, rho=1.225, cd=0.47, area=0.01, dt=0.05
        ... )
        >>> bool(tr.hit_ground), round(tr.time_of_flight, 2)
        (True, 4.74)
    """
    p0, v0, floor = launch_state(position0, velocity0, ground)
    m = positive("mass", mass)
    rho_ = non_negative("rho", rho)
    cd_ = non_negative("cd", cd)
    area_ = non_negative("area", area)
    step = positive("dt", dt)
    grav = non_negative("g", g)
    horizon = positive("t_max", t_max)
    limit = _positive_int("max_steps", max_steps)

    check_step_budget(step, horizon, limit)

    k = rho_ * cd_ * area_ / (2.0 * m)
    gvec = np.zeros_like(p0)
    gvec[-1] = -grav

    def accel(v: FloatArray) -> FloatArray:
        a: FloatArray = gvec - k * float(np.linalg.norm(v)) * v
        return a

    if p0[-1] == floor and (v0[-1] < 0.0 or (v0[-1] == 0.0 and grav > 0.0)):
        return build_trajectory(
            np.zeros(1),
            p0[None, :],
            v0[None, :],
            g=grav,
            ground=floor,
            hit_ground=True,
            max_height=0.0,
        )

    raw = integrate(
        accel,
        p0,
        v0,
        dt=step,
        t_max=horizon,
        max_steps=limit,
        ground=floor,
        drag_rate=k,
        gravity=grav,
    )
    _, apex_z = locate_apex(raw, accel)
    t, position, velocity = raw.t, raw.position, raw.velocity
    if raw.crossed:
        t_hit, x_hit, v_hit = locate_impact(raw, accel, ground=floor)
        t = np.append(t[:-1], t_hit)
        position = np.vstack([position[:-1], x_hit])
        velocity = np.vstack([velocity[:-1], v_hit])
    return build_trajectory(
        t,
        position,
        velocity,
        g=grav,
        ground=floor,
        hit_ground=raw.crossed,
        max_height=max(apex_z - floor, 0.0),
    )
