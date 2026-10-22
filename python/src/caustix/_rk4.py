"""Integrador Runge-Kutta de 4.º orden de paso fijo (uso interno)."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from caustix.errors import IntegrationError
from caustix.results import FloatArray

Accel = Callable[[FloatArray], FloatArray]

_STEP_COUNT_TOL = 1e-9
# Límite de estabilidad de RK4 sobre el eje real negativo (|h·λ| < 2,785). Para el arrastre
# cuadrático el autovalor a lo largo de v es λ = −2·k·|v|, que se acota con max(|v|, v_t).
_RK4_STABILITY_LIMIT = 2.78
_LANDING_ULPS = 4.0


@dataclass(frozen=True)
class RK4Result:
    """Muestras de una integración RK4.

    Attributes:
        t: Instantes, forma ``(N,)``.
        position: Posiciones, forma ``(N, dim)``.
        velocity: Velocidades, forma ``(N, dim)``.
        crossed: ``True`` si la última muestra es la primera que está bajo el suelo o sobre
            él descendiendo (``z == ground`` con ``v_z < 0``: aterrizaje exacto en la muestra).
    """

    t: FloatArray
    position: FloatArray
    velocity: FloatArray
    crossed: bool


def step_count(dt: float, t_max: float) -> int:
    """Número de pasos para cubrir ``t_max`` con paso ``dt`` (el último se acorta)."""
    ratio = t_max / dt
    if not math.isfinite(ratio):
        return sys.maxsize
    nearest = round(ratio)
    if abs(ratio - nearest) <= _STEP_COUNT_TOL * max(1.0, ratio):
        return max(int(nearest), 1)
    return int(math.ceil(ratio))


def check_step_budget(dt: float, t_max: float, max_steps: int) -> int:
    """Número de pasos ``t_max/dt``; lanza ``ValueError`` si supera ``max_steps``."""
    n = step_count(dt, t_max)
    if n > max_steps:
        raise ValueError(
            f"max_steps={max_steps!r} insuficiente: t_max/dt requiere {n} pasos "
            f"(t_max={t_max!r}, dt={dt!r}); reduce t_max o aumenta dt/max_steps"
        )
    return n


def _rk4_step(
    accel: Accel, x: FloatArray, v: FloatArray, h: float
) -> tuple[FloatArray, FloatArray]:
    k1v = accel(v)
    k2v = accel(v + 0.5 * h * k1v)
    k3v = accel(v + 0.5 * h * k2v)
    k4v = accel(v + h * k3v)
    k2x = v + 0.5 * h * k1v
    k3x = v + 0.5 * h * k2v
    k4x = v + h * k3v
    x_new = x + (h / 6.0) * (v + 2.0 * k2x + 2.0 * k3x + k4x)
    v_new = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return x_new, v_new


def _check_stability(
    drag_rate: float, speed_floor: float, v: FloatArray, h: float, dt: float, at: float
) -> None:
    if drag_rate <= 0.0:
        return
    # |v| calculada o, si es mayor, la velocidad terminal: con un dt grueso RK4 oscila o cae en
    # un punto fijo con |v| muy inferior a la real y la métrica local no vería la rigidez.
    stiffness = 2.0 * drag_rate * max(float(np.linalg.norm(v)), speed_floor) * h
    if not stiffness <= _RK4_STABILITY_LIMIT:
        raise IntegrationError(
            f"dt={dt!r} demasiado grande: 2·k·max(|v|, v_t)·dt = {stiffness:.3g} supera el "
            f"límite de estabilidad de RK4 ({_RK4_STABILITY_LIMIT}) cerca de t={at!r} s; reduzca dt"
        )


def integrate(
    accel: Accel,
    position0: FloatArray,
    velocity0: FloatArray,
    *,
    dt: float,
    t_max: float,
    max_steps: int,
    ground: float | None = None,
    drag_rate: float = 0.0,
    gravity: float = 0.0,
) -> RK4Result:
    """Integra ``x' = v``, ``v' = accel(v)`` con RK4 clásico de paso fijo.

    Con ``ground`` indicado, la integración se detiene en la primera muestra cuyo eje
    vertical (último) cae por debajo del suelo, o está exactamente en él con velocidad
    vertical negativa (``crossed=True``); esa muestra solo sirve para localizar el evento,
    que se refina aparte. «Exactamente» incluye el residuo de redondeo acumulado (unos pocos
    ulp por paso, relativo a la escala vertical): esa muestra se fija en ``ground`` para que el
    aterrizaje sobre la malla no dependa de que la aritmética sea bit a bit exacta.

    Con ``drag_rate`` (= ``k`` de ``a = … − k·|v|·v``) > 0 se comprueba antes y después de cada
    paso la estabilidad de RK4: ``2·k·max(|v|, v_t)·h`` debe ser <= 2,78, con
    ``v_t = √(gravity/k)`` la velocidad terminal (0 si ``gravity`` es 0).

    Raises:
        ValueError: Si el número de pasos ``t_max/dt`` supera ``max_steps`` (se comprueba
            antes de reservar memoria).
        IntegrationError: Si el estado deja de ser finito o ``dt`` supera el límite de
            estabilidad de RK4.
    """
    n = check_step_budget(dt, t_max, max_steps)
    dim = position0.shape[0]
    t = np.empty(n + 1, dtype=np.float64)
    pos = np.empty((n + 1, dim), dtype=np.float64)
    vel = np.empty((n + 1, dim), dtype=np.float64)
    t[0] = 0.0
    pos[0] = position0
    vel[0] = velocity0
    x = position0.copy()
    v = velocity0.copy()
    crossed = False
    last = 0
    speed_floor = math.sqrt(gravity / drag_rate) if drag_rate > 0.0 and gravity > 0.0 else 0.0
    z_scale = max(1.0, abs(x[-1]), abs(ground) if ground is not None else 0.0)
    landing_tol = _LANDING_ULPS * np.finfo(np.float64).eps * max(n, 16)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(n):
            t_next = t_max if i == n - 1 else (i + 1) * dt
            h = dt if i < n - 1 else t_max - i * dt
            _check_stability(drag_rate, speed_floor, v, h, dt, i * dt)
            x, v = _rk4_step(accel, x, v, h)
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
                raise IntegrationError(
                    f"estado no finito en t={t_next!r} s (dt={dt!r}); reduzca dt"
                )
            # También tras el paso: desde el reposo un dt grueso explota en el primer paso.
            _check_stability(drag_rate, speed_floor, v, h, dt, t_next)
            t[i + 1] = t_next
            pos[i + 1] = x
            vel[i + 1] = v
            last = i + 1
            if ground is None:
                continue
            z_scale = max(z_scale, abs(x[-1]))
            on_ground = abs(x[-1] - ground) <= landing_tol * z_scale and v[-1] < 0.0
            if x[-1] < ground or on_ground:
                if on_ground:
                    pos[i + 1, -1] = ground
                crossed = True
                break
    return RK4Result(
        t=t[: last + 1].copy(),
        position=pos[: last + 1].copy(),
        velocity=vel[: last + 1].copy(),
        crossed=crossed,
    )
