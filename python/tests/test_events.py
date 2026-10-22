"""Eventos de impacto y ápice por interpolación de Hermite (AC-7, AC-9)."""

import math

import numpy as np
import pytest

from caustix._events import locate_apex, locate_impact
from caustix._rk4 import RK4Result, integrate

G = 9.80665
M, RHO, CD, AREA = 1.0, 1.225, 0.47, 0.01
K = RHO * CD * AREA / (2.0 * M)
V_T = math.sqrt(G / K)


def accel(v: np.ndarray) -> np.ndarray:
    a: np.ndarray = -K * np.linalg.norm(v) * v
    a[-1] -= G
    return a


def run(
    p0: list[float], v0: list[float], dt: float, t_max: float, ground: float | None = 0.0
) -> RK4Result:
    return integrate(
        accel, np.array(p0), np.array(v0), dt=dt, t_max=t_max, max_steps=10**7, ground=ground
    )


def observed_orders(errors: list[float]) -> list[float]:
    return [math.log2(a / b) for a, b in zip(errors[:-1], errors[1:], strict=True)]


def test_impact_time_converges_with_order_at_least_three() -> None:
    y0 = 100.0
    t_star = (V_T / G) * math.acosh(math.exp(G * y0 / V_T**2))
    errors = []
    for dt in (0.4, 0.2, 0.1, 0.05):
        res = run([0.0, y0], [0.0, 0.0], dt, 60.0)
        assert res.crossed
        t_imp, pos, _vel = locate_impact(res, accel, ground=0.0)
        errors.append(abs(t_imp - t_star))
        assert abs(pos[-1]) <= 1e-9 * y0  # exactamente en el suelo
    assert all(e > 1e-12 for e in errors)
    assert all(o >= 3.0 for o in observed_orders(errors)), observed_orders(errors)


def test_impact_point_is_exactly_on_ground_and_between_samples() -> None:
    res = run([0.0, 50.0], [20.0, 10.0], 0.1, 60.0, ground=2.0)
    t_imp, pos, vel = locate_impact(res, accel, ground=2.0)
    assert pos[-1] == 2.0
    assert res.t[-2] < t_imp <= res.t[-1]
    assert vel[-1] < 0.0
    assert pos[0] > res.position[-2, 0]


def test_apex_height_converges_with_order_at_least_three() -> None:
    v0 = [50.0 * math.cos(math.radians(60)), 50.0 * math.sin(math.radians(60))]
    ref_t, ref_z = locate_apex(run([0.0, 0.0], v0, 1e-4, 6.0, ground=None), accel)
    errors_t, errors_z = [], []
    for dt in (0.1, 0.05, 0.025, 0.0125):
        t_a, z_a = locate_apex(run([0.0, 0.0], v0, dt, 6.0, ground=None), accel)
        errors_t.append(abs(t_a - ref_t))
        errors_z.append(abs(z_a - ref_z))
    assert all(e > 1e-12 for e in errors_z)
    assert all(o >= 3.0 for o in observed_orders(errors_z)), observed_orders(errors_z)
    # El error del instante del ápice cambia de signo al refinar y roza el ruido de
    # redondeo, así que solo se exige que converja (AC-9 pide el orden de la altura).
    assert errors_t[-1] < errors_t[0] / 100.0
    assert errors_t[-1] < 1e-10


def test_apex_edge_cases() -> None:
    # empieza descendiendo: el ápice es la muestra inicial
    res = run([0.0, 7.0], [3.0, -1.0], 0.1, 1.0, ground=None)
    assert locate_apex(res, accel) == (0.0, 7.0)
    # sigue subiendo al final: el máximo es la última muestra
    res = run([0.0, 0.0], [0.0, 500.0], 0.1, 0.3, ground=None)
    t_a, z_a = locate_apex(res, accel)
    assert t_a == res.t[-1]
    assert z_a == res.position[-1, -1]


def mechanical_energy(res: RK4Result, mass: float) -> np.ndarray:
    ke = 0.5 * mass * np.sum(res.velocity**2, axis=1)
    return ke + mass * G * res.position[:, -1]


def simpson(y: np.ndarray, h: float) -> float:
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


def test_energy_is_non_increasing_and_balance_converges() -> None:
    v0 = [30.0, 40.0]
    residuals = []
    for dt in (4e-3, 2e-3, 1e-3):
        res = run([0.0, 5.0], v0, dt, 3.0, ground=None)
        e = mechanical_energy(res, M)
        assert np.all(np.diff(e) <= 1e-12 * e[0])
        speed = np.linalg.norm(res.velocity, axis=1)
        power = M * K * speed**3  # F_d·v con F_d = m·k·|v|·v
        residuals.append(abs((e[-1] - e[0]) + simpson(power, dt)) / e[0])
    assert residuals[-1] <= 1e-6
    assert residuals[0] > residuals[1] > residuals[2]


def test_locate_impact_requires_a_crossing() -> None:
    with pytest.raises(ValueError, match="cruce"):
        locate_impact(run([0.0, 1000.0], [0.0, 0.0], 0.1, 1.0), accel, ground=0.0)
