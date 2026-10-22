"""Núcleo RK4 contra la solución analítica de la caída con arrastre (AC-5, AC-6, AC-11, AC-12)."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from caustix._rk4 import integrate
from caustix.errors import IntegrationError

G = 9.80665
M, RHO, CD, AREA = 1.0, 1.225, 0.47, 0.01
K = RHO * CD * AREA / (2.0 * M)
V_T = math.sqrt(G / K)
Y0 = 10000.0


def make_accel(k: float, g: float = G) -> Callable[[np.ndarray], np.ndarray]:
    def accel(v: np.ndarray) -> np.ndarray:
        a: np.ndarray = -k * np.linalg.norm(v) * v
        a[-1] -= g
        return a

    return accel


def _logcosh(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2.0 * ax)) - math.log(2.0)


def analytic(t: np.ndarray, y0: float = Y0) -> tuple[np.ndarray, np.ndarray]:
    x = G * t / V_T
    return y0 - (V_T**2 / G) * _logcosh(x), -V_T * np.tanh(x)


def fall(dt: float, t_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    res = integrate(
        make_accel(K), np.array([0.0, Y0]), np.zeros(2), dt=dt, t_max=t_max, max_steps=10**7
    )
    return res.t, res.position, res.velocity


def test_precision_against_analytic_fall() -> None:
    t, pos, vel = fall(1e-3, 5.0)
    y_ref, v_ref = analytic(t)
    assert np.max(np.abs(pos[:, 1] - y_ref) / np.abs(y_ref)) <= 1e-8
    assert np.max(np.abs(vel[1:, 1] - v_ref[1:]) / np.abs(v_ref[1:])) <= 1e-8
    assert np.all(pos[:, 0] == 0.0)


def test_terminal_velocity() -> None:
    _, _, vel = fall(0.05, 70.0)
    assert abs(abs(vel[-1, 1]) / V_T - 1.0) <= 1e-3


def test_observed_order_is_four() -> None:
    dts = [0.4, 0.2, 0.1, 0.05]
    err_y, err_v = [], []
    for dt in dts:
        t, pos, vel = fall(dt, 16.0)
        y_ref, v_ref = analytic(t[-1:])
        err_y.append(abs(pos[-1, 1] - y_ref[0]))
        err_v.append(abs(vel[-1, 1] - v_ref[0]))
    for errs in (err_y, err_v):
        for coarse, fine in zip(errs[:-1], errs[1:], strict=True):
            assert fine > 1e-11  # error muy por encima del ruido de redondeo
            order = math.log2(coarse / fine)
            assert 3.8 <= order <= 4.2, order


def test_samples_and_shortened_last_step() -> None:
    res = integrate(
        make_accel(K),
        np.array([0.0, 100.0]),
        np.array([5.0, 5.0]),
        dt=0.3,
        t_max=1.0,
        max_steps=100,
    )
    np.testing.assert_allclose(res.t, [0.0, 0.3, 0.6, 0.9, 1.0], rtol=0, atol=1e-15)
    assert res.t[-1] == 1.0
    assert res.position.shape == (5, 2)
    assert res.velocity.shape == (5, 2)
    assert res.crossed is False


def test_exact_multiple_does_not_add_tiny_step() -> None:
    res = integrate(
        make_accel(K), np.array([0.0, 100.0]), np.zeros(2), dt=1e-3, t_max=5.0, max_steps=10**6
    )
    assert res.t.shape == (5001,)


def test_t_max_reached_without_ground() -> None:
    res = integrate(
        make_accel(K),
        np.array([0.0, 1000.0]),
        np.zeros(2),
        dt=0.1,
        t_max=2.0,
        max_steps=1000,
        ground=0.0,
    )
    assert res.crossed is False
    assert res.t[-1] == 2.0


def test_stops_at_first_step_below_ground() -> None:
    res = integrate(
        make_accel(K),
        np.array([0.0, 100.0]),
        np.zeros(2),
        dt=0.1,
        t_max=60.0,
        max_steps=10**5,
        ground=0.0,
    )
    assert res.crossed is True
    assert res.position[-1, 1] < 0.0 <= res.position[-2, 1]
    assert res.t[-1] < 60.0


def test_launch_from_ground_going_up_is_not_a_crossing() -> None:
    res = integrate(
        make_accel(K),
        np.array([0.0, 0.0]),
        np.array([10.0, 20.0]),
        dt=0.05,
        t_max=0.5,
        max_steps=1000,
        ground=0.0,
    )
    assert res.crossed is False


def test_max_steps_exceeded_raises_before_allocating() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        integrate(
            make_accel(K), np.array([0.0, 1.0]), np.zeros(2), dt=1e-12, t_max=600.0, max_steps=10**6
        )
    with pytest.raises(ValueError, match="max_steps"):
        integrate(
            make_accel(K), np.array([0.0, 1.0]), np.zeros(2), dt=0.1, t_max=10.0, max_steps=99
        )
    # justo en el límite: permitido
    res = integrate(
        make_accel(K), np.array([0.0, 1e6]), np.zeros(2), dt=0.1, t_max=10.0, max_steps=100
    )
    assert res.t.shape == (101,)


def test_non_finite_state_raises_integration_error() -> None:
    with pytest.raises(IntegrationError):
        integrate(
            make_accel(1.0),
            np.array([0.0, 1e6]),
            np.array([1e3, 1e3]),
            dt=10.0,
            t_max=1000.0,
            max_steps=10**6,
        )


def test_inputs_not_mutated() -> None:
    y0 = np.array([0.0, 50.0])
    v0 = np.array([3.0, 4.0])
    integrate(make_accel(K), y0, v0, dt=0.1, t_max=1.0, max_steps=100)
    np.testing.assert_array_equal(y0, [0.0, 50.0])
    np.testing.assert_array_equal(v0, [3.0, 4.0])


@pytest.mark.parametrize("dt, t_max", [(1e-10, 1e308), (5e-324, 1.0)])
def test_step_count_overflow_raises_value_error(dt: float, t_max: float) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        integrate(
            make_accel(K), np.array([0.0, 10.0]), np.zeros(2), dt=dt, t_max=t_max, max_steps=10**6
        )
