"""projectile_with_drag: límites, validación, límites de simulación (AC-8, AC-10, AC-12, AC-16)."""

import math
from typing import Any

import numpy as np
import pytest

from caustix import IntegrationError, launch_velocity, projectile, projectile_with_drag

G = 9.80665
BODY: dict[str, Any] = {"mass": 1.0, "rho": 1.225, "cd": 0.47, "area": 0.01}
K = 1.225 * 0.47 * 0.01 / 2.0
V_T = math.sqrt(G / K)


@pytest.mark.parametrize("coefficient", [0.0, 1e-12])
@pytest.mark.parametrize("dim", [2, 3])
def test_vacuum_limit_matches_analytic(coefficient: float, dim: int) -> None:
    v0 = launch_velocity(30.0, math.radians(55.0), math.radians(20.0), dim=dim)
    p0 = np.zeros(dim)
    p0[-1] = 4.0
    dt = 0.01
    exact = projectile(p0, v0, dt=dt)
    drag = projectile_with_drag(
        p0, v0, mass=1.0, rho=1.0, cd=1.0, area=coefficient, dt=dt, t_max=600.0
    )
    assert drag.hit_ground is True
    assert drag.t.shape == exact.t.shape
    np.testing.assert_allclose(drag.t, exact.t, rtol=1e-9, atol=1e-12)
    # rtol respecto a la escala de la trayectoria: cerca del suelo la diferencia física
    # (~1e-9 m con ρ·C_d·A = 1e-12) supera 1e-9 en relación al propio valor de z.
    for got, want in ((drag.position, exact.position), (drag.velocity, exact.velocity)):
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9 * float(np.max(np.abs(want))))
    assert drag.time_of_flight == pytest.approx(exact.time_of_flight, rel=1e-9)
    assert drag.range == pytest.approx(exact.range, rel=1e-9)
    assert drag.max_height == pytest.approx(exact.max_height, rel=1e-9)


def test_fall_with_drag_matches_analytic_impact_time() -> None:
    y0 = 100.0
    t_star = (V_T / G) * math.acosh(math.exp(G * y0 / V_T**2))
    tr = projectile_with_drag([0.0, y0], [0.0, 0.0], dt=0.05, **BODY)
    assert tr.hit_ground is True
    assert tr.time_of_flight == pytest.approx(t_star, rel=1e-8)
    assert tr.position[-1, -1] == 0.0
    assert tr.max_height == pytest.approx(y0)


def test_drag_shortens_range_and_height() -> None:
    v0 = launch_velocity(50.0, math.radians(45.0))
    free = projectile([0.0, 0.0], v0, dt=0.01)
    drag = projectile_with_drag([0.0, 0.0], v0, dt=0.01, **BODY)
    assert drag.range < free.range
    assert drag.max_height < free.max_height
    assert drag.time_of_flight < free.time_of_flight
    assert drag.max_height >= float(np.max(drag.position[:, -1]))


def test_energy_never_increases_and_last_sample_on_ground() -> None:
    v0 = launch_velocity(40.0, math.radians(35.0), math.radians(15.0), dim=3)
    tr = projectile_with_drag([0.0, 0.0, 2.0], v0, dt=0.005, ground=1.0, **BODY)
    e = tr.mechanical_energy(BODY["mass"])
    assert np.all(np.diff(e) <= 1e-12 * e[0])
    assert tr.position[-1, -1] == 1.0
    assert tr.hit_ground is True


def test_t_max_reached_without_impact() -> None:
    tr = projectile_with_drag([0.0, 1000.0], [0.0, 0.0], dt=0.1, t_max=2.0, **BODY)
    assert tr.hit_ground is False
    assert tr.t[-1] == 2.0
    assert tr.time_of_flight == 2.0


def test_zero_gravity_never_lands() -> None:
    tr = projectile_with_drag([0.0, 1.0], [5.0, 5.0], dt=0.1, g=0.0, t_max=3.0, **BODY)
    assert tr.hit_ground is False


def test_max_steps_limits() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        projectile_with_drag([0.0, 1.0], [1.0, 1.0], dt=1e-9, **BODY)
    with pytest.raises(ValueError, match="max_steps"):
        projectile_with_drag([0.0, 1.0], [1.0, 1.0], dt=0.1, t_max=10.0, max_steps=50, **BODY)


def test_non_finite_state_raises_integration_error() -> None:
    with pytest.raises(IntegrationError):
        projectile_with_drag(
            [0.0, 1e6],
            [1e3, 0.0],
            dt=10.0,
            t_max=1000.0,
            g=0.0,
            mass=1.0,
            rho=1.0,
            cd=1.0,
            area=1.0,
        )


def test_starts_on_ground_going_down_is_single_sample() -> None:
    tr = projectile_with_drag([0.0, 0.0], [3.0, -1.0], dt=0.1, **BODY)
    assert tr.t.shape == (1,)
    assert tr.hit_ground is True
    assert tr.time_of_flight == 0.0


def test_rho_is_required() -> None:
    with pytest.raises(TypeError):
        projectile_with_drag(  # type: ignore[call-arg]
            [0.0, 1.0], [1.0, 1.0], mass=1.0, cd=0.5, area=0.1, dt=0.1
        )


def test_deterministic_and_arguments_not_mutated() -> None:
    p0 = np.array([0.0, 10.0])
    v0 = np.array([12.0, 15.0])
    a = projectile_with_drag(p0, v0, dt=0.01, **BODY)
    b = projectile_with_drag(p0, v0, dt=0.01, **BODY)
    np.testing.assert_array_equal(p0, [0.0, 10.0])
    np.testing.assert_array_equal(v0, [12.0, 15.0])
    for name in ("t", "position", "velocity"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes()
    assert a.time_of_flight == b.time_of_flight


def test_result_is_immutable() -> None:
    tr = projectile_with_drag([0.0, 10.0], [3.0, 4.0], dt=0.05, **BODY)
    with pytest.raises(ValueError, match="read-only"):
        tr.position[0, 0] = 1.0
    with pytest.raises(AttributeError):
        tr.hit_ground = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("override", "name"),
    [
        ({"mass": 0.0}, "mass"),
        ({"mass": -1.0}, "mass"),
        ({"mass": math.nan}, "mass"),
        ({"rho": -0.1}, "rho"),
        ({"rho": math.inf}, "rho"),
        ({"cd": -0.1}, "cd"),
        ({"cd": math.nan}, "cd"),
        ({"area": -1.0}, "area"),
        ({"dt": 0.0}, "dt"),
        ({"dt": -1.0}, "dt"),
        ({"dt": math.inf}, "dt"),
        ({"g": -1.0}, "g"),
        ({"g": math.nan}, "g"),
        ({"ground": math.nan}, "ground"),
        ({"ground": 10.0}, "position0"),
        ({"t_max": 0.0}, "t_max"),
        ({"t_max": math.nan}, "t_max"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": 1.5}, "max_steps"),
        ({"max_steps": True}, "max_steps"),
    ],
)
def test_invalid_scalar_inputs(override: dict[str, Any], name: str) -> None:
    kwargs: dict[str, Any] = {**BODY, "dt": 0.1, **override}
    with pytest.raises(ValueError, match=name):
        projectile_with_drag([0.0, 1.0], [1.0, 1.0], **kwargs)


@pytest.mark.parametrize(
    ("p0", "v0", "name"),
    [
        ([0.0], [1.0], "position0"),
        ([0.0, 0.0], [1.0, 1.0, 1.0], "velocity0"),
        ([0.0, math.nan], [1.0, 1.0], "position0"),
        ([0.0, 0.0], [math.inf, 1.0], "velocity0"),
        ([0.0] * 4, [1.0] * 4, "position0"),
    ],
)
def test_invalid_vectors(p0: list[float], v0: list[float], name: str) -> None:
    with pytest.raises(ValueError, match=name):
        projectile_with_drag(p0, v0, dt=0.1, **BODY)


@pytest.mark.parametrize("z0", [0.0, 1e-3])
@pytest.mark.parametrize("dt", [0.21, 0.5, 1.0])
def test_coarse_dt_longer_than_flight_time_locates_the_real_impact(z0: float, dt: float) -> None:
    """Review 1: con dt > tiempo de vuelo el impacto no puede ser el redondeo a la malla."""
    kwargs: dict[str, Any] = {**BODY, "t_max": 5.0, "max_steps": 10**8}
    p0, v0 = [0.0, z0], [10.0, 1.0]
    ref = projectile_with_drag(p0, v0, dt=1e-4, **kwargs)
    coarse = projectile_with_drag(p0, v0, dt=dt, **kwargs)
    assert coarse.hit_ground is True
    assert coarse.time_of_flight == pytest.approx(ref.time_of_flight, rel=5e-3)
    assert coarse.range == pytest.approx(ref.range, rel=5e-3)
    assert coarse.position[-1, -1] == 0.0


NO_DRAG: dict[str, Any] = {"mass": 1.0, "rho": 0.0, "cd": 1.0, "area": 1.0, "g": 10.0}


@pytest.mark.parametrize("dt", [0.125, 0.25, 0.5, 1.0])
def test_exact_landing_on_a_grid_sample_is_the_impact(dt: float) -> None:
    # Caída libre exacta (g=10): z(t) = 5 - 5 t² toca el suelo en t = 1 s, que es una muestra
    # de la malla para estos dt (potencias de 2): el impacto es esa muestra, no un error.
    tr = projectile_with_drag([0.0, 5.0], [1.0, 0.0], dt=dt, **NO_DRAG)
    assert tr.hit_ground is True
    assert tr.time_of_flight == pytest.approx(1.0, abs=1e-12)
    assert tr.position[-1, -1] == 0.0
    assert tr.position[-1, 0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(tr.t) > 0.0)
    assert tr.t.shape == projectile([0.0, 5.0], [1.0, 0.0], dt=dt, g=10.0).t.shape


@pytest.mark.parametrize("dt", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("ground", [0.0, 3.0])
def test_exact_landing_after_initial_ascent(dim: int, ground: float, dt: float) -> None:
    # v0 vertical = 20, g = 10: vuelve a la altura inicial en t = 4 s (muestra de la malla).
    p0 = [0.0] * (dim - 1) + [ground]
    v0 = [1.0] + [0.0] * (dim - 2) + [20.0]
    tr = projectile_with_drag(p0, v0, dt=dt, ground=ground, **NO_DRAG)
    assert tr.hit_ground is True
    assert tr.time_of_flight == pytest.approx(4.0, abs=1e-12)
    assert tr.position[-1, -1] == ground
    # REQ-10/REQ-12: t estrictamente creciente y misma malla que la solución en el vacío.
    assert np.all(np.diff(tr.t) > 0.0)
    free = projectile(p0, v0, dt=dt, g=10.0, ground=ground)
    assert tr.t.shape == free.t.shape
    np.testing.assert_allclose(tr.t, free.t, atol=1e-12)


@pytest.mark.parametrize("dt", [0.25, 0.5, 1.0])
def test_landing_exactly_at_t_max_counts_as_impact(dt: float) -> None:
    tr = projectile_with_drag([0.0, 5.0], [1.0, 0.0], dt=dt, t_max=1.0, **NO_DRAG)
    assert tr.hit_ground is True
    assert tr.time_of_flight == 1.0
    assert tr.position[-1, -1] == 0.0
    assert np.all(np.diff(tr.t) > 0.0)


def test_t_max_just_before_landing_is_not_an_impact() -> None:
    tr = projectile_with_drag([0.0, 5.0], [1.0, 0.0], dt=0.5, t_max=0.75, **NO_DRAG)
    assert tr.hit_ground is False
    assert tr.t[-1] == 0.75


def test_step_budget_is_checked_even_when_launched_on_the_ground_going_down() -> None:
    # Contrato de error independiente de la entrada: mismo ValueError que con vz > 0.
    for vz in (-1.0, 1.0):
        with pytest.raises(ValueError, match="max_steps"):
            projectile_with_drag([0.0, 0.0], [1.0, vz], dt=1e-9, t_max=600.0, **NO_DRAG)


def test_zero_gravity_body_resting_on_ground_does_not_land() -> None:
    # g = 0 y vz = 0: nada lo empuja hacia el suelo; no hay impacto (§5 de la spec).
    tr = projectile_with_drag([0.0, 0.0], [1.0, 0.0], dt=0.1, t_max=1.0, **{**NO_DRAG, "g": 0.0})
    assert tr.hit_ground is False
    assert tr.t[-1] == pytest.approx(1.0)


def test_body_on_ground_moving_down_or_pulled_down_lands_at_t0() -> None:
    for vz, g in [(0.0, 10.0), (-1.0, 0.0), (-1.0, 10.0)]:
        tr = projectile_with_drag([0.0, 0.0], [1.0, vz], dt=0.1, **{**NO_DRAG, "g": g})
        assert tr.hit_ground is True
        assert tr.time_of_flight == 0.0


FAST: dict[str, Any] = {"mass": 1.0, "rho": 1.2, "cd": 0.4, "area": 0.01}


@pytest.mark.parametrize("dt", [5.0, 20.0, 100.0, 1e6])
def test_unstable_finite_dt_raises_instead_of_returning_nonsense(dt: float) -> None:
    # 2·k·|v|·dt > ~2,78 (límite de estabilidad de RK4): antes devolvía max_height < 0.
    with pytest.raises(IntegrationError, match="dt"):
        projectile_with_drag([0.0, 0.0], [100.0, 100.0], dt=dt, **FAST)


def test_stable_dt_just_below_the_limit_is_accepted() -> None:
    k = 1.2 * 0.4 * 0.01 / 2.0
    speed = math.hypot(100.0, 100.0)
    dt = 2.7 / (2.0 * k * speed)
    tr = projectile_with_drag([0.0, 0.0], [100.0, 100.0], dt=dt, **FAST)
    assert tr.max_height >= 0.0


def test_unstable_dt_is_detected_when_speed_grows_during_a_fall() -> None:
    # Parte en reposo (estable) y la caída libre acelera hasta superar el límite.
    with pytest.raises(IntegrationError, match="dt"):
        projectile_with_drag(
            [0.0, 1e6], [0.0, 0.0], dt=50.0, t_max=600.0, mass=1.0, rho=1.2, cd=0.4, area=1.0
        )


def test_zero_drag_never_trips_the_stability_check() -> None:
    tr = projectile_with_drag([0.0, 100.0], [0.0, 1e3], dt=1e3, t_max=1e6, **{**FAST, "rho": 0.0})
    assert tr.hit_ground is True


# --- Review ronda 4: aterrizaje sobre la malla con dt no dyádico (#1, #3) y estabilidad (#2) ---

GRID_DTS = [0.01, 0.05, 0.1, 0.2, 0.3]


@pytest.mark.parametrize("dt", GRID_DTS)
@pytest.mark.parametrize("p0, v0", [([0.0, 5.0], [1.0, 0.0]), ([0.0, 0.0], [1.0, 10.0])])
def test_grid_landing_with_non_dyadic_dt_has_no_duplicate_sample(
    dt: float, p0: list[float], v0: list[float]
) -> None:
    # Caída/vuelo en el vacío que toca el suelo en t = 1 s o 2 s: con dt = 0,1 el residuo de
    # redondeo (+5e-16) no debe duplicar la última muestra.
    tr = projectile_with_drag(p0, v0, dt=dt, **NO_DRAG)
    free = projectile(p0, v0, dt=dt, g=10.0)
    assert tr.hit_ground is True
    assert np.all(np.diff(tr.t) > 0.0)
    assert tr.t.shape == free.t.shape
    assert tr.position[-1, -1] == 0.0
    assert tr.time_of_flight == pytest.approx(free.time_of_flight, abs=1e-9)


@pytest.mark.parametrize("dt", GRID_DTS)
def test_landing_at_t_max_with_non_dyadic_dt_counts_as_impact(dt: float) -> None:
    tr = projectile_with_drag([0.0, 5.0], [1.0, 0.0], dt=dt, t_max=1.0, **NO_DRAG)
    assert tr.hit_ground is True
    assert tr.position[-1, -1] == 0.0
    assert np.all(np.diff(tr.t) > 0.0)


LIGHT: dict[str, Any] = {"mass": 0.001, "rho": 1.2, "cd": 1.0, "area": 0.01, "t_max": 600.0}


@pytest.mark.parametrize("dt", [0.2, 0.22, 0.25])
def test_stiff_dt_is_rejected_even_when_computed_speed_is_low(dt: float) -> None:
    # k = 6, v_t = 1,28 m/s: 2·k·v_t·dt > 2,78 aunque la |v| calculada sea baja (review #2).
    with pytest.raises(IntegrationError, match="dt"):
        projectile_with_drag([0.0, 100.0], [0.0, 0.0], dt=dt, **LIGHT)


def test_light_body_with_fine_enough_dt_is_accepted_and_correct() -> None:
    coarse = projectile_with_drag([0.0, 100.0], [0.0, 0.0], dt=0.15, **LIGHT)
    fine = projectile_with_drag([0.0, 100.0], [0.0, 0.0], dt=0.01, **LIGHT)
    assert coarse.hit_ground is True
    assert coarse.time_of_flight == pytest.approx(fine.time_of_flight, rel=1e-3)


def test_max_height_is_never_negative_for_a_tiny_vertical_speed() -> None:
    tr = projectile_with_drag([0.0, 0.0], [1.0, 1e-20], dt=0.01, **{**NO_DRAG, "rho": 1.0})
    assert tr.max_height >= 0.0
