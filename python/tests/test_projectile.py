"""Tiro parabólico sin arrastre contra fórmulas cerradas (AC-3, AC-4, AC-10)."""

import math

import numpy as np
import pytest

from caustix import launch_velocity, projectile

G = 9.80665
ANGLES = [0.0, 30.0, 45.0, 60.0, 90.0]


def _expected(
    v0: np.ndarray, p0: np.ndarray, g: float, ground: float
) -> tuple[float, float, float]:
    vz, z0 = v0[-1], p0[-1]
    t_f = (vz + math.sqrt(vz**2 + 2 * g * (z0 - ground))) / g
    horizontal = float(np.linalg.norm(v0[:-1])) * t_f
    z_apex = z0 + vz**2 / (2 * g) if vz > 0 else z0
    return t_f, horizontal, z_apex - ground


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize(("z0", "ground"), [(0.0, 0.0), (5.0, 0.0), (5.0, 2.0)])
@pytest.mark.parametrize("dim", [2, 3])
def test_matches_closed_form(angle: float, z0: float, ground: float, dim: int) -> None:
    v0 = launch_velocity(20.0, math.radians(angle), math.radians(25.0), dim=dim)
    p0 = np.zeros(dim)
    p0[-1] = z0
    tr = projectile(p0, v0, dt=0.05, ground=ground)
    t_f, rng, h = _expected(v0, p0, G, ground)

    assert tr.hit_ground is True
    assert tr.time_of_flight == pytest.approx(t_f, rel=1e-12)
    assert tr.range == pytest.approx(rng, rel=1e-12, abs=1e-12)
    assert tr.max_height == pytest.approx(h, rel=1e-12, abs=1e-12)
    assert tr.t[0] == 0.0
    assert tr.t[-1] == pytest.approx(t_f, rel=1e-12)
    assert tr.position[-1, -1] == ground  # última muestra exactamente en el suelo
    # muestras intermedias: fórmulas cerradas
    t = tr.t[:, None]
    gvec = np.zeros(dim)
    gvec[-1] = -G
    np.testing.assert_allclose(tr.position, p0 + v0 * t + 0.5 * gvec * t**2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tr.velocity, v0 + gvec * t, rtol=1e-12, atol=1e-12)


def test_samples_use_dt_and_last_is_impact() -> None:
    tr = projectile([0.0, 0.0], [10.0, 10.0], dt=0.25)
    assert np.allclose(np.diff(tr.t)[:-1], 0.25)
    assert 0.0 < np.diff(tr.t)[-1] <= 0.25 + 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_energy_conservation(dim: int) -> None:
    v0 = launch_velocity(35.0, math.radians(50.0), math.radians(10.0), dim=dim)
    p0 = np.zeros(dim)
    p0[-1] = 3.0
    tr = projectile(p0, v0, dt=0.01)
    m = 1.7
    e = tr.mechanical_energy(m)
    assert np.max(np.abs(e - e[0])) / abs(e[0]) <= 1e-12


def test_starts_on_ground_going_down_is_single_sample() -> None:
    tr = projectile([0.0, 0.0], [3.0, -1.0], dt=0.1)
    assert tr.t.shape == (1,)
    assert tr.time_of_flight == 0.0
    assert tr.hit_ground is True
    assert tr.range == 0.0
    assert tr.max_height == 0.0


def test_starts_on_ground_going_up_flies() -> None:
    tr = projectile([0.0, 0.0], [3.0, 1.0], dt=0.1)
    assert tr.time_of_flight == pytest.approx(2.0 / G, rel=1e-12)


def test_arguments_not_mutated_and_deterministic() -> None:
    p0 = np.array([0.0, 1.0])
    v0 = np.array([4.0, 5.0])
    a = projectile(p0, v0, dt=0.1)
    b = projectile(p0, v0, dt=0.1)
    np.testing.assert_array_equal(p0, [0.0, 1.0])
    np.testing.assert_array_equal(v0, [4.0, 5.0])
    assert a.position.tobytes() == b.position.tobytes()


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"dt": 0.0}, "dt"),
        ({"dt": -0.1}, "dt"),
        ({"dt": math.nan}, "dt"),
        ({"dt": 0.1, "g": 0.0}, "g"),
        ({"dt": 0.1, "g": -9.8}, "g"),
        ({"dt": 0.1, "g": math.inf}, "g"),
        ({"dt": 0.1, "ground": math.nan}, "ground"),
        ({"dt": 0.1, "ground": 5.0}, "position0"),  # y0 < suelo
    ],
)
def test_invalid_scalars(kwargs: dict[str, float], name: str) -> None:
    with pytest.raises(ValueError, match=name):
        projectile([0.0, 1.0], [1.0, 1.0], **kwargs)


@pytest.mark.parametrize(
    ("p0", "v0", "name"),
    [
        ([0.0], [1.0], "position0"),  # dim < 2
        ([0.0, 0.0], [1.0, 1.0, 1.0], "velocity0"),
        ([0.0, math.nan], [1.0, 1.0], "position0"),
        ([0.0, 0.0], [math.inf, 1.0], "velocity0"),
    ],
)
def test_invalid_vectors(p0: list[float], v0: list[float], name: str) -> None:
    with pytest.raises(ValueError, match=name):
        projectile(p0, v0, dt=0.1)


def test_launch_velocity_2d_and_3d() -> None:
    v2 = launch_velocity(10.0, math.radians(30.0))
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    np.testing.assert_allclose(v2, [10 * c, 10 * s])
    v3 = launch_velocity(10.0, math.radians(30.0), math.radians(90.0), dim=3)
    np.testing.assert_allclose(v3, [0.0, 10 * c, 10 * s], atol=1e-12)
    assert np.linalg.norm(v3) == pytest.approx(10.0, rel=1e-14)


@pytest.mark.parametrize(
    ("speed", "elev", "az", "dim", "name"),
    [
        (-1.0, 0.1, 0.0, 2, "speed"),
        (math.nan, 0.1, 0.0, 2, "speed"),
        (1.0, math.inf, 0.0, 2, "elevation"),
        (1.0, 0.1, math.nan, 3, "azimuth"),
        (1.0, 0.1, 0.0, 1, "dim"),
        (1.0, 0.1, 0.0, 4, "dim"),
    ],
)
def test_launch_velocity_invalid(speed: float, elev: float, az: float, dim: int, name: str) -> None:
    with pytest.raises(ValueError, match=name):
        launch_velocity(speed, elev, az, dim=dim)


@pytest.mark.parametrize(
    ("v0", "dt"),
    [
        ([1.0, 1e6], 1e-9),  # demasiadas muestras: antes MemoryError
        ([1.0, 1e308], 0.1),  # tiempo de vuelo desbordado: antes OverflowError
    ],
)
def test_sample_count_is_bounded_before_allocating(v0: list[float], dt: float) -> None:
    with pytest.raises(ValueError, match="muestras"):
        projectile([0.0, 0.0], v0, dt=dt)


def test_huge_dt_keeps_initial_sample() -> None:
    tr = projectile([0.0, 5.0], [0.0, 0.0], dt=1e13)
    assert tr.t[0] == 0.0
    assert tr.t[-1] == pytest.approx(tr.time_of_flight)
    assert tr.time_of_flight == pytest.approx(math.sqrt(2 * 5.0 / 9.80665))


@pytest.mark.parametrize("vz", [-1e100, -1e200, -1e300])
def test_huge_downward_speed_lands_almost_immediately(vz: float) -> None:
    # vz*vz desbordaba a inf y el tiempo de vuelo salía 0 con el cuerpo a 5 m del suelo.
    tr = projectile([0.0, 5.0], [0.0, vz], dt=0.1)
    assert tr.hit_ground is True
    assert 0.0 < tr.time_of_flight < 1e-90
    assert tr.position[-1, -1] == 0.0
    assert tr.t[-1] == tr.time_of_flight
