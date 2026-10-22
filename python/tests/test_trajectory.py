"""Trajectory: inmutabilidad, energías y derivados (AC-16)."""

import dataclasses
import math

import numpy as np
import pytest

from caustix import projectile
from caustix.results import Trajectory, build_trajectory

G = 9.80665


def _traj() -> Trajectory:
    t = np.array([0.0, 1.0, 2.0])
    position = np.array([[0.0, 10.0], [3.0, 12.0], [6.0, 0.0]])
    velocity = np.array([[3.0, 4.0], [3.0, 0.0], [3.0, -4.0]])
    return build_trajectory(t, position, velocity, g=G, ground=0.0, hit_ground=True)


def test_derived_fields() -> None:
    tr = _traj()
    assert tr.time_of_flight == 2.0
    assert tr.range == pytest.approx(6.0)
    assert tr.max_height == pytest.approx(12.0)
    assert tr.hit_ground is True
    assert tr.g == G


def test_max_height_relative_to_ground_and_explicit_override() -> None:
    t = np.array([0.0, 1.0])
    position = np.array([[0.0, 5.0], [1.0, 3.0]])
    velocity = np.zeros((2, 2))
    tr = build_trajectory(t, position, velocity, g=G, ground=2.0, hit_ground=False)
    assert tr.max_height == pytest.approx(3.0)
    tr2 = build_trajectory(
        t, position, velocity, g=G, ground=2.0, hit_ground=False, max_height=4.25
    )
    assert tr2.max_height == 4.25


def test_range_3d_uses_horizontal_plane() -> None:
    t = np.array([0.0, 1.0])
    position = np.array([[1.0, 1.0, 5.0], [4.0, 5.0, 0.0]])
    tr = build_trajectory(t, position, np.zeros((2, 3)), g=G, ground=0.0, hit_ground=True)
    assert tr.range == pytest.approx(5.0)


def test_energies_per_sample() -> None:
    tr = _traj()
    m = 2.0
    ke = tr.kinetic_energy(m)
    pe = tr.potential_energy(m)
    np.testing.assert_allclose(ke, [0.5 * m * 25.0, 0.5 * m * 9.0, 0.5 * m * 25.0])
    np.testing.assert_allclose(pe, m * G * np.array([10.0, 12.0, 0.0]))
    np.testing.assert_allclose(tr.mechanical_energy(m), ke + pe)
    assert ke.shape == (3,)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_energy_requires_positive_mass(bad: float) -> None:
    with pytest.raises(ValueError, match="mass"):
        _traj().kinetic_energy(bad)
    with pytest.raises(ValueError, match="mass"):
        _traj().mechanical_energy(bad)


def test_immutability() -> None:
    tr = _traj()
    with pytest.raises(dataclasses.FrozenInstanceError):
        tr.range = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        tr.position = np.zeros((1, 2))  # type: ignore[misc]
    for arr in (tr.t, tr.position, tr.velocity):
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        tr.position[0, 0] = 1.0


def test_build_does_not_alias_inputs() -> None:
    t = np.array([0.0, 1.0])
    position = np.array([[0.0, 1.0], [1.0, 0.0]])
    velocity = np.zeros((2, 2))
    tr = build_trajectory(t, position, velocity, g=G, ground=0.0, hit_ground=True)
    position[0, 0] = 99.0
    assert tr.position[0, 0] == 0.0


def test_results_compare_by_identity_and_are_hashable() -> None:
    # Con campos ndarray, ``==`` por campos lanzaría ValueError y ``hash`` TypeError.
    a = projectile([0.0, 5.0], [1.0, 0.0], dt=0.1)
    b = projectile([0.0, 5.0], [1.0, 0.0], dt=0.1)
    assert a == a
    assert a != b
    assert len({a, b}) == 2
