"""MRU y MRUA contra sus fórmulas cerradas."""

import dataclasses
import math

import numpy as np
import pytest

from caustix import Motion, mru, mrua

RNG_SEED = 12345
DIMS = [1, 2, 3]


def _vec(rng: np.random.Generator, dim: int) -> np.ndarray:
    return rng.uniform(-50.0, 50.0, size=dim)


@pytest.mark.parametrize("dim", DIMS)
def test_mru_scalar_time(dim: int) -> None:
    rng = np.random.default_rng(RNG_SEED)
    x0, v = _vec(rng, dim), _vec(rng, dim)
    t = 3.7
    m = mru(x0, v, t)
    assert isinstance(m, Motion)
    assert m.position.shape == (dim,)
    assert m.velocity.shape == (dim,)
    np.testing.assert_allclose(m.position, x0 + v * t, rtol=1e-12)
    np.testing.assert_allclose(m.velocity, v, rtol=1e-12)


@pytest.mark.parametrize("dim", DIMS)
def test_mru_array_time(dim: int) -> None:
    rng = np.random.default_rng(RNG_SEED)
    x0, v = _vec(rng, dim), _vec(rng, dim)
    t = np.linspace(-2.0, 10.0, 25)
    m = mru(x0, v, t)
    assert m.t.shape == (25,)
    assert m.position.shape == (25, dim)
    assert m.velocity.shape == (25, dim)
    np.testing.assert_allclose(m.position, x0 + np.outer(t, v), rtol=1e-12)
    np.testing.assert_allclose(m.velocity, np.tile(v, (25, 1)), rtol=1e-12)


@pytest.mark.parametrize("dim", DIMS)
def test_mrua_scalar_and_array_time(dim: int) -> None:
    rng = np.random.default_rng(RNG_SEED)
    x0, v0, a = _vec(rng, dim), _vec(rng, dim), _vec(rng, dim)
    ts = np.linspace(0.0, 8.0, 17)
    m = mrua(x0, v0, a, ts)
    assert m.position.shape == (17, dim)
    np.testing.assert_allclose(
        m.position, x0 + np.outer(ts, v0) + 0.5 * np.outer(ts**2, a), rtol=1e-12
    )
    np.testing.assert_allclose(m.velocity, v0 + np.outer(ts, a), rtol=1e-12)
    m1 = mrua(x0, v0, a, 2.5)
    assert m1.position.shape == (dim,)
    np.testing.assert_allclose(m1.position, x0 + v0 * 2.5 + 0.5 * a * 2.5**2, rtol=1e-12)
    np.testing.assert_allclose(m1.velocity, v0 + a * 2.5, rtol=1e-12)


def test_mrua_1d_torricelli_identity() -> None:
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(20):
        x0, v0, a = rng.uniform(-10, 10, 3)
        t = rng.uniform(0.1, 9.0)
        m = mrua([x0], [v0], [a], t)
        dx = m.position[0] - x0
        np.testing.assert_allclose(m.velocity[0] ** 2, v0**2 + 2 * a * dx, rtol=1e-12, atol=1e-12)


def test_results_are_immutable() -> None:
    m = mru([0.0, 0.0], [1.0, 2.0], np.arange(3.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.position = np.zeros(1)  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        m.position[0, 0] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        m.velocity[0, 0] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        m.t[0] = 5.0


def test_arguments_are_not_mutated() -> None:
    x0 = np.array([1.0, 2.0])
    v = np.array([3.0, 4.0])
    a = np.array([0.5, 0.5])
    t = np.array([0.0, 1.0, 2.0])
    copies = [arr.copy() for arr in (x0, v, a, t)]
    m = mrua(x0, v, a, t)
    for original, saved in zip((x0, v, a, t), copies, strict=True):
        np.testing.assert_array_equal(original, saved)
    assert not np.shares_memory(m.position, x0)
    assert not np.shares_memory(m.t, t)


def test_deterministic_bit_for_bit() -> None:
    args = ([1.0, 2.0, 3.0], [0.3, 0.2, 0.1], [0.0, -9.8, 0.0], np.linspace(0, 5, 11))
    a, b = mrua(*args), mrua(*args)
    assert a.position.tobytes() == b.position.tobytes()
    assert a.velocity.tobytes() == b.velocity.tobytes()


@pytest.mark.parametrize("bad_t", [math.nan, math.inf, [0.0, math.nan], [[0.0, 1.0]], [], "x"])
def test_invalid_time(bad_t: object) -> None:
    with pytest.raises(ValueError, match="t"):
        mru([0.0], [1.0], bad_t)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("x0", "v", "name"),
    [
        ([math.nan, 0.0], [1.0, 1.0], "x0"),
        ([0.0, 0.0], [math.inf, 1.0], "v"),
        ([0.0, 0.0], [1.0, 1.0, 1.0], "v"),
        ([], [], "x0"),
        ([0.0] * 4, [1.0] * 4, "x0"),
    ],
)
def test_mru_invalid_vectors(x0: list[float], v: list[float], name: str) -> None:
    with pytest.raises(ValueError, match=name):
        mru(x0, v, 1.0)


def test_mrua_invalid_acceleration() -> None:
    with pytest.raises(ValueError, match="a"):
        mrua([0.0, 0.0], [1.0, 1.0], [0.0, math.nan], 1.0)
    with pytest.raises(ValueError, match="a"):
        mrua([0.0, 0.0], [1.0, 1.0], [0.0], 1.0)


@pytest.mark.parametrize("bad", [np.array([1 + 2j]), ["1"], [True]])
def test_time_rejects_complex_text_and_bool(bad: object) -> None:
    with pytest.raises(ValueError, match="t "):
        mru([0.0], [1.0], bad)  # type: ignore[arg-type]


def test_mru_overflowing_result_raises_instead_of_returning_inf() -> None:
    with pytest.raises(OverflowError, match="mru"):
        mru([1e308], [1e308], [1.0, 2.0])


def test_mrua_overflowing_result_raises_instead_of_returning_inf() -> None:
    with pytest.raises(OverflowError, match="mrua"):
        mrua([0.0], [0.0], [1e308], 1e10)
