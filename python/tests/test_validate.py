"""Validación en el borde."""

import math

import numpy as np
import pytest

from caustix._validate import (
    as_vector,
    check_same_dim,
    finite_scalar,
    non_negative,
    positive,
)
from caustix.errors import IntegrationError


def test_integration_error_is_arithmetic_error() -> None:
    assert issubclass(IntegrationError, ArithmeticError)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_finite_scalar_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match=r"g.*" + str(bad).replace("-", r"\-")):
        finite_scalar("g", bad)


def test_finite_scalar_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="g"):
        finite_scalar("g", "abc")
    with pytest.raises(ValueError, match="g"):
        finite_scalar("g", np.array([1.0, 2.0]))


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_positive_rejects(bad: float) -> None:
    with pytest.raises(ValueError, match="mass"):
        positive("mass", bad)


@pytest.mark.parametrize("bad", [-1e-9, math.nan, -math.inf, math.inf])
def test_non_negative_rejects(bad: float) -> None:
    with pytest.raises(ValueError, match="rho"):
        non_negative("rho", bad)


def test_scalar_helpers_accept_and_return_float() -> None:
    assert positive("dt", 2) == 2.0
    assert isinstance(positive("dt", 2), float)
    assert non_negative("cd", 0.0) == 0.0
    assert finite_scalar("z", -3) == -3.0


def test_message_contains_value() -> None:
    with pytest.raises(ValueError, match=r"-2\.5"):
        positive("mass", -2.5)


@pytest.mark.parametrize(
    "bad",
    [[math.nan, 0.0], [math.inf], [0.0, -math.inf], [], [1, 2, 3, 4], [[1.0, 2.0]], 5.0, "ab"],
)
def test_as_vector_rejects(bad: object) -> None:
    with pytest.raises(ValueError, match="x0"):
        as_vector("x0", bad)


def test_as_vector_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="x0"):
        as_vector("x0", [1.0, 2.0], dim=3)


def test_as_vector_returns_float64_copy_without_mutating_input() -> None:
    original = np.array([1, 2, 3])
    before = original.copy()
    out = as_vector("x0", original)
    assert out.dtype == np.float64
    assert out.shape == (3,)
    out[0] = 99.0
    np.testing.assert_array_equal(original, before)
    assert not np.shares_memory(out, original)


def test_as_vector_accepts_list_and_tuple() -> None:
    np.testing.assert_array_equal(as_vector("v", [1.0, 2.0]), [1.0, 2.0])
    np.testing.assert_array_equal(as_vector("v", (1.0,)), [1.0])


def test_check_same_dim_ok_returns_dim() -> None:
    assert check_same_dim(x0=np.zeros(2), v=np.ones(2)) == 2


def test_check_same_dim_mismatch_names_parameters() -> None:
    with pytest.raises(ValueError, match=r"x0.*v|v.*x0"):
        check_same_dim(x0=np.zeros(2), v=np.ones(3))


@pytest.mark.parametrize("flag", [True, False, np.True_])
def test_finite_scalar_rejects_booleans(flag: object) -> None:
    with pytest.raises(ValueError, match="mass"):
        finite_scalar("mass", flag)


@pytest.mark.parametrize(
    "bad", [np.array([1 + 2j]), [True, False], ["1", "2"], np.array([True]), [1 + 0j]]
)
def test_as_vector_rejects_complex_bool_and_text(bad: object) -> None:
    with pytest.raises(ValueError, match="x0"):
        as_vector("x0", bad)


@pytest.mark.parametrize("bad", ["1.0", 1 + 2j, np.complex128(1)])
def test_finite_scalar_rejects_text_and_complex(bad: object) -> None:
    with pytest.raises(ValueError, match="mass"):
        finite_scalar("mass", bad)
