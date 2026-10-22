"""Contraste de projectile_with_drag contra las fixtures del validador Fortran (AC-20).

No necesita gfortran: lee las fixtures versionadas (regenerarlas: ``fortran/tools/regen.py``).
"""

import copy

import numpy as np
import pytest

from caustix import projectile_with_drag

from ._fixtures import FortranCase, load_cases

RTOL = 1e-9
ATOL = 1e-12
CASES = load_cases()


def compare(case: FortranCase) -> None:
    tr = projectile_with_drag(
        case.p0,
        case.v0,
        mass=case.mass,
        rho=case.rho,
        cd=case.cd,
        area=case.area,
        dt=case.dt,
        g=case.g,
        ground=case.ground,
        t_max=case.t_max,
    )
    assert tr.hit_ground is case.hit
    assert tr.t.shape[0] == case.n_samples
    idx = np.array(case.indices)
    np.testing.assert_allclose(tr.t[idx], case.t, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(tr.position[idx], case.position, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(tr.velocity[idx], case.velocity, rtol=RTOL, atol=ATOL)
    assert tr.time_of_flight == pytest.approx(case.t_end, rel=RTOL)
    assert tr.max_height == pytest.approx(case.max_height, rel=RTOL)


def test_fixture_has_expected_cases() -> None:
    names = {c.name for c in CASES}
    assert {"oblique2d_dt1e-2", "oblique2d_dt1e-3", "oblique3d_dt1e-2", "oblique3d_dt1e-3"} <= names
    assert {"coarse_dt0.21_ground", "coarse_dt0.5_ground"} <= names
    assert {"fall_dt1e-2", "fall_dt1e-3"} <= names
    assert {"exact_landing_dt0.5", "exact_landing_dt1", "exact_ascent_dt0.5"} <= names
    assert "exact_landing_tmax_dt0.5" in names
    assert all(c.indices[-1] == c.n_samples - 1 for c in CASES)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_python_matches_fortran(case: FortranCase) -> None:
    compare(case)


@pytest.mark.parametrize("field", ["position", "velocity", "t"])
def test_comparison_detects_altered_fixture(field: str) -> None:
    case = copy.deepcopy(next(c for c in CASES if c.name == "oblique2d_dt1e-2"))
    if field == "t":
        case.t[3] *= 1.0 + 1e-6
    else:
        getattr(case, field)[3][0] *= 1.0 + 1e-6
    with pytest.raises(AssertionError):
        compare(case)


def test_comparison_detects_altered_summary_values() -> None:
    case = copy.deepcopy(CASES[0])
    case.max_height *= 1.0 + 1e-6
    with pytest.raises(AssertionError):
        compare(case)
