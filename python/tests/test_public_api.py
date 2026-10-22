"""Superficie pública congelada."""

import inspect

import caustix

EXPECTED_ALL = [
    "IntegrationError",
    "Motion",
    "Trajectory",
    "launch_velocity",
    "mru",
    "mrua",
    "plot_trajectory",
    "projectile",
    "projectile_with_drag",
]


def test_all_matches_documented_list_exactly() -> None:
    assert sorted(caustix.__all__) == EXPECTED_ALL


def test_every_exported_name_exists() -> None:
    for name in caustix.__all__:
        assert hasattr(caustix, name), name


def test_version_is_available_but_not_exported_symbol() -> None:
    assert caustix.__version__ == "1.0.0"


def test_py_typed_marker_is_shipped() -> None:
    from importlib.resources import files

    assert files("caustix").joinpath("py.typed").is_file()


def test_documented_signatures() -> None:
    sig = inspect.signature(caustix.projectile_with_drag)
    params = sig.parameters
    assert list(params)[:2] == ["position0", "velocity0"]
    keyword_only = [n for n, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY]
    assert keyword_only == [
        "mass", "rho", "cd", "area", "dt", "g", "ground", "t_max", "max_steps",
    ]  # fmt: skip
    assert params["rho"].default is inspect.Parameter.empty
    assert params["g"].default == 9.80665
    assert params["t_max"].default == 600.0
    assert params["max_steps"].default == 1_000_000
    free = inspect.signature(caustix.projectile).parameters
    assert [n for n, p in free.items() if p.kind is inspect.Parameter.KEYWORD_ONLY] == [
        "dt", "g", "ground",
    ]  # fmt: skip
    launch = inspect.signature(caustix.launch_velocity).parameters
    assert launch["azimuth"].default == 0.0
    assert launch["dim"].kind is inspect.Parameter.KEYWORD_ONLY
    assert launch["dim"].default == 2
