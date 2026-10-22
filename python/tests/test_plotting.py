"""Graficado opcional de trayectorias."""

import subprocess
import sys
from typing import Any

import numpy as np
import pytest

import caustix
from caustix import plot_trajectory, projectile


def _trajectory(dim: int = 2) -> caustix.Trajectory:
    v0 = [10.0, 20.0] if dim == 2 else [10.0, 5.0, 20.0]
    return projectile([0.0] * dim, v0, dt=0.1)


def test_without_matplotlib_numeric_api_works_and_plot_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    assert caustix.mru([0.0], [1.0], 2.0).position.tolist() == [2.0]
    tr = _trajectory()
    with pytest.raises(ImportError, match=r"pip install caustix\[plot\]"):
        plot_trajectory(tr)


def test_import_and_numeric_api_in_clean_interpreter_without_matplotlib() -> None:
    code = (
        "import sys; sys.modules['matplotlib'] = None; sys.modules['matplotlib.pyplot'] = None\n"
        "import caustix\n"
        "assert caustix.projectile([0.0, 0.0], [1.0, 1.0], dt=0.1).hit_ground\n"
        "try:\n"
        "    caustix.plot_trajectory(caustix.projectile([0.0, 0.0], [1.0, 1.0], dt=0.1))\n"
        "except ImportError as exc:\n"
        "    assert 'pip install caustix[plot]' in str(exc)\n"
        "else:\n"
        "    raise SystemExit('no ImportError')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_plot_2d_returns_axes_with_trajectory_line() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    import matplotlib

    matplotlib.use("Agg")
    tr = _trajectory()
    ax = plot_trajectory(tr)
    try:
        assert len(ax.lines) == 1
        x, y = ax.lines[0].get_data()
        np.testing.assert_array_equal(x, tr.position[:, 0])
        np.testing.assert_array_equal(y, tr.position[:, 1])
        assert ax.get_xlabel() and ax.get_ylabel()
    finally:
        plt.close("all")


def test_plot_2d_reuses_given_axes() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    import matplotlib

    matplotlib.use("Agg")
    _, ax = plt.subplots()
    try:
        assert plot_trajectory(_trajectory(), ax=ax) is ax
        assert len(ax.lines) == 1
    finally:
        plt.close("all")


def test_plot_3d_creates_3d_axes() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    import matplotlib

    matplotlib.use("Agg")
    tr = _trajectory(3)
    ax = plot_trajectory(tr)
    try:
        assert ax.name == "3d"
        assert len(ax.lines) == 1
        line: Any = ax.lines[0]
        x, y, z = line.get_data_3d()
        np.testing.assert_array_equal(x, tr.position[:, 0])
        np.testing.assert_array_equal(y, tr.position[:, 1])
        np.testing.assert_array_equal(z, tr.position[:, 2])
    finally:
        plt.close("all")


def test_plot_3d_rejects_flat_axes() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    import matplotlib

    matplotlib.use("Agg")
    _, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="ax"):
            plot_trajectory(_trajectory(3), ax=ax)
    finally:
        plt.close("all")
