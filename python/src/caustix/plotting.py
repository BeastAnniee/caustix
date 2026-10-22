"""Graficado opcional de trayectorias (requiere el extra ``caustix[plot]``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from caustix.results import Trajectory

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_MISSING = "plot_trajectory requiere matplotlib: pip install caustix[plot]"


def plot_trajectory(trajectory: Trajectory, ax: Axes | None = None) -> Axes:
    """Dibuja una trayectoria en 2D (x–z) o en 3D.

    matplotlib se importa de forma perezosa: el resto de la librería funciona sin él.

    Args:
        trajectory: Trayectoria a dibujar (posiciones en m).
        ax: Ejes donde dibujar. Si es ``None`` se crea una figura nueva (con proyección
            ``3d`` cuando la trayectoria es 3D).

    Returns:
        Los ejes de matplotlib con la línea de la trayectoria.

    Raises:
        ImportError: Si matplotlib no está instalado (``pip install caustix[plot]``).
        ValueError: Si la trayectoria es 3D y ``ax`` no tiene proyección ``3d``.

    Example:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from caustix import projectile
        >>> ax = plot_trajectory(projectile([0.0, 0.0], [3.0, 4.0], dt=0.1))
        >>> len(ax.lines)
        1
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(_MISSING) from exc

    position = trajectory.position
    is_3d = position.shape[1] == 3
    if ax is None:
        figure = plt.figure()
        ax = figure.add_subplot(projection="3d") if is_3d else figure.add_subplot()
    elif is_3d and getattr(ax, "name", "") != "3d":
        raise ValueError("ax debe tener proyección '3d' para una trayectoria 3D")

    if is_3d:
        ax.plot(position[:, 0], position[:, 1], position[:, 2])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")  # type: ignore[union-attr]
    else:
        ax.plot(position[:, 0], position[:, 1])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
    return ax
