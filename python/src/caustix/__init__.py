"""caustix: cinemática y trayectorias (MRU, MRUA, tiro parabólico con y sin arrastre)."""

from caustix.drag import projectile_with_drag
from caustix.errors import IntegrationError
from caustix.kinematics import mru, mrua
from caustix.plotting import plot_trajectory
from caustix.projectile import launch_velocity, projectile
from caustix.results import Motion, Trajectory

__version__ = "1.0.0"

__all__ = [
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
