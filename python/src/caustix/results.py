"""Resultados inmutables devueltos por la API pública."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from caustix._validate import positive

FloatArray = NDArray[np.float64]


def freeze(array: FloatArray) -> FloatArray:
    """Devuelve una copia ``float64`` del array marcada como solo lectura."""
    frozen = np.array(array, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, eq=False)
class Motion:
    """Resultado de un movimiento rectilíneo (MRU o MRUA).

    Attributes:
        t: Instantes evaluados en s; forma ``()`` para un ``t`` escalar o ``(N,)``.
        position: Posición en m; forma ``(dim,)`` o ``(N, dim)``.
        velocity: Velocidad en m/s; misma forma que ``position``.

    Example:
        >>> import numpy as np
        >>> m = Motion(t=np.array(1.0), position=np.array([2.0]), velocity=np.array([3.0]))
        >>> m.position.flags.writeable
        False
    """

    t: FloatArray
    position: FloatArray
    velocity: FloatArray

    def __post_init__(self) -> None:
        """Congela los arrays: copias de solo lectura."""
        object.__setattr__(self, "t", freeze(self.t))
        object.__setattr__(self, "position", freeze(self.position))
        object.__setattr__(self, "velocity", freeze(self.velocity))


@dataclass(frozen=True, eq=False)
class Trajectory:
    """Trayectoria de un tiro parabólico (con o sin arrastre).

    El eje vertical es el último eje de ``position``; la gravedad apunta a su sentido
    negativo. Los arrays son de solo lectura.

    Attributes:
        t: Instantes de las muestras en s, forma ``(N,)``.
        position: Posición en m, forma ``(N, dim)``.
        velocity: Velocidad en m/s, forma ``(N, dim)``.
        time_of_flight: Duración del vuelo en s (``t[-1] - t[0]``).
        range: Distancia horizontal en m entre la primera y la última muestra.
        max_height: Altura máxima en m sobre el nivel del suelo.
        hit_ground: ``True`` si la trayectoria termina en el suelo; ``False`` si
            se alcanzó ``t_max`` sin impacto.
        g: Aceleración de la gravedad en m/s² usada en la simulación.

    Example:
        >>> import numpy as np
        >>> tr = build_trajectory(
        ...     np.array([0.0, 2.0]),
        ...     np.array([[0.0, 5.0], [6.0, 0.0]]),
        ...     np.array([[3.0, 0.0], [3.0, -9.0]]),
        ...     g=9.80665, ground=0.0, hit_ground=True,
        ... )
        >>> tr.range, tr.max_height, tr.hit_ground
        (6.0, 5.0, True)
    """

    t: FloatArray
    position: FloatArray
    velocity: FloatArray
    time_of_flight: float
    range: float
    max_height: float
    hit_ground: bool
    g: float

    def __post_init__(self) -> None:
        """Congela los arrays: copias de solo lectura."""
        object.__setattr__(self, "t", freeze(self.t))
        object.__setattr__(self, "position", freeze(self.position))
        object.__setattr__(self, "velocity", freeze(self.velocity))

    def kinetic_energy(self, mass: float) -> FloatArray:
        """Energía cinética por muestra.

        Ecuación: ``E_c = ½·m·|v|²``.

        Args:
            mass: Masa en kg, finita y > 0.

        Returns:
            Array de forma ``(N,)`` en J.

        Raises:
            ValueError: Si ``mass`` no es finita y positiva.

        Example:
            >>> import numpy as np
            >>> tr = build_trajectory(
            ...     np.array([0.0]), np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]),
            ...     g=9.80665, ground=0.0, hit_ground=True,
            ... )
            >>> tr.kinetic_energy(2.0).tolist()
            [25.0]
        """
        m = positive("mass", mass)
        speed_sq = np.sum(self.velocity * self.velocity, axis=-1)
        result: FloatArray = 0.5 * m * speed_sq
        return result

    def potential_energy(self, mass: float) -> FloatArray:
        """Energía potencial gravitatoria por muestra (gravedad uniforme).

        Ecuación: ``E_p = m·g·z`` con ``z`` la coordenada vertical absoluta.

        Args:
            mass: Masa en kg, finita y > 0.

        Returns:
            Array de forma ``(N,)`` en J.

        Raises:
            ValueError: Si ``mass`` no es finita y positiva.

        Example:
            >>> import numpy as np
            >>> tr = build_trajectory(
            ...     np.array([0.0]), np.array([[0.0, 10.0]]), np.zeros((1, 2)),
            ...     g=10.0, ground=0.0, hit_ground=True,
            ... )
            >>> tr.potential_energy(2.0).tolist()
            [200.0]
        """
        m = positive("mass", mass)
        result: FloatArray = m * self.g * self.position[:, -1]
        return result

    def mechanical_energy(self, mass: float) -> FloatArray:
        """Energía mecánica por muestra.

        Ecuación: ``E = E_c + E_p``.

        Args:
            mass: Masa en kg, finita y > 0.

        Returns:
            Array de forma ``(N,)`` en J.

        Raises:
            ValueError: Si ``mass`` no es finita y positiva.

        Example:
            >>> import numpy as np
            >>> tr = build_trajectory(
            ...     np.array([0.0]), np.array([[0.0, 10.0]]), np.array([[3.0, 4.0]]),
            ...     g=10.0, ground=0.0, hit_ground=True,
            ... )
            >>> tr.mechanical_energy(2.0).tolist()
            [225.0]
        """
        result: FloatArray = self.kinetic_energy(mass) + self.potential_energy(mass)
        return result


def build_trajectory(
    t: FloatArray,
    position: FloatArray,
    velocity: FloatArray,
    *,
    g: float,
    ground: float,
    hit_ground: bool,
    max_height: float | None = None,
) -> Trajectory:
    """Construye una ``Trajectory`` derivando tiempo de vuelo, alcance y altura máxima.

    Uso interno. Si ``max_height`` es ``None`` se toma el máximo de las muestras sobre
    ``ground``; los integradores con eventos pasan el valor interpolado.
    """
    horizontal = position[-1, :-1] - position[0, :-1]
    if max_height is None:
        max_height = float(np.max(position[:, -1]) - ground)
    return Trajectory(
        t=t,
        position=position,
        velocity=velocity,
        time_of_flight=float(t[-1] - t[0]),
        range=float(np.linalg.norm(horizontal)),
        max_height=float(max_height),
        hit_ground=hit_ground,
        g=float(g),
    )
