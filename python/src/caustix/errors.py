"""Excepciones públicas de caustix."""


class IntegrationError(ArithmeticError):
    """La integración numérica produjo un estado no finito (NaN o infinito).

    Se lanza en lugar de devolver una trayectoria con NaN o infinitos; suele indicar un
    ``dt`` demasiado grande o velocidades extremas. Las entradas inválidas lanzan
    ``ValueError``.

    Example:
        >>> issubclass(IntegrationError, ArithmeticError)
        True
    """
