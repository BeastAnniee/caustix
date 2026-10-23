# caustix

[![CI](https://github.com/BeastAnniee/caustix/actions/workflows/ci.yml/badge.svg)](https://github.com/BeastAnniee/caustix/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/caustix)](https://pypi.org/project/caustix/)
[![Python](https://img.shields.io/pypi/pyversions/caustix)](https://pypi.org/project/caustix/)
[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)

Librería de física en Python: **cinemática y trayectorias** — MRU, MRUA y tiro parabólico con y sin
resistencia del aire (RK4). Para estudiantes, docentes e ingenieros que necesitan resultados
verificables sin un stack científico pesado: solo depende de NumPy.

## Instalación

```bash
pip install caustix          # solo NumPy
pip install caustix[plot]    # con matplotlib para plot_trajectory
```

Requiere Python ≥ 3.10.

## Ejemplo

```python
import math
from caustix import launch_velocity, projectile

v0 = launch_velocity(30.0, math.radians(45.0))
tr = projectile([0.0, 0.0], v0, dt=0.01)
print(round(tr.range, 2), round(tr.max_height, 2), round(tr.time_of_flight, 3))
# 91.77 22.94 4.326
```

La guía de uso completa (MRU, MRUA, tiro con arrastre, cómo elegir `dt` y la API pública) está en
[python/README.md](python/README.md).

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| [`python/`](python/) | El paquete `caustix`, sus tests y las herramientas de empaquetado. |
| [`fortran/`](fortran/) | Validador independiente del RK4, solo para desarrollo. No se publica. |

## Desarrollo

Desde `python/`:

```bash
python -m pip install -e . --group dev
python -m ruff check . && python -m mypy . && python -m pytest -q
```

## Licencia

[MIT](LICENSE).
