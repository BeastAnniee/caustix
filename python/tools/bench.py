"""Benchmark manual (no es parte del gate): tiempo de 10^4 pasos RK4 con arrastre.

Objetivo informativo: < 1 s en un portátil actual. Uso (desde ``python/``)::

    python tools/bench.py
"""

from __future__ import annotations

import time

from caustix import projectile_with_drag

STEPS = 10_000
DT = 1e-3
REPEATS = 5


def run_once() -> tuple[float, int]:
    """Ejecuta una simulación de ``STEPS`` pasos sin impacto y devuelve (segundos, muestras)."""
    start = time.perf_counter()
    trajectory = projectile_with_drag(
        [0.0, 1.0e6],
        [30.0, 40.0],
        mass=1.0,
        rho=1.225,
        cd=0.47,
        area=0.01,
        dt=DT,
        t_max=STEPS * DT,
    )
    elapsed = time.perf_counter() - start
    return elapsed, int(trajectory.t.shape[0])


def main() -> int:
    """Imprime el mejor y el peor tiempo de ``REPEATS`` ejecuciones."""
    times = []
    samples = 0
    for _ in range(REPEATS):
        elapsed, samples = run_once()
        times.append(elapsed)
    print(
        f"{samples - 1} pasos RK4 con arrastre: mejor {min(times):.3f} s, peor {max(times):.3f} s"
    )
    print("objetivo informativo: < 1 s (no es gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
