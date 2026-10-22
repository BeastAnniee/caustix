# caustix

Librería de física en Python: **cinemática y trayectorias** — MRU, MRUA y tiro parabólico con y sin
resistencia del aire (RK4). Para estudiantes, docentes e ingenieros que necesitan resultados
verificables sin un stack científico pesado: solo depende de NumPy.

Todo en unidades SI (m, s, kg, m/s, m/s²). Las funciones son puras, devuelven resultados inmutables y
fallan de forma explícita ante entradas inválidas (`ValueError`) o integraciones inestables
(`IntegrationError`).

## Instalación

```bash
pip install caustix          # solo NumPy
pip install caustix[plot]    # con matplotlib para plot_trajectory
```

Requiere Python ≥ 3.10. Licencia MIT.

## Uso

Cinemática analítica (MRU y MRUA):

```python
>>> from caustix import mru, mrua
>>> mru([0.0, 0.0], [3.0, 4.0], 2.0).position.tolist()
[6.0, 8.0]
>>> mrua([0.0], [0.0], [2.0], [0.0, 1.0, 2.0]).position.tolist()
[[0.0], [1.0], [4.0]]

```

Tiro parabólico sin arrastre (solución analítica; el eje vertical es el último; ángulos en radianes):

```python
>>> import math
>>> from caustix import launch_velocity, projectile
>>> v0 = launch_velocity(30.0, math.radians(45.0))
>>> tr = projectile([0.0, 0.0], v0, dt=0.01)
>>> round(tr.range, 2), round(tr.max_height, 2), round(tr.time_of_flight, 3)
(91.77, 22.94, 4.326)

```

Tiro con arrastre cuadrático (RK4 de paso fijo; `rho` es obligatorio):

```python
>>> from caustix import projectile_with_drag
>>> d = projectile_with_drag(
...     [0.0, 0.0], v0, mass=0.145, rho=1.225, cd=0.47, area=0.0043, dt=0.01
... )
>>> round(d.range, 2), round(d.max_height, 2), bool(d.hit_ground)
(58.93, 17.72, True)
>>> e = d.mechanical_energy(0.145)   # energía mecánica por muestra (J)
>>> bool(e[-1] < e[0])
True

```

Las trayectorias llevan `t`, `position`, `velocity` (arrays de solo lectura), `time_of_flight`, `range`,
`max_height`, `hit_ground` y `g`, más `kinetic_energy`, `potential_energy` y `mechanical_energy`. El
impacto y el ápice se localizan por interpolación dentro del paso, y la última muestra queda exactamente
en el suelo. Si se alcanza `t_max` (600 s por defecto) sin impacto, `hit_ground` es `False`.

## Elegir `dt`

El RK4 tiene error global O(dt⁴). En el caso de referencia (caída de 100 m con arrastre,
`m = 1 kg`, `ρ = 1,225`, `C_d = 0,47`, `A = 0,01 m²`) el error del tiempo de impacto es ≈ 3·10⁻⁷ s con
`dt = 0,4 s` y ≈ 1·10⁻¹⁰ s con `dt = 0,05 s`. Para uso general, `dt = 0,01 s` es una buena opción de
partida; si `dt` es demasiado grande para la dinámica (`2·k·max(|v|, v_t)·dt > 2,78`, con `k = ρ·C_d·A/(2m)` y `v_t = √(g/k)` la velocidad terminal; es el
límite de estabilidad de RK4), se lanza `IntegrationError` en lugar de devolver una trayectoria absurda. `t_max / dt` no puede superar `max_steps` (1 000 000 por defecto): el límite se aplica a priori sobre `t_max`, así que con `dt` muy pequeño
baja `t_max` (p. ej. al doble del tiempo de vuelo esperado) o sube `max_steps`.

## API pública (1.0)

`mru`, `mrua`, `launch_velocity`, `projectile`, `projectile_with_drag`, `plot_trajectory`, `Motion`,
`Trajectory`, `IntegrationError`. La versión está en `caustix.__version__`.

## Verificación

El rigor se comprueba con tests contra soluciones analíticas, conservación de energía y orden de
convergencia del integrador. Además, el repositorio incluye un validador independiente en Fortran
(`fortran/`) que genera casos de referencia con los que se contrasta el RK4 de Python. **Es solo una
herramienta de desarrollo: no se publica y usar `caustix` no requiere compilador Fortran.**
