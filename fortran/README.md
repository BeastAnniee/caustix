# Validador Fortran (solo desarrollo)

Implementación **independiente** del RK4 con arrastre cuadrático y del evento de impacto/ápice de
`caustix`. Sirve para contrastar el código Python (REQ-20 a REQ-22; ADR-0005). **No se publica**:
nada de este directorio entra en el sdist ni en la wheel, e instalar o usar `caustix` no requiere
`gfortran`.

| Archivo | Qué hace |
|---|---|
| `src/caustix_rk4.f90` | Módulo con el RK4, la interpolación de Hermite y la bisección acotada (60 iteraciones). |
| `app/verify.f90` | Comprueba la caída con arrastre contra su solución analítica y el orden de convergencia (AC-5, AC-6, AC-9). |
| `app/gen_fixtures.f90` | Lee `cases.txt` y escribe la fixture CSV (17 cifras significativas). |
| `cases.txt` | Casos de referencia (oblicuo 2D y 3D, caída; `dt` = 1e-2 y 1e-3). |
| `tools/regen.py` | Compila (`gfortran -std=f2008 -O2 -ffp-contract=off -Wall -Werror`), ejecuta `verify`, regenera y compara. |

La fixture versionada vive en `python/tests/fixtures/fortran_rk4.csv`; los tests de Python la leen sin
necesitar `gfortran` (`python/tests/test_fortran_crosscheck.py`).

## Comandos (desde `fortran/`)

```bash
python tools/regen.py --check   # compila, verifica y compara con la fixture (rtol 1e-12); por defecto
python tools/regen.py --write   # sobrescribe la fixture versionada
```

Si falta `gfortran`, el script termina con código 2 y un mensaje claro: nunca se omite en silencio.
Quien no tenga `gfortran` puede ejecutar el gate solo del paquete con
`node scripts/tc-verify.mjs --component=app`.

## Regenerar las fixtures

Solo cuando cambie el algoritmo o `cases.txt`. **Revisa el diff de la fixture** antes de aceptarla: si
Python y Fortran divergen, hay que entender la causa; no se acepta una fixture nueva para «hacer pasar»
el contraste.
