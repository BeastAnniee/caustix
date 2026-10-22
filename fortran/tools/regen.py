"""Compila el validador Fortran, lo ejecuta y comprueba o regenera las fixtures.

Uso (desde ``fortran/``)::

    python tools/regen.py --check   # por defecto: compila, verifica y compara con la fixture
    python tools/regen.py --write   # además sobrescribe la fixture versionada

Requiere ``gfortran``. Si falta, termina con error (código 2); nunca se omite en silencio.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT.parent / "python" / "tests" / "fixtures" / "fortran_rk4.csv"
CASES = ROOT / "cases.txt"
SOURCES = [ROOT / "src" / "caustix_rk4.f90"]
FLAGS = ["-std=f2008", "-O2", "-ffp-contract=off", "-Wall", "-Werror"]
RTOL = 1e-12


def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, **kwargs)  # type: ignore[call-overload]
    except OSError as exc:
        sys.stderr.write(
            f"ERROR: no se pudo ejecutar {Path(cmd[0]).name}: {exc}\n"
            "Si el sistema bloquea binarios recién compilados (p. ej. Control de aplicaciones "
            "de Windows), ejecuta esta comprobación en CI Linux o en otro equipo.\n"
        )
        raise SystemExit(1) from exc


def _build(gfortran: str, app: Path, out: Path) -> None:
    # -J: los .mod se generan junto al ejecutable (temporal), no en el árbol del repo.
    result = _run(
        [gfortran, *FLAGS, "-J", str(out.parent), "-o", str(out), *map(str, SOURCES), str(app)]
    )
    if result.returncode != 0:
        sys.stderr.write(f"Fallo al compilar {app.name}:\n{result.stdout}{result.stderr}\n")
        raise SystemExit(1)


def _compare(new: str, old: str) -> list[str]:
    problems: list[str] = []
    new_lines, old_lines = new.splitlines(), old.splitlines()
    if len(new_lines) != len(old_lines):
        return [f"número de líneas distinto: regenerado={len(new_lines)} versionado={len(old_lines)}"]
    for number, (a, b) in enumerate(zip(new_lines, old_lines, strict=True), start=1):
        fa, fb = a.split(","), b.split(",")
        if len(fa) != len(fb):
            problems.append(f"línea {number}: número de columnas distinto")
            continue
        for column, (x, y) in enumerate(zip(fa, fb, strict=True), start=1):
            try:
                u, v = float(x), float(y)
            except ValueError:
                if x.strip() != y.strip():
                    problems.append(f"línea {number}, columna {column}: {x!r} != {y!r}")
                continue
            if not math.isclose(u, v, rel_tol=RTOL, abs_tol=1e-300):
                problems.append(f"línea {number}, columna {column}: {u!r} != {v!r} (rtol {RTOL})")
    return problems


def main() -> int:
    """Punto de entrada; devuelve el código de salida."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="comparar con la fixture (por defecto)")
    group.add_argument("--write", action="store_true", help="sobrescribir la fixture versionada")
    args = parser.parse_args()

    gfortran = shutil.which("gfortran")
    if gfortran is None:
        sys.stderr.write(
            "ERROR: no se encontró 'gfortran' en el PATH. Instálalo para regenerar o comprobar "
            "las fixtures (los tests de Python no lo necesitan: solo leen las fixtures).\n"
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="caustix-fortran-") as tmp:
        tmpdir = Path(tmp)
        verify = tmpdir / "verify.exe"
        gen = tmpdir / "gen_fixtures.exe"
        _build(gfortran, ROOT / "app" / "verify.f90", verify)
        _build(gfortran, ROOT / "app" / "gen_fixtures.f90", gen)

        checked = _run([str(verify)])
        sys.stdout.write(checked.stdout)
        if checked.returncode != 0:
            sys.stderr.write(f"La verificación analítica del validador falló:\n{checked.stderr}\n")
            return 1

        produced = tmpdir / "fortran_rk4.csv"
        made = _run([str(gen), str(CASES), str(produced)])
        if made.returncode != 0:
            sys.stderr.write(f"gen_fixtures falló:\n{made.stdout}{made.stderr}\n")
            return 1
        new_text = produced.read_text(encoding="utf-8")

        if args.write:
            FIXTURE.parent.mkdir(parents=True, exist_ok=True)
            FIXTURE.write_text(new_text, encoding="utf-8", newline="\n")
            sys.stdout.write(f"Fixture escrita: {FIXTURE}\n")
            return 0

        if not FIXTURE.exists():
            sys.stderr.write(f"Falta la fixture versionada: {FIXTURE} (ejecuta --write)\n")
            return 1
        problems = _compare(new_text, FIXTURE.read_text(encoding="utf-8"))
        if problems:
            sys.stderr.write("La fixture regenerada difiere de la versionada:\n")
            sys.stderr.write("\n".join(problems[:20]) + "\n")
            return 1
    sys.stdout.write("OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
