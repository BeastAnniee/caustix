"""Verifica el empaquetado de caustix antes de publicar.

Construye sdist y wheel en un directorio temporal, ejecuta ``twine check``, inspecciona los
metadatos y el contenido, e instala la wheel en un venv limpio para importar ``caustix``.
Termina con código 1 si algo falla. Requiere red (instala NumPy en el venv limpio).

Uso (desde ``python/``)::

    python tools/check_dist.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from email.parser import Parser
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
NAME = "caustix"
VERSION = re.search(
    r'^__version__ = "([^"]+)"', (PROJECT_DIR / "src/caustix/__init__.py").read_text("utf-8"), re.M
)[1]  # type: ignore[index]
PYTHON_REQUIRES = ">=3.10"
FORTRAN_SUFFIXES = {".f", ".f90", ".f95", ".for", ".mod", ".o"}
SDIST_ALLOWED_FILES = {
    "PKG-INFO",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    ".gitignore",
}
SDIST_ALLOWED_DIRS = ("src/", "tests/", "tools/")
FORBIDDEN_NAMES = re.compile(r"(^|/)(\.env(\..*)?|.*\.(pem|key|pfx)|__pycache__|.*\.pyc)$")


def _requirement_parts(requirement: str) -> tuple[str, str]:
    """Separa un ``Requires-Dist`` en (nombre normalizado, marcador sin espacios ni comillas)."""
    spec, _, marker = requirement.partition(";")
    name = re.split(r"[\s<>=!~(\[]", spec.strip(), maxsplit=1)[0].lower().replace("_", "-")
    return name, re.sub(r"[\s'\"]", "", marker)


def _fortran_problems(names: list[str]) -> list[str]:
    problems = []
    for member in names:
        path = Path(member)
        parts = {part.lower() for part in path.parts}
        named = "fortran" in parts or "fortran" in path.name.lower()
        if named or path.suffix.lower() in FORTRAN_SUFFIXES:
            problems.append(f"contiene un archivo de Fortran: {member}")
    return problems


def _metadata_problems(text: str) -> list[str]:
    problems: list[str] = []
    meta = Parser().parsestr(text)
    if meta.get("Name", "").lower() != NAME:
        problems.append(f"Name inesperado: {meta.get('Name')!r}")
    if meta.get("Version") != VERSION:
        problems.append(f"Version inesperada: {meta.get('Version')!r} (se espera {VERSION})")
    if (meta.get("Requires-Python") or "").replace(" ", "") != PYTHON_REQUIRES:
        problems.append(f"Requires-Python inesperado: {meta.get('Requires-Python')!r}")
    classifiers = meta.get_all("Classifier") or []
    is_mit = meta.get("License-Expression") == "MIT" or any("MIT License" in c for c in classifiers)
    if not is_mit:
        problems.append("la licencia no es MIT (falta licencia MIT en los metadatos)")

    seen_numpy = False
    for requirement in meta.get_all("Requires-Dist") or []:
        name, marker = _requirement_parts(requirement)
        if name == "numpy" and not marker:
            seen_numpy = True
        elif name == "matplotlib" and marker == "extra==plot":
            continue
        else:
            problems.append(f"dependencia no permitida: {requirement!r}")
    if not seen_numpy:
        problems.append("falta la dependencia de ejecución numpy")
    return problems


def analyze_wheel(path: Path) -> list[str]:
    """Devuelve la lista de problemas de una wheel (vacía si es correcta)."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        metadata_files = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            return [f"se esperaba un METADATA, hay {len(metadata_files)}"]
        problems = _metadata_problems(zf.read(metadata_files[0]).decode("utf-8"))
    if f"{NAME}/py.typed" not in names:
        problems.append("falta caustix/py.typed")
    problems.extend(_fortran_problems(names))
    return problems


def analyze_sdist(path: Path) -> list[str]:
    """Devuelve la lista de problemas de un sdist (vacía si es correcto)."""
    problems: list[str] = []
    with tarfile.open(path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        names = []
        for member in members:
            _, _, relative = member.name.partition("/")
            names.append(relative)
        pkg_info = next((m for m in members if m.name.endswith("/PKG-INFO")), None)
        if pkg_info is None:
            problems.append("falta PKG-INFO")
        else:
            extracted = tf.extractfile(pkg_info)
            assert extracted is not None
            problems.extend(_metadata_problems(extracted.read().decode("utf-8")))
    for relative in names:
        if FORBIDDEN_NAMES.search(relative):
            problems.append(f"archivo no permitido (secreto o temporal): {relative}")
        elif relative not in SDIST_ALLOWED_FILES and not relative.startswith(SDIST_ALLOWED_DIRS):
            problems.append(f"archivo ajeno al paquete: {relative}")
    if "LICENSE" not in names:
        problems.append("falta LICENSE")
    if f"src/{NAME}/py.typed" not in names:
        problems.append("falta src/caustix/py.typed")
    problems.extend(_fortran_problems(names))
    return problems


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


def _check_clean_install(wheel: Path, workdir: Path) -> list[str]:
    """Instala la wheel en un venv limpio y comprueba importación y uso básico."""
    env_dir = workdir / "clean-venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    python = env_dir / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    installed = _run([str(python), "-m", "pip", "install", "--quiet", str(wheel)])
    if installed.returncode != 0:
        return [f"pip install falló:\n{installed.stdout}{installed.stderr}"]
    script = (
        "import caustix\n"
        f"assert caustix.__version__ == {VERSION!r}, caustix.__version__\n"
        "tr = caustix.projectile([0.0, 0.0], [3.0, 4.0], dt=0.1)\n"
        "assert tr.hit_ground\n"
        "d = caustix.projectile_with_drag([0.0, 0.0], [3.0, 4.0], mass=1.0, rho=1.2, cd=0.5,\n"
        "                                 area=0.01, dt=0.01)\n"
        "assert d.hit_ground\n"
    )
    # -I: aislado del directorio actual para no importar el código fuente del repositorio
    result = _run([str(python), "-I", "-c", script], cwd=str(workdir))
    if result.returncode != 0:
        return [f"el uso básico en el venv limpio falló:\n{result.stdout}{result.stderr}"]
    return []


def main() -> int:
    """Punto de entrada; devuelve el código de salida."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="caustix-dist-") as tmp:
        workdir = Path(tmp)
        out = workdir / "dist"
        built = _run([sys.executable, "-m", "build", "--outdir", str(out)], cwd=str(PROJECT_DIR))
        if built.returncode != 0:
            sys.stderr.write(f"python -m build falló:\n{built.stdout}{built.stderr}\n")
            return 1
        wheels = sorted(out.glob("*.whl"))
        sdists = sorted(out.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            sys.stderr.write(f"se esperaba 1 wheel y 1 sdist; hay {len(wheels)} y {len(sdists)}\n")
            return 1

        twine = _run(
            [sys.executable, "-m", "twine", "check", "--strict", str(wheels[0]), str(sdists[0])]
        )
        if twine.returncode != 0:
            problems.append(f"twine check falló:\n{twine.stdout}{twine.stderr}")
        problems.extend(f"wheel: {p}" for p in analyze_wheel(wheels[0]))
        problems.extend(f"sdist: {p}" for p in analyze_sdist(sdists[0]))
        problems.extend(f"venv limpio: {p}" for p in _check_clean_install(wheels[0], workdir))

    if problems:
        sys.stderr.write("check_dist: FALLO\n" + "\n".join(f"- {p}" for p in problems) + "\n")
        return 1
    sys.stdout.write("check_dist: OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
