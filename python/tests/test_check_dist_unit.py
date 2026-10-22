"""Análisis de metadatos de empaquetado con distribuciones sintéticas."""

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "check_dist.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_dist", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_dist = _load()

GOOD_METADATA = """Metadata-Version: 2.4
Name: caustix
Version: 1.0.0
Summary: Fisica en Python
Requires-Python: >=3.10
License-Expression: MIT
License-File: LICENSE
Provides-Extra: plot
Requires-Dist: numpy>=1.24
Requires-Dist: matplotlib; extra == 'plot'

# caustix
Menciona Fortran solo como herramienta de desarrollo.
"""

GOOD_FILES = {
    "caustix/__init__.py": "__version__ = '1.0.0'\n",
    "caustix/py.typed": "",
    "caustix-1.0.0.dist-info/licenses/LICENSE": "MIT License\n",
}


def make_wheel(
    tmp_path: Path, metadata: str = GOOD_METADATA, files: dict[str, str] | None = None
) -> Path:
    path = tmp_path / "caustix-1.0.0-py3-none-any.whl"
    content = dict(GOOD_FILES if files is None else files)
    content["caustix-1.0.0.dist-info/METADATA"] = metadata
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in content.items():
            zf.writestr(name, text)
    return path


def make_sdist(tmp_path: Path, files: dict[str, str]) -> Path:
    path = tmp_path / "caustix-1.0.0.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"caustix-1.0.0/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


GOOD_SDIST = {
    "PKG-INFO": GOOD_METADATA,
    "pyproject.toml": "[project]\n",
    "README.md": "# caustix\n",
    "LICENSE": "MIT License\n",
    "CHANGELOG.md": "# Changelog\n",
    "src/caustix/__init__.py": "",
    "src/caustix/py.typed": "",
    "tests/test_x.py": "",
    "tests/fixtures/reference.csv": "1,2\n",
    "tools/check_dist.py": "",
}


def test_good_wheel_has_no_problems(tmp_path: Path) -> None:
    assert check_dist.analyze_wheel(make_wheel(tmp_path)) == []


@pytest.mark.parametrize(
    ("extra_line", "needle"),
    [
        ("Requires-Dist: scipy>=1.10", "scipy"),
        ("Requires-Dist: pytest; extra == 'dev'", "pytest"),
        ("Requires-Dist: matplotlib", "matplotlib"),
        ("Requires-Dist: gfortran-wrapper", "gfortran"),
    ],
)
def test_unexpected_dependency_is_reported(tmp_path: Path, extra_line: str, needle: str) -> None:
    metadata = GOOD_METADATA.replace(
        "Requires-Dist: numpy>=1.24", f"Requires-Dist: numpy>=1.24\n{extra_line}"
    )
    problems = check_dist.analyze_wheel(make_wheel(tmp_path, metadata))
    assert any(needle in p for p in problems), problems


def test_missing_numpy_is_reported(tmp_path: Path) -> None:
    metadata = GOOD_METADATA.replace("Requires-Dist: numpy>=1.24\n", "")
    assert any("numpy" in p for p in check_dist.analyze_wheel(make_wheel(tmp_path, metadata)))


def test_matplotlib_with_extra_marker_double_quotes_is_accepted(tmp_path: Path) -> None:
    metadata = GOOD_METADATA.replace("extra == 'plot'", 'extra == "plot"')
    assert check_dist.analyze_wheel(make_wheel(tmp_path, metadata)) == []


@pytest.mark.parametrize(
    "bad_member",
    ["fortran/src/caustix_rk4.f90", "caustix/fortran/x.py", "caustix/solver.f90"],
)
def test_fortran_files_are_reported(tmp_path: Path, bad_member: str) -> None:
    files = {**GOOD_FILES, bad_member: "x"}
    problems = check_dist.analyze_wheel(make_wheel(tmp_path, files=files))
    assert any("fortran" in p.lower() or ".f90" in p for p in problems), problems


def test_missing_py_typed_is_reported(tmp_path: Path) -> None:
    files = {k: v for k, v in GOOD_FILES.items() if not k.endswith("py.typed")}
    problems = check_dist.analyze_wheel(make_wheel(tmp_path, files=files))
    assert any("py.typed" in p for p in problems)


@pytest.mark.parametrize(
    ("old", "new", "needle"),
    [
        ("Requires-Python: >=3.10", "Requires-Python: >=3.8", "Requires-Python"),
        ("License-Expression: MIT", "License-Expression: GPL-3.0-only", "licencia"),
        ("Name: caustix", "Name: otro", "Name"),
        ("Version: 1.0.0", "Version: 1.0.1", "Version"),
    ],
)
def test_metadata_fields_are_checked(tmp_path: Path, old: str, new: str, needle: str) -> None:
    problems = check_dist.analyze_wheel(make_wheel(tmp_path, GOOD_METADATA.replace(old, new)))
    assert any(needle in p for p in problems), problems


def test_mit_via_classifier_is_accepted(tmp_path: Path) -> None:
    metadata = GOOD_METADATA.replace(
        "License-Expression: MIT", "Classifier: License :: OSI Approved :: MIT License"
    )
    assert check_dist.analyze_wheel(make_wheel(tmp_path, metadata)) == []


def test_good_sdist_has_no_problems(tmp_path: Path) -> None:
    assert check_dist.analyze_sdist(make_sdist(tmp_path, GOOD_SDIST)) == []


@pytest.mark.parametrize(
    "bad_member",
    [
        "fortran/cases.txt",
        "fortran/src/caustix_rk4.f90",
        "src/caustix/x.f90",
        "tests/fixtures/fortran_rk4.csv",
        "tests/test_fortran_crosscheck.py",
    ],
)
def test_sdist_with_fortran_is_reported(tmp_path: Path, bad_member: str) -> None:
    problems = check_dist.analyze_sdist(make_sdist(tmp_path, {**GOOD_SDIST, bad_member: "x"}))
    assert any("fortran" in p.lower() or ".f90" in p for p in problems), problems


@pytest.mark.parametrize("stray", [".env", ".venv/pyvenv.cfg", "id_rsa.pem", "dist/old.whl"])
def test_sdist_with_foreign_files_is_reported(tmp_path: Path, stray: str) -> None:
    problems = check_dist.analyze_sdist(make_sdist(tmp_path, {**GOOD_SDIST, stray: "x"}))
    assert problems, stray


def test_sdist_missing_license_or_typed_marker(tmp_path: Path) -> None:
    files = {k: v for k, v in GOOD_SDIST.items() if k not in {"LICENSE", "src/caustix/py.typed"}}
    problems = check_dist.analyze_sdist(make_sdist(tmp_path, files))
    assert any("LICENSE" in p for p in problems)
    assert any("py.typed" in p for p in problems)


def test_expected_version_comes_from_the_package() -> None:
    import caustix

    assert caustix.__version__ == check_dist.VERSION
