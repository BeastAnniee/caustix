"""Documentación de cada símbolo público y ejemplos ejecutables (AC-17, REQ-18)."""

import doctest
import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

import caustix

MODULES = [
    "caustix.kinematics",
    "caustix.projectile",
    "caustix.drag",
    "caustix.results",
    "caustix.errors",
    "caustix.plotting",
]
PHYSICS = {"mru", "mrua", "launch_velocity", "projectile", "projectile_with_drag"}
UNITS = re.compile(r"\b(m|s|kg|m/s|m/s²|m/s2|J|kg/m³)\b")


def _documented_objects() -> list[tuple[str, Any]]:
    objects: list[tuple[str, Any]] = []
    for name in caustix.__all__:
        obj = getattr(caustix, name)
        objects.append((name, obj))
        if inspect.isclass(obj):
            for method in ("kinetic_energy", "potential_energy", "mechanical_energy"):
                if hasattr(obj, method):
                    objects.append((f"{name}.{method}", getattr(obj, method)))
    return objects


OBJECTS = _documented_objects()


@pytest.mark.parametrize(("name", "obj"), OBJECTS, ids=[n for n, _ in OBJECTS])
def test_symbol_has_complete_docstring(name: str, obj: Any) -> None:
    doc = inspect.getdoc(obj)
    assert doc, f"{name} no tiene docstring"
    is_callable_api = inspect.isfunction(obj)
    if is_callable_api:
        assert "Args:" in doc, f"{name}: falta la sección Args"
        assert "Raises:" in doc, f"{name}: falta la sección Raises"
        assert UNITS.search(doc), f"{name}: la documentación no indica unidades"
    if inspect.isclass(obj) and "Error" not in name:
        assert "Attributes:" in doc, f"{name}: falta la sección Attributes"
    assert "Example:" in doc and ">>>" in doc, f"{name}: falta un ejemplo ejecutable"
    short = name.split(".")[-1]
    if short in PHYSICS or short.endswith("_energy"):
        assert re.search(r"Ecuaci[óo]n", doc), f"{name}: falta la ecuación implementada"


@pytest.mark.parametrize("module_name", MODULES)
def test_module_doctests_pass(module_name: str) -> None:
    module = importlib.import_module(module_name)
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0


def test_readme_examples_run() -> None:
    readme = Path(__file__).resolve().parent.parent / "README.md"
    result = doctest.testfile(str(readme), module_relative=False, optionflags=doctest.ELLIPSIS)
    assert result.attempted >= 5
    assert result.failed == 0
