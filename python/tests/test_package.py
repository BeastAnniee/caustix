"""Pruebas del esqueleto del paquete."""

import subprocess
import sys


def test_version_is_1_0_0() -> None:
    import caustix

    assert caustix.__version__ == "1.0.0"


def test_import_does_not_require_matplotlib() -> None:
    code = (
        "import sys; sys.modules['matplotlib'] = None; import caustix; assert caustix.__version__"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
