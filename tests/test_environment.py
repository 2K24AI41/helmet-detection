"""Environment sanity checks."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_python_version_is_supported() -> None:
    assert sys.version_info[:2] >= (3, 10)


@pytest.mark.parametrize("module", ["PIL", "numpy", "yaml", "defusedxml", "pytest"])
def test_declared_dependencies_import(module: str) -> None:
    imported = importlib.import_module(module)
    version = getattr(imported, "__version__", "n/a")
    print(f"{module} {version}")


def test_package_imports() -> None:
    import helmet_data

    assert helmet_data.__version__
