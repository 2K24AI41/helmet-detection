"""Shared fixtures. Everything is created under pytest's temporary directories."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from helmet_data.paths import DataLayout

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def layout(tmp_path: Path) -> DataLayout:
    """A data layout inside tmp_path. Nothing is created until a test does so."""
    return DataLayout(tmp_path / "data")


@pytest.fixture
def raw_dir(layout: DataLayout) -> Path:
    """A small raw dataset: nested folders, an empty file, a dotfile, a Unicode name."""
    raw = layout.raw / "demo"
    (raw / "images" / "sub").mkdir(parents=True)
    (raw / "images" / "a.bin").write_bytes(b"abc")
    (raw / "images" / "sub" / "b.bin").write_bytes(b"hello world")
    (raw / "empty.txt").write_bytes(b"")
    (raw / ".hidden").write_bytes(b"secret")
    (raw / "café.txt").write_bytes(b"unicode name")
    return raw


@pytest.fixture
def symlink_to(tmp_path: Path):
    """Create a symlink, skipping the test if symlinks are unavailable."""
    probe_target = tmp_path / "_probe_target"
    probe_target.write_bytes(b"x")
    probe_link = tmp_path / "_probe_link"
    try:
        os.symlink(probe_target, probe_link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symlinks are not available on this system (needs privilege or Developer Mode)")

    def make(target: Path, link: Path, *, directory: bool = False) -> Path:
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, link, target_is_directory=directory)
        return link

    return make


def _make_junction(target: Path, link: Path) -> Path:
    link.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip(f"cannot create a junction: {result.stdout} {result.stderr}")
    return link


@pytest.fixture
def junction_to(request):
    """Create a Windows directory junction (no administrator rights needed)."""
    if os.name != "nt":
        pytest.skip("junctions are a Windows feature")
    created: list[Path] = []

    def make(target: Path, link: Path) -> Path:
        _make_junction(target, link)
        created.append(link)
        return link

    def cleanup() -> None:
        for link in reversed(created):
            try:
                os.rmdir(link)  # removes the junction only, never its target
            except OSError:
                pass

    request.addfinalizer(cleanup)
    return make


@pytest.fixture
def dir_link(request, tmp_path: Path):
    """Create a directory link: a symlink where possible, else a junction on Windows."""
    created: list[Path] = []

    def make(target: Path, link: Path) -> Path:
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError):
            if os.name != "nt":
                pytest.skip("directory links are not available on this system")
            _make_junction(target, link)
        created.append(link)
        return link

    def cleanup() -> None:
        for link in reversed(created):
            try:
                os.rmdir(link)
            except OSError:
                try:
                    os.unlink(link)
                except OSError:
                    pass

    request.addfinalizer(cleanup)
    return make
