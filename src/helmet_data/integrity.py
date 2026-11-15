"""Raw-data integrity manifests.

A manifest records every file under a raw dataset directory (relative POSIX path,
size, SHA-256) and every *empty directory*. Verifying it before and after a run
proves the raw tree was not changed.

Schema: manifest version 1 consists of exactly the keys ``version``, ``files`` and
``empty_dirs``. Version 1 INCLUDES ``empty_dirs``; no earlier manifest format was ever
persisted, so the version was not bumped. Any later schema change must bump
``MANIFEST_VERSION``.

Behaviour that is deliberate and documented:
* Empty directories ARE part of the manifest (``empty_dirs``): creating or removing a
  leaf directory is detected. Non-empty directories are implied by their files.
* File metadata (timestamps, permissions, attributes) is NOT part of the manifest;
  only content, names and structure are.
* Any directory-listing or file-read failure raises ``IntegrityError``; the manifest
  is never silently incomplete.
* Symlinks and Windows reparse points (including junctions) anywhere in the tree
  raise ``UnsafeRawTree``. Files flagged as reparse points (for example OneDrive
  placeholders) are refused too: the module fails closed.
* Manifests are canonical: entries and empty directories are strictly sorted and
  unique, and ``read_manifest`` rejects anything whose bytes differ from the
  canonical serialisation.

Nothing here writes into the raw directory. Manifests are written with
``write_manifest`` into ``data/interim`` only, atomically, and never overwrite an
existing file unless ``overwrite=True`` is passed explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from .models import validate_relative_posix_path, validate_sha256
from .paths import DataLayout, is_link_like, write_file_atomic

PathLike = Union[str, "os.PathLike[str]"]

MANIFEST_VERSION = 1
_CHUNK_SIZE = 1024 * 1024


class IntegrityError(RuntimeError):
    """Raised when a raw tree cannot be manifested or a manifest is malformed."""


class UnsafeRawTree(IntegrityError):
    """Raised when a raw tree contains symlinks or other reparse points."""


def _check_path(path: Any) -> str:
    try:
        return validate_relative_posix_path(path)
    except ValueError as exc:
        raise IntegrityError(str(exc)) from exc


@dataclass(frozen=True)
class ManifestEntry:
    path: str  # relative, POSIX-style
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _check_path(self.path)
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise IntegrityError(f"size must be a non-negative integer, got {self.size!r}")
        try:
            validate_sha256(self.sha256)
        except ValueError as exc:
            raise IntegrityError(str(exc)) from exc


def _require_strictly_sorted(label: str, values: tuple[str, ...]) -> None:
    for earlier, later in zip(values, values[1:]):
        if not earlier < later:
            raise IntegrityError(f"{label} must be strictly sorted and unique near {later!r}")


@dataclass(frozen=True)
class Manifest:
    entries: tuple[ManifestEntry, ...]
    empty_dirs: tuple[str, ...] = ()
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or not all(
            isinstance(e, ManifestEntry) for e in self.entries
        ):
            raise IntegrityError("entries must be a tuple of ManifestEntry")
        if not isinstance(self.empty_dirs, tuple):
            raise IntegrityError("empty_dirs must be a tuple")
        for directory in self.empty_dirs:
            _check_path(directory)
        if isinstance(self.version, bool) or self.version != MANIFEST_VERSION:
            raise IntegrityError(f"unsupported manifest version: {self.version!r}")
        paths = tuple(e.path for e in self.entries)
        _require_strictly_sorted("file entries", paths)
        _require_strictly_sorted("empty_dirs", self.empty_dirs)
        file_set = set(paths)
        ancestors = {
            "/".join(p.split("/")[:i]) for p in paths for i in range(1, len(p.split("/")))
        }
        for directory in self.empty_dirs:
            if directory in file_set:
                raise IntegrityError(f"path is both a file and an empty directory: {directory!r}")
            if directory in ancestors:
                raise IntegrityError(f"empty directory contains files: {directory!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "files": [{"path": e.path, "size": e.size, "sha256": e.sha256} for e in self.entries],
            "empty_dirs": list(self.empty_dirs),
        }


@dataclass(frozen=True)
class ManifestDiff:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    modified: tuple[str, ...]
    added_empty_dirs: tuple[str, ...] = ()
    removed_empty_dirs: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (
            self.added
            or self.removed
            or self.modified
            or self.added_empty_dirs
            or self.removed_empty_dirs
        )


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def _hash_file(path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise IntegrityError(f"cannot read file {path}: {exc}") from exc
    return size, digest.hexdigest()


def _raise_walk_error(error: OSError) -> None:
    """``os.walk`` swallows errors unless given this callback."""
    raise IntegrityError(f"cannot read directory {error.filename}: {error}") from error


def build_manifest(raw_dir: PathLike) -> Manifest:
    """Hash every file below ``raw_dir`` (hidden files included) and list empty directories.

    Raises ``IntegrityError`` on any traversal or read failure and ``UnsafeRawTree``
    on symlinks/reparse points.
    """
    root = os.fspath(raw_dir)
    if not os.path.isdir(root):
        raise IntegrityError(f"raw directory not found: {root}")
    if is_link_like(root):
        raise UnsafeRawTree(f"raw directory is a symlink or reparse point: {root}")

    entries: list[ManifestEntry] = []
    empty_dirs: list[str] = []
    for current, dirnames, filenames in os.walk(root, onerror=_raise_walk_error, followlinks=False):
        dirnames.sort()
        for name in dirnames:
            full = os.path.join(current, name)
            if is_link_like(full):
                raise UnsafeRawTree(f"symlink or reparse point in raw tree: {full}")
        if not dirnames and not filenames and current != root:
            empty_dirs.append(Path(os.path.relpath(current, root)).as_posix())
        for name in sorted(filenames):
            full = os.path.join(current, name)
            if is_link_like(full):
                raise UnsafeRawTree(f"symlink or reparse point in raw tree: {full}")
            if not os.path.isfile(full):
                raise IntegrityError(f"not a regular file: {full}")
            size, sha256 = _hash_file(full)
            relative = Path(os.path.relpath(full, root)).as_posix()
            entries.append(ManifestEntry(path=relative, size=size, sha256=sha256))

    entries.sort(key=lambda entry: entry.path)
    empty_dirs.sort()
    return Manifest(entries=tuple(entries), empty_dirs=tuple(empty_dirs))


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def manifest_bytes(manifest: Manifest) -> bytes:
    """Canonical JSON bytes: identical manifests give identical bytes."""
    text = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return (text + "\n").encode("utf-8")


def manifest_sha256(manifest: Manifest) -> str:
    """A single hash identifying the whole raw tree."""
    return hashlib.sha256(manifest_bytes(manifest)).hexdigest()


def write_manifest(
    manifest: Manifest, dest: PathLike, layout: DataLayout, *, overwrite: bool = False
) -> Path:
    """Write a manifest atomically into ``layout.interim``.

    The destination must be a file inside ``data/interim`` whose parent directory
    already exists (create it with ``paths.ensure_dir``). An existing file is never
    overwritten unless ``overwrite=True``. Destinations in raw, outside interim, or
    reached through ``..``/symlinks/junctions are refused.
    """
    return write_file_atomic(
        layout, dest, manifest_bytes(manifest), area="interim", overwrite=overwrite
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"duplicate key in manifest: {key!r}")
        result[key] = value
    return result


def read_manifest(path: PathLike) -> Manifest:
    """Read and strictly validate a manifest; non-canonical files are rejected."""
    try:
        raw = Path(path).read_bytes()
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, ValueError, RecursionError) as exc:  # JSON/UTF-8 errors, absurd nesting
        raise IntegrityError(f"cannot read manifest {path}: {exc}") from exc

    if not isinstance(data, dict) or set(data) != {"version", "files", "empty_dirs"}:
        raise IntegrityError(f"unsupported manifest structure: {path}")
    if isinstance(data["version"], bool) or data["version"] != MANIFEST_VERSION:
        raise IntegrityError(f"unsupported manifest version in {path}")
    files, empty_dirs = data["files"], data["empty_dirs"]
    if not isinstance(files, list) or not isinstance(empty_dirs, list):
        raise IntegrityError(f"manifest lists are malformed: {path}")

    entries = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise IntegrityError(f"malformed manifest entry in {path}: {item!r}")
        entries.append(ManifestEntry(path=item["path"], size=item["size"], sha256=item["sha256"]))
    for directory in empty_dirs:
        if not isinstance(directory, str):
            raise IntegrityError(f"malformed empty_dirs entry in {path}: {directory!r}")

    manifest = Manifest(entries=tuple(entries), empty_dirs=tuple(empty_dirs))
    if manifest_bytes(manifest) != raw:
        raise IntegrityError(f"manifest is not in canonical form: {path}")
    return manifest


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def diff_manifests(expected: Manifest, actual: Manifest) -> ManifestDiff:
    expected_map = {e.path: e for e in expected.entries}
    actual_map = {e.path: e for e in actual.entries}
    return ManifestDiff(
        added=tuple(sorted(set(actual_map) - set(expected_map))),
        removed=tuple(sorted(set(expected_map) - set(actual_map))),
        modified=tuple(
            sorted(
                path
                for path in set(expected_map) & set(actual_map)
                if expected_map[path] != actual_map[path]
            )
        ),
        added_empty_dirs=tuple(sorted(set(actual.empty_dirs) - set(expected.empty_dirs))),
        removed_empty_dirs=tuple(sorted(set(expected.empty_dirs) - set(actual.empty_dirs))),
    )


def verify_manifest(raw_dir: PathLike, manifest: Manifest) -> ManifestDiff:
    """Compare the current raw tree with ``manifest`` (content changes are detected by hash)."""
    return diff_manifests(manifest, build_manifest(raw_dir))
