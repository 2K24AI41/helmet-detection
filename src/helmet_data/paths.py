"""Data directory layout and path safety guards.

Rules enforced here:
* Raw data is read-only: nothing may be written at or below ``data/raw``.
* The raw root itself must not be a symlink or junction (reparse point). Links inside
  the raw tree are refused too, consistently by ``open_raw`` and the integrity module.
* Raw files are opened in binary read mode only, and only if they really are regular
  files inside ``data/raw`` (``..``, symlinks and junctions cannot escape).
* Path segments follow portable Windows-safe rules (no reserved device names, no
  trailing dots/spaces, no ``<>:"|?*``), applied on every platform.
* Directory creation is explicit and bounded: nothing creates a project root, and
  nothing creates arbitrary parent trees.
* Files are written atomically and, by default, never overwrite an existing file.

Creating a ``DataLayout`` never touches the file system.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

PathLike = Union[str, "os.PathLike[str]"]

#: Build ids: lowercase letters, digits, ``_`` and ``-``; start with a letter/digit.
#: Always use ``BUILD_ID_PATTERN.fullmatch`` (strict whole-string match; a trailing
#: newline or any other trailing character does not match).
BUILD_ID_PATTERN = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,63}\Z")

#: Windows reserved device names (compared case-insensitively, ignoring extension).
RESERVED_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)

#: The only directories (besides ``raw``) that the pipeline creates under the data root.
WRITABLE_AREAS = ("interim", "review", "splits", "processed", "reports")

_FORBIDDEN_SEGMENT_CHARS = frozenset('<>:"|?*')


class PathSafetyError(ValueError):
    """Raised when a path would violate a safety rule."""


class RawWriteError(PathSafetyError):
    """Raised on an attempt to write at or below the raw data directory."""


class RawRootError(PathSafetyError):
    """Raised when the raw root is a symlink or junction."""


class BuildDirExistsError(PathSafetyError):
    """Raised when a build directory already exists (builds are never overwritten)."""


class DestinationExistsError(PathSafetyError):
    """Raised when a file would be overwritten without ``overwrite=True``."""


@dataclass(frozen=True)
class DataLayout:
    """The directory structure under the data root. Construction creates nothing."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def review(self) -> Path:
        return self.root / "review"

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def is_link_like(path: PathLike) -> bool:
    """True for symlinks and, on Windows, any reparse point (junctions, etc.)."""
    target = os.fspath(path)
    if os.path.islink(target):
        return True
    try:
        attributes = os.lstat(target).st_file_attributes  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _reject_stream_syntax(path_text: str, original: str) -> str:
    """Refuse NTFS alternate-data-stream syntax (``name:stream``, ``dir::$INDEX_ALLOCATION``).

    A legitimate Windows path has a ``:`` only in its drive spec (``X:``). Anything else
    can name the same file or directory in a spelling the comparison would not recognise.
    """
    body = path_text[2:] if re.match(r"^[A-Za-z]:", path_text) else path_text
    if ":" in body:
        raise PathSafetyError(f"alternate data stream syntax is not supported: {original!r}")
    return path_text


def _clean_prefix(text: str) -> str:
    """Normalise Windows extended-length spellings, or refuse them.

    On Windows:
    * ``\\\\?\\X:\\...`` (and ``\\??\\X:\\...``, ``//?/X:/...``) is an ordinary drive path
      in another spelling, so the prefix is stripped and the rest is compared normally;
    * ``\\\\.\\`` device paths, ``\\\\?\\UNC\\...``, volume-GUID/``GLOBALROOT`` forms and
      every UNC path (``\\\\server\\share``) are refused with ``PathSafetyError``. They
      can name the same files as a local path (for example ``\\\\localhost\\D$\\...``)
      and cannot be compared reliably, so no spelling may authorise anything.
    On other platforms the text is returned unchanged.
    """
    if os.name != "nt":
        return text
    unified = text.replace("/", "\\")
    lowered = unified.lower()
    if lowered.startswith(("\\\\?\\unc\\", "\\??\\unc\\")):
        raise PathSafetyError(f"UNC paths are not supported: {text!r}")
    if lowered.startswith("\\\\.\\"):
        raise PathSafetyError(f"Windows device paths are not supported: {text!r}")
    for prefix in ("\\\\?\\", "\\??\\"):
        if unified.startswith(prefix):
            rest = unified[len(prefix):]
            if re.match(r"^[A-Za-z]:(\\|$)", rest):
                return _reject_stream_syntax(rest, text)
            raise PathSafetyError(f"unsupported extended-length path: {text!r}")
    if unified.startswith("\\\\"):
        raise PathSafetyError(f"UNC paths are not supported: {text!r}")
    return _reject_stream_syntax(text, text)


def _canonical(path: PathLike) -> str:
    """Absolute, symlink-resolved, case-normalised form used for comparisons.

    Extended-length spellings are normalised both before and after link resolution;
    device, UNC and other unsupported forms raise ``PathSafetyError``.
    """
    resolved = os.path.realpath(_clean_prefix(os.fspath(path)))
    return os.path.normcase(_clean_prefix(resolved))


def _is_within(path: PathLike, base: PathLike, *, allow_equal: bool) -> bool:
    child = _canonical(path)
    parent = _canonical(base)
    try:
        common = os.path.commonpath([child, parent])
    except ValueError:  # different drives
        return False
    if common != parent:
        return False
    return allow_equal or child != parent


def _lexically_within(path: PathLike, base: PathLike) -> bool:
    """Strictly inside ``base`` after purely textual normalisation (``..`` collapsed)."""
    child = os.path.normcase(os.path.abspath(_clean_prefix(os.fspath(path))))
    parent = os.path.normcase(os.path.abspath(_clean_prefix(os.fspath(base))))
    try:
        return os.path.commonpath([child, parent]) == parent and child != parent
    except ValueError:
        return False


def check_segment(piece: str, original: str = "") -> None:
    """Validate a single path component with portable Windows-safe rules."""
    shown = original or piece
    if piece in ("", ".", ".."):
        raise PathSafetyError(f"empty or relative path component in: {shown!r}")
    for char in piece:
        if ord(char) < 32 or char in _FORBIDDEN_SEGMENT_CHARS:
            raise PathSafetyError(f"forbidden character {char!r} in path segment: {shown!r}")
    if piece[-1] in (".", " "):
        raise PathSafetyError(f"path segment must not end with a dot or space: {shown!r}")
    if piece.split(".")[0].rstrip(" ").lower() in RESERVED_DEVICE_NAMES:
        raise PathSafetyError(f"reserved device name in path segment: {shown!r}")


def safe_join(base: PathLike, *parts: str) -> Path:
    """Join ``parts`` onto ``base``, rejecting anything that could escape ``base``.

    Rejected: non-strings, empty parts, absolute/UNC/drive forms, ``.``/``..``/empty
    components, ``<>:"|?*`` and control characters, trailing dots or spaces, Windows
    reserved device names, and any result resolving (through symlinks/junctions)
    outside ``base``. Both ``/`` and ``\\`` are treated as separators.
    """
    for part in parts:
        if not isinstance(part, str) or part == "":
            raise PathSafetyError(f"invalid path segment: {part!r}")
        if part[0] in ("/", "\\") or os.path.isabs(part):
            raise PathSafetyError(f"absolute path segment not allowed: {part!r}")
        for piece in re.split(r"[\\/]", part):
            check_segment(piece, part)
    joined = Path(base).joinpath(*parts)
    if not _is_within(joined, base, allow_equal=True):
        raise PathSafetyError(f"path escapes its base directory: {joined}")
    return joined


# --------------------------------------------------------------------------- #
# Raw-data guards
# --------------------------------------------------------------------------- #
def assert_raw_root_safe(layout: DataLayout) -> None:
    """Raise ``RawRootError`` if ``data/raw`` exists and is a symlink or junction."""
    raw = layout.raw
    if os.path.lexists(raw) and is_link_like(raw):
        raise RawRootError(f"the raw root must not be a symlink or junction: {raw}")


def assert_writable(path: PathLike, layout: DataLayout) -> None:
    """Raise ``RawWriteError`` if ``path`` is the raw directory or lies inside it.

    The check resolves ``..``, symlinks, junctions and (on Windows) letter case.
    """
    assert_raw_root_safe(layout)
    if _is_within(path, layout.raw, allow_equal=True):
        raise RawWriteError(f"writing to the raw data directory is not allowed: {path}")


def open_raw(layout: DataLayout, path: PathLike) -> BinaryIO:
    """Open a regular file inside ``data/raw`` in binary read mode.

    A relative ``path`` is taken relative to ``data/raw``. The file must lie strictly
    inside ``data/raw`` both textually (``..`` collapsed) and after resolving links,
    no component below the raw root may be a symlink/junction, and the opened handle
    must be a regular file. There is no write mode.
    """
    assert_raw_root_safe(layout)
    raw_plain = Path(_clean_prefix(os.fspath(layout.raw)))
    candidate = Path(_clean_prefix(os.fspath(path)))
    if not candidate.is_absolute():
        candidate = raw_plain / candidate
    if not _lexically_within(candidate, raw_plain):
        raise PathSafetyError(f"file is not inside the raw data directory: {path}")
    if not _is_within(candidate, layout.raw, allow_equal=False):
        raise PathSafetyError(f"file resolves outside the raw data directory: {path}")

    raw_abs = os.path.abspath(raw_plain)
    relative = os.path.relpath(os.path.abspath(candidate), raw_abs)
    current = raw_abs
    for piece in relative.split(os.sep):
        current = os.path.join(current, piece)
        if os.path.lexists(current) and is_link_like(current):
            raise PathSafetyError(f"symlink or junction inside the raw tree: {current}")

    resolved = os.path.realpath(candidate)
    if not os.path.isfile(resolved):
        raise PathSafetyError(f"not a regular file: {path}")
    handle = open(resolved, "rb")
    try:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise PathSafetyError(f"not a regular file: {path}")
    except BaseException:
        handle.close()
        raise
    return handle


# --------------------------------------------------------------------------- #
# Bounded directory creation and atomic writing
# --------------------------------------------------------------------------- #
def _require_data_root(layout: DataLayout) -> None:
    if not layout.root.is_dir():
        raise PathSafetyError(
            f"the data root does not exist (it is never created automatically): {layout.root}"
        )


def _ensure_area(layout: DataLayout, area: str) -> Path:
    if area not in WRITABLE_AREAS:
        raise PathSafetyError(f"unknown or non-writable data area: {area!r}")
    _require_data_root(layout)
    area_dir = layout.root / area
    if os.path.lexists(area_dir) and is_link_like(area_dir):
        raise PathSafetyError(f"data area must not be a symlink or junction: {area_dir}")
    assert_writable(area_dir, layout)
    area_dir.mkdir(exist_ok=True)
    return area_dir


def ensure_dir(layout: DataLayout, area: str, *parts: str) -> Path:
    """Create ``<root>/<area>/<parts...>`` and return it.

    The data root must already exist and ``area`` must be one of ``WRITABLE_AREAS``;
    ``parts`` are checked with ``safe_join`` and cannot leave the area.
    """
    area_dir = _ensure_area(layout, area)
    if not parts:
        return area_dir
    target = safe_join(area_dir, *parts)
    assert_writable(target, layout)
    target.mkdir(parents=True, exist_ok=True)
    return target


def new_build_dir(layout: DataLayout, build_id: str) -> Path:
    """Create and return a new ``processed/<build_id>`` directory.

    ``build_id`` must match ``[a-z0-9][a-z0-9_-]{0,63}`` and not be a Windows reserved
    device name. Existing builds are never overwritten. The data root must exist; only
    ``processed/`` (one level) and the build directory are created.
    """
    if (
        not isinstance(build_id, str)
        or not BUILD_ID_PATTERN.fullmatch(build_id)
        or build_id in RESERVED_DEVICE_NAMES
    ):
        raise PathSafetyError(f"invalid build id: {build_id!r}")
    processed = _ensure_area(layout, "processed")
    target = processed / build_id
    if not _is_within(target, processed, allow_equal=False):
        raise PathSafetyError(f"build directory escapes processed/: {target}")
    assert_writable(target, layout)
    try:
        target.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise BuildDirExistsError(f"build directory already exists: {target}") from exc
    return target


def write_file_atomic(
    layout: DataLayout,
    dest: PathLike,
    data: bytes,
    *,
    area: str = "interim",
    overwrite: bool = False,
) -> Path:
    """Write ``data`` to ``dest`` atomically; refuse to overwrite unless asked.

    ``dest`` must be a file strictly inside ``<root>/<area>`` (checked after resolving
    ``..``, symlinks and junctions), with a safe file name, in a directory that already
    exists. Nothing is created except the file. The data is written to a temporary file
    in the same directory, flushed, and published by an operation that fails if the
    destination exists (rename on Windows, hard link on POSIX); the temporary file is
    always cleaned up, including on failure.
    """
    if area not in WRITABLE_AREAS:
        raise PathSafetyError(f"unknown or non-writable data area: {area!r}")
    assert_raw_root_safe(layout)
    destination = Path(_clean_prefix(os.fspath(dest)))
    area_dir = layout.root / area
    if not _is_within(destination, area_dir, allow_equal=False):
        raise PathSafetyError(f"destination must be inside {area_dir}: {destination}")
    assert_writable(destination, layout)
    check_segment(destination.name)
    parent = destination.parent
    if not parent.is_dir():
        raise PathSafetyError(f"parent directory does not exist (not created): {parent}")
    if not overwrite and os.path.lexists(destination):
        raise DestinationExistsError(f"destination already exists: {destination}")

    descriptor, temp_name = tempfile.mkstemp(dir=parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            if overwrite:
                os.replace(temp_name, destination)
            elif os.name == "nt":
                os.rename(temp_name, destination)  # fails if the destination exists
            else:
                os.link(temp_name, destination)  # fails if the destination exists
                os.unlink(temp_name)
        except FileExistsError as exc:
            raise DestinationExistsError(f"destination already exists: {destination}") from exc
    finally:
        if os.path.lexists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    return destination
