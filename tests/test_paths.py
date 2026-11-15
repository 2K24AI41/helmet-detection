"""Tests for helmet_data.paths."""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from helmet_data.paths import (
    BuildDirExistsError,
    DataLayout,
    DestinationExistsError,
    PathSafetyError,
    RawRootError,
    RawWriteError,
    assert_raw_root_safe,
    assert_writable,
    ensure_dir,
    is_link_like,
    new_build_dir,
    open_raw,
    safe_join,
    write_file_atomic,
)

IS_WINDOWS = sys.platform == "win32"


def make_root(layout: DataLayout) -> DataLayout:
    layout.root.mkdir(parents=True, exist_ok=True)
    return layout


def listing(path: Path) -> list[str]:
    return sorted(p.relative_to(path).as_posix() for p in path.rglob("*"))


# --------------------------------------------------------------------------- #
# DataLayout
# --------------------------------------------------------------------------- #
def test_layout_paths_and_no_side_effects(tmp_path: Path) -> None:
    root = tmp_path / "data"
    layout = DataLayout(root)
    assert layout.raw == root / "raw"
    assert layout.interim == root / "interim"
    assert layout.review == root / "review"
    assert layout.splits == root / "splits"
    assert layout.processed == root / "processed"
    assert layout.reports == root / "reports"
    assert not root.exists()


def test_layout_accepts_string_root(tmp_path: Path) -> None:
    assert DataLayout(str(tmp_path / "d")).raw == tmp_path / "d" / "raw"


# --------------------------------------------------------------------------- #
# safe_join
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "good", ["a b", "caf\u00e9", "img 1.jpg", "a/b/c.txt", "a\\b", ".hidden", "a.b", "console.txt", "nullable", "com10", "lpt0"]
)
def test_safe_join_allows_ordinary_names(tmp_path: Path, good: str) -> None:
    assert safe_join(tmp_path, good) == Path(tmp_path).joinpath(*good.replace("\\", "/").split("/"))


@pytest.mark.parametrize(
    "bad",
    [
        "", ".", "..", "../x", "a/../../x", "a\\..\\x", "./x", "a/./b",
        "/etc/passwd", "\\windows", "C:\\Windows", "C:relative", "\\\\server\\share\\x", "//server/share",
        "a//b", "a/", "a\\", "a\x00b", "a\tb", "a\nb",
        "a:b", "sub/C:x", "x.txt:stream", 'a"b', "a<b", "a>b", "a|b", "a?b", "a*b",
        ".. ", "...", "x.", "x ", "dir./x", "dir /x",
        "NUL", "nul", "CON", "con.txt", "PRN", "AUX", "aux.log", "COM1", "com9.txt", "LPT1", "lpt9", "sub/NUL", "NUL /x",
    ],
)
def test_safe_join_rejects_unsafe_segments(tmp_path: Path, bad: str) -> None:
    with pytest.raises(PathSafetyError):
        safe_join(tmp_path, bad)


def test_safe_join_rejects_non_strings_and_bad_later_parts(tmp_path: Path) -> None:
    with pytest.raises(PathSafetyError):
        safe_join(tmp_path, Path("x"))  # type: ignore[arg-type]
    with pytest.raises(PathSafetyError):
        safe_join(tmp_path, "ok", "..")
    with pytest.raises(PathSafetyError):
        safe_join(tmp_path, "ok", "/abs")


def test_safe_join_with_no_parts_returns_base(tmp_path: Path) -> None:
    assert safe_join(tmp_path) == tmp_path


def test_safe_join_rejects_directory_link_escape(tmp_path: Path, dir_link) -> None:
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    dir_link(outside, base / "link")
    with pytest.raises(PathSafetyError):
        safe_join(base, "link", "x.txt")
    with pytest.raises(PathSafetyError):
        safe_join(base, "link")


def test_safe_join_rejects_junction_escape(tmp_path: Path, junction_to) -> None:
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    junction_to(outside, base / "j")
    assert is_link_like(base / "j")
    with pytest.raises(PathSafetyError):
        safe_join(base, "j", "secret.txt")


def test_safe_join_rejects_symlink_escape(tmp_path: Path, symlink_to) -> None:
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    symlink_to(outside, base / "link", directory=True)
    with pytest.raises(PathSafetyError):
        safe_join(base, "link", "x.txt")


def test_safe_join_allows_link_that_stays_inside_base(tmp_path: Path, dir_link) -> None:
    base = tmp_path / "base"
    (base / "real").mkdir(parents=True)
    dir_link(base / "real", base / "alias")
    assert safe_join(base, "alias", "x.txt") == base / "alias" / "x.txt"


def test_is_link_like(tmp_path: Path, dir_link) -> None:
    target = tmp_path / "t"
    target.mkdir()
    link = dir_link(target, tmp_path / "l")
    assert is_link_like(link)
    assert not is_link_like(target)
    assert not is_link_like(tmp_path / "missing")


# --------------------------------------------------------------------------- #
# assert_writable and the raw root
# --------------------------------------------------------------------------- #
def test_writes_inside_raw_are_refused(layout: DataLayout) -> None:
    for target in (layout.raw, layout.raw / "demo", layout.raw / "demo" / "x.txt", layout.raw / "n" / "d" / "x.txt"):
        with pytest.raises(RawWriteError):
            assert_writable(target, layout)


def test_dotdot_into_raw_is_refused_and_out_of_raw_is_allowed(layout: DataLayout) -> None:
    with pytest.raises(RawWriteError):
        assert_writable(layout.interim / ".." / "raw" / "x.txt", layout)
    assert_writable(layout.raw / ".." / "interim" / "x.json", layout)


@pytest.mark.parametrize("name", ["interim", "review", "splits", "processed", "reports"])
def test_non_raw_directories_are_writable(layout: DataLayout, name: str) -> None:
    assert_writable(getattr(layout, name) / "file.txt", layout)


def test_similarly_named_sibling_is_not_confused_with_raw(layout: DataLayout) -> None:
    assert_writable(layout.root / "raw_backup" / "x.txt", layout)


@pytest.mark.skipif(not IS_WINDOWS, reason="case-insensitive paths are a Windows behaviour")
def test_mixed_case_raw_is_refused_on_windows(layout: DataLayout) -> None:
    with pytest.raises(RawWriteError):
        assert_writable(layout.root / "RAW" / "x.txt", layout)


def test_directory_link_into_raw_is_refused(layout: DataLayout, dir_link) -> None:
    layout.raw.mkdir(parents=True)
    layout.interim.mkdir(parents=True)
    link = dir_link(layout.raw, layout.interim / "sneaky")
    with pytest.raises(RawWriteError):
        assert_writable(link / "x.txt", layout)


def test_symlink_into_raw_is_refused(layout: DataLayout, symlink_to) -> None:
    layout.raw.mkdir(parents=True)
    layout.interim.mkdir(parents=True)
    link = symlink_to(layout.raw, layout.interim / "sneaky", directory=True)
    with pytest.raises(RawWriteError):
        assert_writable(link / "x.txt", layout)


def test_raw_root_that_is_a_directory_link_is_refused_everywhere(layout: DataLayout, tmp_path: Path, dir_link) -> None:
    make_root(layout)
    real = tmp_path / "elsewhere_raw"
    (real / "demo").mkdir(parents=True)
    (real / "demo" / "f.bin").write_bytes(b"x")
    dir_link(real, layout.raw)
    with pytest.raises(RawRootError):
        assert_raw_root_safe(layout)
    with pytest.raises(RawRootError):
        assert_writable(layout.interim / "x", layout)
    with pytest.raises(RawRootError):
        open_raw(layout, "demo/f.bin")
    with pytest.raises(RawRootError):
        write_file_atomic(layout, layout.interim / "m.json", b"{}")
    with pytest.raises(RawRootError):
        new_build_dir(layout, "b1")


def test_raw_root_junction(layout: DataLayout, tmp_path: Path, junction_to) -> None:
    make_root(layout)
    real = tmp_path / "elsewhere_raw"
    real.mkdir()
    junction_to(real, layout.raw)
    with pytest.raises(RawRootError):
        assert_raw_root_safe(layout)


def test_missing_raw_root_is_not_an_error(layout: DataLayout) -> None:
    assert_raw_root_safe(layout)


# --------------------------------------------------------------------------- #
# open_raw
# --------------------------------------------------------------------------- #
def test_open_raw_reads_bytes_with_relative_and_absolute_paths(layout: DataLayout, raw_dir: Path) -> None:
    with open_raw(layout, Path("demo") / "images" / "a.bin") as handle:
        assert handle.read() == b"abc"
        assert handle.mode == "rb"
    with open_raw(layout, raw_dir / "images" / "a.bin") as handle:
        assert handle.read() == b"abc"
    with open_raw(layout, str(raw_dir / "caf\u00e9.txt")) as handle:
        assert handle.read() == b"unicode name"


def test_open_raw_handle_is_read_only(layout: DataLayout, raw_dir: Path) -> None:
    with open_raw(layout, raw_dir / "images" / "a.bin") as handle:
        assert not handle.writable()
        with pytest.raises(io.UnsupportedOperation):
            handle.write(b"x")  # type: ignore[arg-type]
    assert (raw_dir / "images" / "a.bin").read_bytes() == b"abc"


def test_open_raw_rejects_file_outside_raw(layout: DataLayout, raw_dir: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"nope")
    with pytest.raises(PathSafetyError):
        open_raw(layout, outside)


@pytest.mark.parametrize("attempt", ["../secret.txt", "demo/../../secret.txt", "..\\secret.txt", "demo/../../data/secret.txt"])
def test_open_raw_rejects_dotdot_traversal(layout: DataLayout, raw_dir: Path, attempt: str) -> None:
    (layout.root / "secret.txt").write_bytes(b"secret")
    with pytest.raises(PathSafetyError):
        open_raw(layout, attempt)


def test_open_raw_rejects_raw_directory_and_directories(layout: DataLayout, raw_dir: Path) -> None:
    for target in (layout.raw, raw_dir, raw_dir / "images"):
        with pytest.raises(PathSafetyError):
            open_raw(layout, target)


def test_open_raw_rejects_missing_file(layout: DataLayout, raw_dir: Path) -> None:
    with pytest.raises(PathSafetyError):
        open_raw(layout, raw_dir / "missing.bin")


def test_open_raw_rejects_similarly_named_sibling(layout: DataLayout, raw_dir: Path) -> None:
    sibling = layout.root / "raw_backup"
    sibling.mkdir()
    (sibling / "f.bin").write_bytes(b"x")
    with pytest.raises(PathSafetyError):
        open_raw(layout, sibling / "f.bin")


def test_open_raw_rejects_directory_link_leading_outside(layout: DataLayout, raw_dir: Path, tmp_path: Path, dir_link) -> None:
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "f.txt").write_bytes(b"outside")
    dir_link(outside, raw_dir / "linkdir")
    with pytest.raises(PathSafetyError):
        open_raw(layout, raw_dir / "linkdir" / "f.txt")
    with pytest.raises(PathSafetyError):
        open_raw(layout, "demo/linkdir/f.txt")


def test_open_raw_rejects_junction_leading_outside(layout: DataLayout, raw_dir: Path, tmp_path: Path, junction_to) -> None:
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "f.txt").write_bytes(b"outside")
    junction_to(outside, raw_dir / "jlink")
    with pytest.raises(PathSafetyError):
        open_raw(layout, raw_dir / "jlink" / "f.txt")


def test_open_raw_rejects_links_that_stay_inside_raw(layout: DataLayout, raw_dir: Path, dir_link) -> None:
    dir_link(raw_dir / "images", raw_dir / "alias")
    with pytest.raises(PathSafetyError, match="symlink or junction"):
        open_raw(layout, raw_dir / "alias" / "a.bin")


def test_open_raw_rejects_file_symlink_leading_outside(layout: DataLayout, raw_dir: Path, tmp_path: Path, symlink_to) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"secret")
    link = symlink_to(secret, raw_dir / "link.txt")
    with pytest.raises(PathSafetyError):
        open_raw(layout, link)


# --------------------------------------------------------------------------- #
# ensure_dir
# --------------------------------------------------------------------------- #
def test_ensure_dir_requires_existing_root_and_creates_nothing_otherwise(layout: DataLayout, tmp_path: Path) -> None:
    with pytest.raises(PathSafetyError, match="data root does not exist"):
        ensure_dir(layout, "interim")
    assert not layout.root.exists()
    assert listing(tmp_path) == []


def test_ensure_dir_creates_only_the_requested_area_tree(layout: DataLayout) -> None:
    make_root(layout)
    created = ensure_dir(layout, "interim", "demo", "deeper")
    assert created == layout.interim / "demo" / "deeper"
    assert created.is_dir()
    assert listing(layout.root) == ["interim", "interim/demo", "interim/demo/deeper"]


@pytest.mark.parametrize("area", ["raw", "data", "", "../x", "INTERIM"])
def test_ensure_dir_rejects_unknown_or_raw_areas(layout: DataLayout, area: str) -> None:
    make_root(layout)
    with pytest.raises(PathSafetyError):
        ensure_dir(layout, area)
    assert listing(layout.root) == []


@pytest.mark.parametrize("part", ["..", "../x", "/abs", "C:\\x", "NUL", "x."])
def test_ensure_dir_rejects_unsafe_parts(layout: DataLayout, part: str) -> None:
    make_root(layout)
    with pytest.raises(PathSafetyError):
        ensure_dir(layout, "interim", part)


def test_ensure_dir_rejects_area_that_is_a_link(layout: DataLayout, tmp_path: Path, dir_link) -> None:
    make_root(layout)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    dir_link(elsewhere, layout.interim)
    with pytest.raises(PathSafetyError):
        ensure_dir(layout, "interim", "sub")
    assert listing(elsewhere) == []


# --------------------------------------------------------------------------- #
# new_build_dir
# --------------------------------------------------------------------------- #
def test_new_build_dir_creates_directory(layout: DataLayout) -> None:
    make_root(layout)
    created = new_build_dir(layout, "build-001_a")
    assert created == layout.processed / "build-001_a"
    assert created.is_dir()
    assert listing(layout.root) == ["processed", "processed/build-001_a"]


def test_new_build_dir_never_overwrites(layout: DataLayout) -> None:
    make_root(layout)
    created = new_build_dir(layout, "build-001")
    (created / "keep.txt").write_bytes(b"keep")
    with pytest.raises(BuildDirExistsError):
        new_build_dir(layout, "build-001")
    assert (created / "keep.txt").read_bytes() == b"keep"


@pytest.mark.parametrize(
    "bad",
    [
        "", ".", "..", "../x", "a/b", "a\\b", "C:x", "-start", "_start", ".hidden", "x" * 65, "a b",
        "Build", "UPPER", "abc.", "abc ", "a.b", "CON", "con", "nul", "aux", "prn", "com1", "lpt9", "caf\u00e9",
    ],
)
def test_new_build_dir_rejects_unsafe_ids_and_creates_nothing(layout: DataLayout, bad: str) -> None:
    make_root(layout)
    with pytest.raises(PathSafetyError):
        new_build_dir(layout, bad)
    assert listing(layout.root) == []


def test_new_build_dir_accepts_maximum_length_and_reserved_lookalikes(layout: DataLayout) -> None:
    make_root(layout)
    for good in ("a" * 64, "console", "com10", "nullable", "0"):
        assert new_build_dir(layout, good).is_dir()


def test_new_build_dir_rejects_non_string_id(layout: DataLayout) -> None:
    make_root(layout)
    with pytest.raises(PathSafetyError):
        new_build_dir(layout, 123)  # type: ignore[arg-type]


def test_new_build_dir_does_not_create_a_missing_project_root(layout: DataLayout, tmp_path: Path) -> None:
    with pytest.raises(PathSafetyError, match="data root does not exist"):
        new_build_dir(layout, "b1")
    assert listing(tmp_path) == []


def test_new_build_dir_refuses_processed_linked_to_raw(layout: DataLayout, dir_link) -> None:
    make_root(layout)
    layout.raw.mkdir()
    dir_link(layout.raw, layout.processed)
    with pytest.raises(PathSafetyError):
        new_build_dir(layout, "b1")
    assert listing(layout.raw) == []


def test_new_build_dir_refuses_processed_junction_to_raw(layout: DataLayout, junction_to) -> None:
    make_root(layout)
    layout.raw.mkdir()
    junction_to(layout.raw, layout.processed)
    with pytest.raises(PathSafetyError):
        new_build_dir(layout, "b1")
    assert listing(layout.raw) == []


# --------------------------------------------------------------------------- #
# write_file_atomic
# --------------------------------------------------------------------------- #
@pytest.fixture
def writable(layout: DataLayout) -> DataLayout:
    make_root(layout)
    ensure_dir(layout, "interim")
    return layout


def test_write_file_atomic_writes_and_leaves_no_temp_files(writable: DataLayout) -> None:
    dest = writable.interim / "m.json"
    assert write_file_atomic(writable, dest, b"payload") == dest
    assert dest.read_bytes() == b"payload"
    assert listing(writable.interim) == ["m.json"]


def test_write_file_atomic_refuses_existing_destination_by_default(writable: DataLayout) -> None:
    dest = writable.interim / "m.json"
    write_file_atomic(writable, dest, b"original")
    with pytest.raises(DestinationExistsError):
        write_file_atomic(writable, dest, b"replacement")
    assert dest.read_bytes() == b"original"
    assert listing(writable.interim) == ["m.json"]


def test_write_file_atomic_overwrites_only_when_asked(writable: DataLayout) -> None:
    dest = writable.interim / "m.json"
    write_file_atomic(writable, dest, b"original")
    write_file_atomic(writable, dest, b"replacement", overwrite=True)
    assert dest.read_bytes() == b"replacement"
    assert listing(writable.interim) == ["m.json"]


def test_overwrite_does_not_write_through_a_hard_link(writable: DataLayout, tmp_path: Path) -> None:
    other = tmp_path / "other.txt"
    other.write_bytes(b"shared")
    dest = writable.interim / "m.json"
    try:
        os.link(other, dest)
    except (OSError, NotImplementedError):
        pytest.skip("hard links are not available")
    write_file_atomic(writable, dest, b"new", overwrite=True)
    assert dest.read_bytes() == b"new"
    assert other.read_bytes() == b"shared"


def test_write_file_atomic_refuses_destinations_in_raw(writable: DataLayout, raw_dir: Path) -> None:
    before = listing(writable.raw)
    for dest in (raw_dir / "m.json", writable.raw / "m.json"):
        with pytest.raises(PathSafetyError):
            write_file_atomic(writable, dest, b"x")
    assert listing(writable.raw) == before


def test_write_file_atomic_refuses_destinations_outside_the_area(writable: DataLayout, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    for dest in (outside / "m.json", writable.root / "m.json", writable.reports / "m.json", writable.interim):
        with pytest.raises(PathSafetyError):
            write_file_atomic(writable, dest, b"x")
    assert listing(outside) == []
    assert not (writable.root / "m.json").exists()


def test_write_file_atomic_refuses_traversal(writable: DataLayout) -> None:
    with pytest.raises(PathSafetyError):
        write_file_atomic(writable, writable.interim / ".." / "m.json", b"x")
    with pytest.raises(PathSafetyError):
        write_file_atomic(writable, writable.interim / ".." / "raw" / "m.json", b"x")
    assert not (writable.root / "m.json").exists()


def test_write_file_atomic_does_not_create_parent_trees(writable: DataLayout) -> None:
    with pytest.raises(PathSafetyError, match="parent directory does not exist"):
        write_file_atomic(writable, writable.interim / "not" / "there" / "m.json", b"x")
    assert listing(writable.interim) == []


def test_write_file_atomic_rejects_unsafe_names_and_areas(writable: DataLayout) -> None:
    for name in ("NUL", "CON.json", "x.", "a:b"):
        with pytest.raises(PathSafetyError):
            write_file_atomic(writable, writable.interim / name, b"x")
    with pytest.raises(PathSafetyError):
        write_file_atomic(writable, writable.interim / "m.json", b"x", area="raw")
    assert listing(writable.interim) == []


def test_write_file_atomic_refuses_directory_link_escape(writable: DataLayout, tmp_path: Path, dir_link) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    dir_link(outside, writable.interim / "linked")
    with pytest.raises(PathSafetyError):
        write_file_atomic(writable, writable.interim / "linked" / "m.json", b"x")
    assert listing(outside) == []


def test_write_file_atomic_refuses_junction_into_raw(writable: DataLayout, junction_to) -> None:
    writable.raw.mkdir()
    junction_to(writable.raw, writable.interim / "sneaky")
    with pytest.raises(PathSafetyError):
        write_file_atomic(writable, writable.interim / "sneaky" / "m.json", b"x")
    assert listing(writable.raw) == []


def test_write_file_atomic_refuses_symlink_escape(writable: DataLayout, tmp_path: Path, symlink_to) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_to(outside, writable.interim / "linked", directory=True)
    with pytest.raises(PathSafetyError):
        write_file_atomic(writable, writable.interim / "linked" / "m.json", b"x")
    assert listing(outside) == []


@pytest.mark.parametrize("failing", ["fsync", "publish"])
def test_failed_writes_leave_no_destination_and_no_temp_file(
    writable: DataLayout, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    def boom(*args, **kwargs):
        raise OSError("simulated failure")

    dest = writable.interim / "m.json"
    with monkeypatch.context() as patch:
        if failing == "fsync":
            patch.setattr(os, "fsync", boom)
        else:
            patch.setattr(os, "rename" if os.name == "nt" else "link", boom)
        with pytest.raises(OSError, match="simulated failure"):
            write_file_atomic(writable, dest, b"payload")
    assert not dest.exists()
    assert listing(writable.interim) == []


def test_failed_overwrite_keeps_the_original(writable: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = writable.interim / "m.json"
    write_file_atomic(writable, dest, b"original")

    def boom(*args, **kwargs):
        raise OSError("simulated failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            write_file_atomic(writable, dest, b"replacement", overwrite=True)
    assert dest.read_bytes() == b"original"
    assert listing(writable.interim) == ["m.json"]


# --------------------------------------------------------------------------- #
# H1: Windows extended-length, device and UNC spellings cannot bypass the raw guard
# --------------------------------------------------------------------------- #
windows_only = pytest.mark.skipif(not IS_WINDOWS, reason="Windows path spellings")


def extended_spellings(path: Path) -> dict[str, str]:
    """Several spellings of one absolute drive path (e.g. D:\\data\\raw\\x.txt)."""
    plain = str(path)
    return {
        "extended": "\\\\?\\" + plain,
        "extended lowercase drive": "\\\\?\\" + plain[0].lower() + plain[1:],
        "extended uppercase": "\\\\?\\" + plain.upper(),
        "extended forward slashes": "//?/" + plain.replace("\\", "/"),
        "nt namespace": "\\??\\" + plain,
    }


def unsupported_spellings(path: Path) -> dict[str, str]:
    plain = str(path)
    drive, rest = plain[0], plain[2:]
    return {
        "device": "\\\\.\\" + plain,
        "device forward slashes": "//./" + plain.replace("\\", "/"),
        "extended unc": "\\\\?\\UNC\\localhost\\" + drive + "$" + rest,
        "extended unc lowercase": "\\\\?\\unc\\localhost\\" + drive + "$" + rest,
        "nt namespace unc": "\\??\\UNC\\localhost\\" + drive + "$" + rest,
        "admin share unc": "\\\\localhost\\" + drive + "$" + rest,
        "unc forward slashes": "//localhost/" + drive + "$" + rest.replace("\\", "/"),
        "unc server share": "\\\\server\\share" + rest,
        "volume guid": "\\\\?\\Volume{00000000-0000-0000-0000-000000000000}" + rest,
        "globalroot": "\\\\?\\GLOBALROOT\\Device\\HarddiskVolume1" + rest,
        "extended without colon": "\\\\?\\" + drive + rest,
        "bare extended prefix": "\\\\?\\",
    }


@pytest.fixture
def writable_layout(layout: DataLayout) -> DataLayout:
    make_root(layout)
    ensure_dir(layout, "interim")
    return layout


@windows_only
@pytest.mark.parametrize("tail", ["", "x.txt", "demo/x.txt", "a/b/c/x.txt"])
def test_extended_length_spellings_of_raw_paths_are_blocked(layout: DataLayout, tail: str) -> None:
    layout.raw.mkdir(parents=True)
    target = layout.raw.joinpath(*tail.split("/")) if tail else layout.raw
    for label, spelling in extended_spellings(target).items():
        with pytest.raises(RawWriteError):
            assert_writable(spelling, layout)
        with pytest.raises(RawWriteError):
            assert_writable(Path(spelling), layout)


@windows_only
def test_extended_length_dotdot_into_raw_is_blocked(layout: DataLayout) -> None:
    layout.raw.mkdir(parents=True)
    layout.interim.mkdir()
    sneaky = layout.interim / ".." / "raw" / "x.txt"
    for label, spelling in extended_spellings(sneaky).items():
        with pytest.raises(RawWriteError):
            assert_writable(spelling, layout)


@windows_only
@pytest.mark.parametrize("raw_exists", [True, False])
def test_device_unc_and_unsupported_spellings_are_refused_not_authorised(layout: DataLayout, raw_exists: bool) -> None:
    if raw_exists:
        layout.raw.mkdir(parents=True)
    for target in (layout.raw, layout.raw / "x.txt", layout.interim / "x.json"):
        for label, spelling in unsupported_spellings(target).items():
            with pytest.raises(PathSafetyError):
                assert_writable(spelling, layout)


@windows_only
def test_no_spelling_of_a_raw_path_is_ever_allowed(layout: DataLayout) -> None:
    layout.raw.mkdir(parents=True)
    target = layout.raw / "x.txt"
    spellings = {**extended_spellings(target), **unsupported_spellings(target), "plain": str(target)}
    for label, spelling in spellings.items():
        try:
            assert_writable(spelling, layout)
        except PathSafetyError:
            continue
        pytest.fail(f"assert_writable allowed a raw path spelled as: {label}: {spelling}")


@windows_only
def test_extended_prefix_regression_guard_does_not_touch_raw(layout: DataLayout) -> None:
    """The original bypass: this spelling reached raw at OS level and was allowed."""
    layout.raw.mkdir(parents=True)
    spelling = "\\\\?\\" + str(layout.raw / "probe.txt")
    with pytest.raises(RawWriteError):
        assert_writable(spelling, layout)
    assert not (layout.raw / "probe.txt").exists()  # only the guard was exercised


@windows_only
def test_extended_length_paths_to_allowed_areas_still_work(writable_layout: DataLayout, raw_dir: Path) -> None:
    layout = writable_layout
    for spelling in extended_spellings(layout.interim / "m.json").values():
        assert_writable(spelling, layout)  # legitimate, safe spelling: not refused
    dest = layout.interim / "m.json"
    write_file_atomic(layout, "\\\\?\\" + str(dest), b"payload")
    assert dest.read_bytes() == b"payload"
    with open_raw(layout, "\\\\?\\" + str(raw_dir / "images" / "a.bin")) as handle:
        assert handle.read() == b"abc"
    extended_root = "\\\\?\\" + str(layout.root)
    assert safe_join(extended_root, "interim", "x") == Path(extended_root) / "interim" / "x"  # stays under its base
    with pytest.raises(PathSafetyError):
        safe_join(extended_root, "..", "x")


@windows_only
def test_layout_rooted_at_an_extended_length_path_still_protects_raw(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "raw").mkdir(parents=True)
    extended = DataLayout(Path("\\\\?\\" + str(root)))
    with pytest.raises(RawWriteError):
        assert_writable(root / "raw" / "x.txt", extended)
    with pytest.raises(RawWriteError):
        assert_writable("\\\\?\\" + str(root / "raw" / "x.txt"), extended)
    assert_writable(root / "interim" / "x.json", extended)


@windows_only
def test_extended_length_junction_into_raw_is_still_blocked(layout: DataLayout, junction_to) -> None:
    layout.raw.mkdir(parents=True)
    layout.interim.mkdir()
    junction_to(layout.raw, layout.interim / "sneaky")
    for spelling in extended_spellings(layout.interim / "sneaky" / "x.txt").values():
        with pytest.raises(RawWriteError):
            assert_writable(spelling, layout)


@windows_only
def test_write_file_atomic_refuses_extended_and_unc_destinations_in_raw(writable_layout: DataLayout) -> None:
    layout = writable_layout
    layout.raw.mkdir()
    target = layout.raw / "m.json"
    for spelling in {**extended_spellings(target), **unsupported_spellings(target)}.values():
        with pytest.raises(PathSafetyError):
            write_file_atomic(layout, spelling, b"x")
    assert listing(layout.raw) == []


# --------------------------------------------------------------------------- #
# H2: strict build-id matching; invalid ids raise PathSafetyError, never an OS error
# --------------------------------------------------------------------------- #
INVALID_BUILD_IDS = [
    "b1\n", "b1\r", "b1\r\n", "b1\n\n", "b1 ", "b1\t", "b1\x00", "b1\x0b", "b1\x0c",
    "b1\u2028", "b1\u0085", "b1\u00a0", "b1\u200b", "\nb1", " b1", "b\n1", "b1.", "b1/", "b1\\",
]


@pytest.mark.parametrize("bad", INVALID_BUILD_IDS)
def test_build_ids_with_unexpected_characters_raise_path_safety_error_only(layout: DataLayout, bad: str) -> None:
    make_root(layout)
    with pytest.raises(PathSafetyError):  # an OSError (or anything else) would fail this
        new_build_dir(layout, bad)
    assert listing(layout.root) == []


def test_build_id_pattern_is_strict_even_when_misused_with_match() -> None:
    from helmet_data.paths import BUILD_ID_PATTERN

    assert BUILD_ID_PATTERN.match("b1\n") is None
    assert BUILD_ID_PATTERN.fullmatch("b1\n") is None
    assert BUILD_ID_PATTERN.fullmatch("b1") is not None
    assert BUILD_ID_PATTERN.fullmatch("a" * 64) is not None
    assert BUILD_ID_PATTERN.fullmatch("a" * 65) is None


# --------------------------------------------------------------------------- #
# M1: deterministic no-overwrite race (no timing, no threads, no randomness)
# --------------------------------------------------------------------------- #
def test_destination_created_between_check_and_publish_is_never_overwritten(
    writable_layout: DataLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = writable_layout
    dest = layout.interim / "m.json"
    real_mkstemp = tempfile.mkstemp
    events: list[str] = []

    def competing_writer(*args, **kwargs):
        # write_file_atomic calls mkstemp after its existence pre-check and before it
        # publishes, so a file created here is exactly the race being defended against.
        assert not dest.exists(), "the pre-check must have run first"
        dest.write_bytes(b"competitor")
        events.append("competitor wrote the destination")
        return real_mkstemp(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "mkstemp", competing_writer)
        with pytest.raises(DestinationExistsError):
            write_file_atomic(layout, dest, b"mine")

    assert events == ["competitor wrote the destination"]
    assert dest.read_bytes() == b"competitor"
    assert listing(layout.interim) == ["m.json"]  # no temp file left behind


def test_race_loser_leaves_no_temp_file_even_for_a_large_payload(
    writable_layout: DataLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = writable_layout
    dest = layout.interim / "big.bin"
    real_mkstemp = tempfile.mkstemp

    def competing_writer(*args, **kwargs):
        dest.write_bytes(b"competitor")
        return real_mkstemp(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "mkstemp", competing_writer)
        with pytest.raises(DestinationExistsError):
            write_file_atomic(layout, dest, os.urandom(1024) * 2048)
    assert dest.read_bytes() == b"competitor"
    assert listing(layout.interim) == ["big.bin"]


def test_explicit_overwrite_still_replaces_a_competitor(
    writable_layout: DataLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = writable_layout
    dest = layout.interim / "m.json"
    real_mkstemp = tempfile.mkstemp

    def competing_writer(*args, **kwargs):
        dest.write_bytes(b"competitor")
        return real_mkstemp(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "mkstemp", competing_writer)
        write_file_atomic(layout, dest, b"mine", overwrite=True)
    assert dest.read_bytes() == b"mine"
    assert listing(layout.interim) == ["m.json"]


@windows_only
def test_paths_longer_than_max_path_inside_raw_are_blocked(layout: DataLayout) -> None:
    """realpath keeps a \\\\?\\ prefix on paths over 260 characters; it must not defeat the guard."""
    layout.raw.mkdir(parents=True)
    segment = "d" * 100
    deep = layout.raw / segment / segment / segment
    os.makedirs("\\\\?\\" + str(deep))
    try:
        target = deep / "x.txt"
        assert len(str(target)) > 260
        for spelling in (str(target), "\\\\?\\" + str(target), str(deep)):
            with pytest.raises(RawWriteError):
                assert_writable(spelling, layout)
        # a long path outside raw is still not mistaken for raw
        long_outside = layout.interim / segment / segment / segment / "x.txt"
        assert len(str(long_outside)) > 260
        assert_writable(long_outside, layout)
    finally:
        shutil.rmtree("\\\\?\\" + str(layout.raw / segment), ignore_errors=True)


@windows_only
def test_refusal_messages_identify_the_kind_of_unsupported_path(layout: DataLayout) -> None:
    layout.raw.mkdir(parents=True)
    target = layout.raw / "x.txt"
    spellings = unsupported_spellings(target)
    for label in ("device", "device forward slashes"):
        with pytest.raises(PathSafetyError, match="device paths are not supported"):
            assert_writable(spellings[label], layout)
    for label in ("extended unc", "extended unc lowercase", "nt namespace unc"):
        with pytest.raises(PathSafetyError, match="UNC paths are not supported"):
            assert_writable(spellings[label], layout)
    for label in ("admin share unc", "unc forward slashes", "unc server share"):
        with pytest.raises(PathSafetyError, match="UNC paths are not supported"):
            assert_writable(spellings[label], layout)
    for label in ("volume guid", "globalroot", "extended without colon", "bare extended prefix"):
        with pytest.raises(PathSafetyError, match="unsupported extended-length path"):
            assert_writable(spellings[label], layout)


@windows_only
def test_a_resolved_path_that_keeps_an_extended_prefix_is_still_compared_correctly(
    layout: DataLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows can keep \\\\?\\ on long resolved paths while the (shorter) raw root has none.

    Simulate that asymmetry deterministically: only the child path comes back prefixed.
    """
    layout.raw.mkdir(parents=True)
    layout.interim.mkdir()
    real_realpath = os.path.realpath

    def asymmetric(path, *args, **kwargs):
        resolved = real_realpath(path, *args, **kwargs)
        if str(resolved).endswith(("x.txt", "x.json")) and not str(resolved).startswith("\\\\?\\"):
            return "\\\\?\\" + str(resolved)
        return resolved

    with monkeypatch.context() as patch:
        patch.setattr(os.path, "realpath", asymmetric)
        with pytest.raises(RawWriteError):
            assert_writable(layout.raw / "x.txt", layout)
        with pytest.raises(RawWriteError):
            assert_writable(layout.raw / "sub" / "x.txt", layout)
        assert_writable(layout.interim / "x.json", layout)


@windows_only
def test_alternate_data_stream_spellings_cannot_authorise_raw_writes(layout: DataLayout) -> None:
    layout.raw.mkdir(parents=True)
    root = str(layout.root)
    spellings = [
        root + "\\raw::$DATA\\x.txt",
        root + "\\raw:stream\\x.txt",
        root + "\\raw::$INDEX_ALLOCATION\\x.txt",
        root + "\\raw:$I30:$INDEX_ALLOCATION\\x.txt",
        root + "\\raw\\x.txt::$DATA",
        root + "\\raw\\x.txt:ads",
        "\\\\?\\" + root + "\\raw::$INDEX_ALLOCATION\\x.txt",
    ]
    for spelling in spellings:
        with pytest.raises(PathSafetyError, match="alternate data stream"):
            assert_writable(spelling, layout)
    assert listing(layout.raw) == []


@windows_only
def test_drive_colon_and_ordinary_names_are_not_mistaken_for_stream_syntax(layout: DataLayout) -> None:
    make_root(layout)
    ensure_dir(layout, "interim")
    assert_writable(str(layout.interim / "a b" / "file.v2.json"), layout)
    assert_writable("\\\\?\\" + str(layout.interim / "m.json"), layout)
    with pytest.raises(PathSafetyError, match="alternate data stream"):
        open_raw(layout, str(layout.raw / "x.bin") + ":ads")
