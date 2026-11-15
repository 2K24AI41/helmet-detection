"""Tests for helmet_data.integrity."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from helmet_data import integrity
from helmet_data.integrity import (
    IntegrityError,
    Manifest,
    ManifestEntry,
    UnsafeRawTree,
    build_manifest,
    manifest_bytes,
    manifest_sha256,
    read_manifest,
    verify_manifest,
    write_manifest,
)
from helmet_data.paths import (
    DataLayout,
    DestinationExistsError,
    PathSafetyError,
    ensure_dir,
)

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
ZERO = "0" * 64


def snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Contents and mtimes of every file below root."""
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def entry(path: str, size: int = 1, sha: str = ZERO) -> ManifestEntry:
    return ManifestEntry(path, size, sha)


@pytest.fixture
def interim(layout: DataLayout) -> DataLayout:
    layout.root.mkdir(parents=True, exist_ok=True)
    ensure_dir(layout, "interim", "demo")
    return layout


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def test_manifest_has_sorted_posix_entries_with_known_hashes(raw_dir: Path) -> None:
    manifest = build_manifest(raw_dir)
    paths = [e.path for e in manifest.entries]
    assert paths == sorted(paths)
    assert set(paths) == {".hidden", "café.txt", "empty.txt", "images/a.bin", "images/sub/b.bin"}
    by_path = {e.path: e for e in manifest.entries}
    assert by_path["images/a.bin"].sha256 == ABC_SHA256
    assert by_path["images/a.bin"].size == 3
    assert by_path["empty.txt"].sha256 == EMPTY_SHA256
    assert by_path[".hidden"].size == 6
    assert all("\\" not in p for p in paths)
    assert manifest.empty_dirs == ()


def test_manifest_is_independent_of_creation_order(tmp_path: Path) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    for root, names in ((first, ["a", "b", "c"]), (second, ["c", "a", "b"])):
        root.mkdir()
        for name in names:
            (root / f"{name}.txt").write_bytes(name.encode())
    assert build_manifest(first) == build_manifest(second)
    assert manifest_bytes(build_manifest(first)) == manifest_bytes(build_manifest(second))


def test_manifest_hashes_large_file_in_chunks(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    payload = os.urandom(1024) * (5 * 1024)  # 5 MiB
    (root / "big.bin").write_bytes(payload)
    found = build_manifest(root).entries[0]
    assert (found.size, found.sha256) == (len(payload), hashlib.sha256(payload).hexdigest())


def test_empty_tree_gives_empty_manifest(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    assert build_manifest(root) == Manifest(entries=())


def test_missing_raw_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError, match="not found"):
        build_manifest(tmp_path / "missing")
    file_path = tmp_path / "file.txt"
    file_path.write_bytes(b"x")
    with pytest.raises(IntegrityError):
        build_manifest(file_path)


def test_building_and_verifying_leave_raw_untouched(raw_dir: Path) -> None:
    before = snapshot(raw_dir)
    listing_before = sorted(p.relative_to(raw_dir).as_posix() for p in raw_dir.rglob("*"))
    manifest = build_manifest(raw_dir)
    verify_manifest(raw_dir, manifest)
    manifest_sha256(manifest)
    assert snapshot(raw_dir) == before
    assert sorted(p.relative_to(raw_dir).as_posix() for p in raw_dir.rglob("*")) == listing_before


def test_copying_tree_gives_same_manifest(raw_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(raw_dir, copy)
    assert build_manifest(copy) == build_manifest(raw_dir)


# --------------------------------------------------------------------------- #
# Failures are never silent (H2)
# --------------------------------------------------------------------------- #
def test_unreadable_directory_raises_instead_of_being_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw"
    (raw / "ok").mkdir(parents=True)
    (raw / "locked").mkdir()
    (raw / "ok" / "f.txt").write_bytes(b"1")
    (raw / "locked" / "g.txt").write_bytes(b"2")
    real_scandir = os.scandir

    def flaky(path="."):
        if str(path).endswith("locked"):
            raise PermissionError(13, "simulated access denied", str(path))
        return real_scandir(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", flaky)
        with pytest.raises(IntegrityError, match="cannot read directory"):
            build_manifest(raw)


def test_unreadable_root_directory_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    def denied(path="."):
        raise PermissionError(13, "simulated access denied", str(path))

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", denied)
        with pytest.raises(IntegrityError, match="cannot read directory"):
            build_manifest(raw)


def test_unreadable_file_raises_instead_of_being_skipped(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_open(path, mode="r", *args, **kwargs):
        if str(path).endswith("a.bin"):
            raise PermissionError(13, "simulated access denied", str(path))
        return open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(integrity, "open", fake_open, raising=False)
        with pytest.raises(IntegrityError, match="cannot read file"):
            build_manifest(raw_dir)


def test_file_that_disappears_during_hashing_raises(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_open(path, mode="r", *args, **kwargs):
        raise FileNotFoundError(2, "vanished", str(path))

    with monkeypatch.context() as patch:
        patch.setattr(integrity, "open", fake_open, raising=False)
        with pytest.raises(IntegrityError):
            build_manifest(raw_dir)


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #
def test_directory_link_in_raw_tree_is_refused(raw_dir: Path, tmp_path: Path, dir_link) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "f.txt").write_bytes(b"x")
    dir_link(elsewhere, raw_dir / "linkdir")
    with pytest.raises(UnsafeRawTree):
        build_manifest(raw_dir)


def test_junction_in_raw_tree_is_refused(raw_dir: Path, tmp_path: Path, junction_to) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "f.txt").write_bytes(b"x")
    junction_to(elsewhere, raw_dir / "jlink")
    with pytest.raises(UnsafeRawTree):
        build_manifest(raw_dir)


def test_junction_as_raw_root_is_refused(tmp_path: Path, junction_to) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "f.txt").write_bytes(b"x")
    link = junction_to(real, tmp_path / "linked_raw")
    with pytest.raises(UnsafeRawTree):
        build_manifest(link)


def test_symlinked_file_in_raw_tree_is_refused(raw_dir: Path, tmp_path: Path, symlink_to) -> None:
    target = tmp_path / "elsewhere.txt"
    target.write_bytes(b"x")
    symlink_to(target, raw_dir / "link.txt")
    with pytest.raises(UnsafeRawTree):
        build_manifest(raw_dir)


def test_symlinked_raw_root_is_refused(tmp_path: Path, symlink_to) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = symlink_to(real, tmp_path / "linked_raw", directory=True)
    with pytest.raises(UnsafeRawTree):
        build_manifest(link)


# --------------------------------------------------------------------------- #
# Empty directories (L5): they ARE part of the manifest
# --------------------------------------------------------------------------- #
def test_empty_directories_are_recorded(raw_dir: Path) -> None:
    (raw_dir / "empty_leaf").mkdir()
    (raw_dir / "images" / "sub" / "deep_empty").mkdir()
    manifest = build_manifest(raw_dir)
    assert manifest.empty_dirs == ("empty_leaf", "images/sub/deep_empty")


def test_adding_an_empty_directory_is_detected(raw_dir: Path) -> None:
    manifest = build_manifest(raw_dir)
    (raw_dir / "new_empty_dir").mkdir()
    diff = verify_manifest(raw_dir, manifest)
    assert not diff.ok
    assert diff.added_empty_dirs == ("new_empty_dir",)
    assert diff.removed_empty_dirs == diff.added == diff.removed == diff.modified == ()


def test_removing_an_empty_directory_is_detected(raw_dir: Path) -> None:
    (raw_dir / "empty_leaf").mkdir()
    manifest = build_manifest(raw_dir)
    (raw_dir / "empty_leaf").rmdir()
    diff = verify_manifest(raw_dir, manifest)
    assert diff.removed_empty_dirs == ("empty_leaf",)
    assert not diff.ok


def test_nested_empty_chain_reports_only_the_leaf(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    (root / "a" / "b" / "c").mkdir(parents=True)
    assert build_manifest(root).empty_dirs == ("a/b/c",)


def test_root_itself_is_not_listed_as_empty_directory(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    assert build_manifest(root).empty_dirs == ()


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def test_unchanged_tree_verifies(raw_dir: Path) -> None:
    diff = verify_manifest(raw_dir, build_manifest(raw_dir))
    assert diff.ok
    assert (diff.added, diff.removed, diff.modified) == ((), (), ())


def test_same_size_content_change_is_detected(raw_dir: Path) -> None:
    manifest = build_manifest(raw_dir)
    (raw_dir / "images" / "a.bin").write_bytes(b"xyz")  # same size as b"abc"
    diff = verify_manifest(raw_dir, manifest)
    assert diff.modified == ("images/a.bin",)
    assert diff.added == diff.removed == ()


def test_size_change_is_detected(raw_dir: Path) -> None:
    manifest = build_manifest(raw_dir)
    (raw_dir / "images" / "a.bin").write_bytes(b"abcd")
    assert verify_manifest(raw_dir, manifest).modified == ("images/a.bin",)


def test_added_and_removed_files_are_detected(raw_dir: Path) -> None:
    manifest = build_manifest(raw_dir)
    (raw_dir / "new.txt").write_bytes(b"new")
    (raw_dir / "empty.txt").unlink()
    diff = verify_manifest(raw_dir, manifest)
    assert diff.added == ("new.txt",)
    assert diff.removed == ("empty.txt",)


def test_rename_shows_as_removed_plus_added(raw_dir: Path) -> None:
    manifest = build_manifest(raw_dir)
    (raw_dir / "images" / "a.bin").rename(raw_dir / "images" / "renamed.bin")
    diff = verify_manifest(raw_dir, manifest)
    assert (diff.removed, diff.added, diff.modified) == (("images/a.bin",), ("images/renamed.bin",), ())


def test_manifest_sha256_changes_with_content(raw_dir: Path) -> None:
    before = manifest_sha256(build_manifest(raw_dir))
    (raw_dir / "images" / "a.bin").write_bytes(b"xyz")
    assert manifest_sha256(build_manifest(raw_dir)) != before


# --------------------------------------------------------------------------- #
# Manifest validation (M4)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": -1},
        {"size": True},
        {"size": 1.5},
        {"size": "3"},
        {"size": None},
        {"sha": "zz"},
        {"sha": "A" * 64},
        {"sha": "a" * 63},
        {"sha": 5},
        {"path": "../x"},
        {"path": "/abs"},
        {"path": "a\\b"},
        {"path": "a//b"},
        {"path": ""},
        {"path": "a:b"},
        {"path": "./a"},
        {"path": 5},
    ],
)
def test_manifest_entry_validation(kwargs) -> None:
    values = {"path": "a.bin", "size": 1, "sha": ZERO}
    values.update(kwargs)
    with pytest.raises(IntegrityError):
        ManifestEntry(values["path"], values["size"], values["sha"])


def test_manifest_requires_canonical_sorted_unique_entries() -> None:
    assert Manifest(entries=(entry("a"), entry("b")))
    with pytest.raises(IntegrityError):
        Manifest(entries=(entry("b"), entry("a")))
    with pytest.raises(IntegrityError):
        Manifest(entries=(entry("a"), entry("a")))
    with pytest.raises(IntegrityError):
        Manifest(entries=[entry("a")])  # type: ignore[arg-type]
    with pytest.raises(IntegrityError):
        Manifest(entries=("a",))  # type: ignore[arg-type]
    with pytest.raises(IntegrityError):
        Manifest(entries=(), empty_dirs=("b", "a"))
    with pytest.raises(IntegrityError):
        Manifest(entries=(), empty_dirs=("a", "a"))
    with pytest.raises(IntegrityError):
        Manifest(entries=(), empty_dirs=("../x",))
    with pytest.raises(IntegrityError):
        Manifest(entries=(), version=2)
    with pytest.raises(IntegrityError):
        Manifest(entries=(), version=True)  # type: ignore[arg-type]


def test_manifest_rejects_empty_dir_clashing_with_files() -> None:
    with pytest.raises(IntegrityError):
        Manifest(entries=(entry("a"),), empty_dirs=("a",))
    with pytest.raises(IntegrityError):
        Manifest(entries=(entry("a/b.txt"),), empty_dirs=("a",))
    assert Manifest(entries=(entry("a/b.txt"),), empty_dirs=("a/c",))


def test_manifests_are_hashable_and_equal_by_value() -> None:
    first = Manifest(entries=(entry("a"),), empty_dirs=("d",))
    assert hash(first) == hash(Manifest(entries=(entry("a"),), empty_dirs=("d",)))


def canonical_text(files, empty_dirs=(), version=1) -> str:
    return json.dumps(
        {"version": version, "files": files, "empty_dirs": list(empty_dirs)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n"


def file_item(path="a", size=1, sha=ZERO):
    return {"path": path, "size": size, "sha256": sha}


def write_text(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "m.json"
    path.write_bytes(text.encode("utf-8"))
    return path


def test_read_manifest_accepts_canonical_text(tmp_path: Path) -> None:
    path = write_text(tmp_path, canonical_text([file_item("a"), file_item("b")], ["d"]))
    assert read_manifest(path) == Manifest(entries=(entry("a"), entry("b")), empty_dirs=("d",))


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        "null",
        "",
        canonical_text([], version=99),
        canonical_text([file_item(size=-5)]),
        canonical_text([file_item(size=1.5)]),
        canonical_text([file_item(size="3")]),
        canonical_text([file_item(size=True)]),
        canonical_text([file_item(sha="zz")]),
        canonical_text([file_item(sha="A" * 64)]),
        canonical_text([file_item(path="../../x")]),
        canonical_text([file_item(path="/abs")]),
        canonical_text([file_item("b"), file_item("a")]),  # not sorted
        canonical_text([file_item("a"), file_item("a")]),  # duplicate path
        canonical_text([], ["b", "a"]),
        canonical_text([file_item("a")], ["a"]),
        canonical_text([{"path": "a", "size": 1}]),  # missing key
        canonical_text([{"path": "a", "size": 1, "sha256": ZERO, "extra": 1}]),  # extra key
        '{"version":1,"files":[]}\n',  # missing empty_dirs
        '{"version":1,"files":[],"empty_dirs":[],"extra":1}\n',
        '{"version":true,"files":[],"empty_dirs":[]}\n',
        '{"version":1,"files":{},"empty_dirs":[]}\n',
        '{"version":1,"files":[],"empty_dirs":[5]}\n',
        '{"version":1,"version":1,"files":[],"empty_dirs":[]}\n',  # duplicate JSON key
    ],
)
def test_read_manifest_rejects_malformed_or_invalid_manifests(tmp_path: Path, text: str) -> None:
    with pytest.raises(IntegrityError):
        read_manifest(write_text(tmp_path, text))


def test_read_manifest_rejects_non_canonical_formatting(tmp_path: Path) -> None:
    pretty = json.dumps(json.loads(canonical_text([file_item("a")])), indent=2) + "\n"
    with pytest.raises(IntegrityError, match="canonical"):
        read_manifest(write_text(tmp_path, pretty))
    with pytest.raises(IntegrityError, match="canonical"):
        read_manifest(write_text(tmp_path, canonical_text([file_item("a")]).rstrip("\n")))
    with pytest.raises(IntegrityError, match="canonical"):
        read_manifest(write_text(tmp_path, canonical_text([file_item("a")]) + "\n"))


def test_read_manifest_rejects_invalid_utf8_and_missing_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(IntegrityError):
        read_manifest(bad)
    with pytest.raises(IntegrityError):
        read_manifest(tmp_path / "missing.json")


def test_same_files_in_a_different_order_cannot_produce_a_different_hash() -> None:
    with pytest.raises(IntegrityError):
        Manifest(entries=(entry("b"), entry("a")))


# --------------------------------------------------------------------------- #
# Writing manifests (M1)
# --------------------------------------------------------------------------- #
def test_manifest_round_trip_and_byte_identical_rewrite(interim: DataLayout, raw_dir: Path) -> None:
    (raw_dir / "empty_leaf").mkdir()
    manifest = build_manifest(raw_dir)
    first = write_manifest(manifest, interim.interim / "demo" / "raw_manifest.json", interim)
    assert read_manifest(first) == manifest
    second = write_manifest(read_manifest(first), interim.interim / "demo" / "again.json", interim)
    assert first.read_bytes() == second.read_bytes()
    assert manifest_sha256(manifest) == hashlib.sha256(first.read_bytes()).hexdigest()


def test_write_manifest_refuses_to_overwrite_a_baseline_by_default(interim: DataLayout, raw_dir: Path) -> None:
    baseline = build_manifest(raw_dir)
    dest = interim.interim / "demo" / "baseline.json"
    write_manifest(baseline, dest, interim)
    (raw_dir / "images" / "a.bin").write_bytes(b"TAMPERED")
    with pytest.raises(DestinationExistsError):
        write_manifest(build_manifest(raw_dir), dest, interim)
    assert read_manifest(dest) == baseline


def test_write_manifest_overwrites_only_when_explicitly_requested(interim: DataLayout, raw_dir: Path) -> None:
    dest = interim.interim / "demo" / "m.json"
    write_manifest(build_manifest(raw_dir), dest, interim)
    (raw_dir / "new.txt").write_bytes(b"n")
    updated = build_manifest(raw_dir)
    write_manifest(updated, dest, interim, overwrite=True)
    assert read_manifest(dest) == updated


def test_write_manifest_refuses_destination_inside_raw(interim: DataLayout, raw_dir: Path) -> None:
    manifest = build_manifest(raw_dir)
    before = snapshot(interim.raw)
    for dest in (raw_dir / "manifest.json", interim.raw / "m.json", interim.raw / "sub" / "m.json"):
        with pytest.raises(PathSafetyError):
            write_manifest(manifest, dest, interim)
    assert snapshot(interim.raw) == before
    assert not (raw_dir / "manifest.json").exists()


def test_write_manifest_refuses_destinations_outside_interim(interim: DataLayout, raw_dir: Path, tmp_path: Path) -> None:
    manifest = build_manifest(raw_dir)
    outside = tmp_path / "not_under_layout"
    outside.mkdir()
    for dest in (outside / "m.json", interim.root / "m.json", interim.reports / "m.json", interim.processed / "m.json"):
        with pytest.raises(PathSafetyError):
            write_manifest(manifest, dest, interim)
    assert list(outside.iterdir()) == []
    assert not (interim.root / "m.json").exists()


def test_write_manifest_refuses_traversal(interim: DataLayout, raw_dir: Path) -> None:
    manifest = build_manifest(raw_dir)
    with pytest.raises(PathSafetyError):
        write_manifest(manifest, interim.interim / ".." / "m.json", interim)
    with pytest.raises(PathSafetyError):
        write_manifest(manifest, interim.interim / "demo" / ".." / ".." / "raw" / "m.json", interim)


def test_write_manifest_does_not_create_parent_trees(interim: DataLayout, raw_dir: Path) -> None:
    with pytest.raises(PathSafetyError):
        write_manifest(build_manifest(raw_dir), interim.interim / "missing" / "deep" / "m.json", interim)
    assert not (interim.interim / "missing").exists()


def test_write_manifest_refuses_directory_link_escape(interim: DataLayout, raw_dir: Path, tmp_path: Path, dir_link) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    dir_link(outside, interim.interim / "linked")
    with pytest.raises(PathSafetyError):
        write_manifest(build_manifest(raw_dir), interim.interim / "linked" / "m.json", interim)
    assert list(outside.iterdir()) == []


def test_write_manifest_refuses_junction_into_raw(interim: DataLayout, raw_dir: Path, junction_to) -> None:
    junction_to(interim.raw, interim.interim / "sneaky")
    before = snapshot(interim.raw)
    with pytest.raises(PathSafetyError):
        write_manifest(build_manifest(raw_dir), interim.interim / "sneaky" / "m.json", interim)
    assert snapshot(interim.raw) == before
    assert not (raw_dir / "m.json").exists()


def test_failed_manifest_write_leaves_nothing_behind(interim: DataLayout, raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise OSError("simulated failure")

    dest = interim.interim / "demo" / "m.json"
    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", boom)
        with pytest.raises(OSError, match="simulated failure"):
            write_manifest(build_manifest(raw_dir), dest, interim)
    assert not dest.exists()
    assert list((interim.interim / "demo").iterdir()) == []


# --------------------------------------------------------------------------- #
# M2: absurdly nested JSON must surface as IntegrityError, not RecursionError
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "[" * 100000 + "]" * 100000,
        '{"a":' * 100000 + "1" + "}" * 100000,
        '{"version":1,"files":' + "[" * 100000 + "]" * 100000 + ',"empty_dirs":[]}',
    ],
    ids=["nested-array", "nested-object", "nested-inside-valid-keys"],
)
def test_deeply_nested_json_raises_integrity_error(tmp_path: Path, text: str) -> None:
    path = tmp_path / "deep.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(IntegrityError, match="cannot read manifest"):
        read_manifest(path)


# --------------------------------------------------------------------------- #
# M1 (manifest level): a destination created between the check and publication
# --------------------------------------------------------------------------- #
def test_manifest_destination_created_during_write_is_never_overwritten(
    interim: DataLayout, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    dest = interim.interim / "demo" / "baseline.json"
    real_mkstemp = tempfile.mkstemp

    def competing_writer(*args, **kwargs):
        dest.write_bytes(b"trusted baseline written by someone else")
        return real_mkstemp(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "mkstemp", competing_writer)
        with pytest.raises(DestinationExistsError):
            write_manifest(build_manifest(raw_dir), dest, interim)
    assert dest.read_bytes() == b"trusted baseline written by someone else"
    assert [p.name for p in dest.parent.iterdir()] == ["baseline.json"]
