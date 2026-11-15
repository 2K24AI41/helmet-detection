"""Tests for helmet_data.config."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from helmet_data.config import (
    ConfigError,
    DedupConfig,
    PipelineConfig,
    SplitConfig,
    config_hash,
    default_config,
    load_config,
    parse_config,
)


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_match_approved_values() -> None:
    cfg = default_config()
    assert cfg.train_imgsz == 640
    assert cfg.projection == "letterbox"
    assert cfg.min_projected_side_px == 16
    assert cfg.oob_tolerance_px == 2
    assert cfg.dhash_max_distance == 6
    assert cfg.max_group_fraction == 0.05
    assert cfg.dedup.box_match_iou == 0.95
    assert (cfg.split.train, cfg.split.val, cfg.split.test) == (0.8, 0.1, 0.1)
    assert cfg.split.seed == 42
    assert cfg.ledger.backup_dir is None


def test_repository_config_loads_and_equals_defaults(repo_root: Path) -> None:
    assert load_config(repo_root / "configs" / "pipeline.yaml") == default_config()


def test_overrides_are_applied(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        "train_imgsz: 320\nprojection: stretch\nmin_projected_side_px: 8.5\noob_tolerance_px: 0\n"
        "dhash_max_distance: 10\nmax_group_fraction: 0.2\ndedup: {box_match_iou: 0.9}\n"
        "split: {train: 0.7, val: 0.2, test: 0.1, seed: 7}\nledger: {backup_dir: 'E:/backups/ledger'}\n",
    )
    cfg = load_config(path)
    assert (cfg.train_imgsz, cfg.projection, cfg.min_projected_side_px) == (320, "stretch", 8.5)
    assert (cfg.oob_tolerance_px, cfg.dhash_max_distance, cfg.max_group_fraction) == (0, 10, 0.2)
    assert cfg.dedup.box_match_iou == 0.9
    assert (cfg.split.train, cfg.split.val, cfg.split.test, cfg.split.seed) == (0.7, 0.2, 0.1, 7)
    assert cfg.ledger.backup_dir == Path("E:/backups/ledger")


def test_partial_sections_keep_other_defaults(tmp_path: Path) -> None:
    cfg = load_config(write_yaml(tmp_path, "split: {seed: 99}\n"))
    assert cfg.split.seed == 99
    assert cfg.split.train == 0.8


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    assert load_config(write_yaml(tmp_path, "")) == default_config()


# --------------------------------------------------------------------------- #
# File access
# --------------------------------------------------------------------------- #
def test_missing_file_or_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path)


def test_undecodable_file_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"train_imgsz: \xff\xfe\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_os_errors_are_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_yaml(tmp_path, "train_imgsz: 640\n")

    def deny(self, *args, **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(Path, "open", deny)
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(path)


# --------------------------------------------------------------------------- #
# Keys, types, ranges
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "unknown_key: 1\n",
        "dedup: {nope: 1}\n",
        "split: {train: 0.8, val: 0.1, test: 0.1, sead: 1}\n",
        "ledger: {path: x}\n",
    ],
)
def test_unknown_keys_are_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write_yaml(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        "train_imgsz: '640'\n",
        "train_imgsz: true\n",
        "train_imgsz: 640.5\n",
        "min_projected_side_px: abc\n",
        "min_projected_side_px: true\n",
        "oob_tolerance_px: false\n",
        "max_group_fraction: true\n",
        "dhash_max_distance: 6.5\n",
        "dhash_max_distance: true\n",
        "projection: 5\n",
        "split: {seed: 1.5}\n",
        "split: {seed: true}\n",
        "split: {train: true, val: 0.1, test: 0.1}\n",
        "dedup: {box_match_iou: high}\n",
        "dedup: {box_match_iou: true}\n",
        "ledger: {backup_dir: 5}\n",
        "ledger: {backup_dir: ''}\n",
        "ledger: {backup_dir: '   '}\n",
        "- just\n- a list\n",
        "train_imgsz:\n",
    ],
)
def test_wrong_types_are_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_config(write_yaml(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        "split: {train: 0.5, val: 0.1, test: 0.1}\n",
        "split: {train: 0.9, val: 0.2, test: 0.1}\n",
        "split: {train: 1.1, val: -0.1, test: 0.0}\n",
        "split: {train: 0, val: 0.5, test: 0.5}\n",
        "split: {train: 0.5, val: 0, test: 0.5}\n",
        "split: {train: 0.5, val: 0.5, test: 0}\n",
        "split: {train: 1, val: 0, test: 0}\n",
        "train_imgsz: 0\n",
        "train_imgsz: -640\n",
        "projection: crop\n",
        "min_projected_side_px: 0\n",
        "min_projected_side_px: -1\n",
        "oob_tolerance_px: -0.5\n",
        "dhash_max_distance: -1\n",
        "dhash_max_distance: 65\n",
        "max_group_fraction: 0\n",
        "max_group_fraction: 1.5\n",
        "dedup: {box_match_iou: 0}\n",
        "dedup: {box_match_iou: 1.2}\n",
        "min_projected_side_px: .nan\n",
        "min_projected_side_px: .inf\n",
    ],
)
def test_invalid_ranges_are_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_config(write_yaml(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        "dhash_max_distance: 0\n",
        "dhash_max_distance: 64\n",
        "max_group_fraction: 1\n",
        "dedup: {box_match_iou: 1}\n",
        "oob_tolerance_px: 0\n",
        "split: {train: 0.98, val: 0.01, test: 0.01}\n",
    ],
)
def test_boundary_values_are_accepted(tmp_path: Path, text: str) -> None:
    assert isinstance(load_config(write_yaml(tmp_path, text)), PipelineConfig)


@pytest.mark.parametrize(
    "text",
    ["dedup:\n", "dedup: {}\n", "dedup: 5\n", "dedup: [1]\n", "split:\n", "split: {}\n", "split: abc\n",
     "ledger:\n", "ledger: {}\n", "ledger: [x]\n"],
)
def test_empty_or_mistyped_sections_are_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_config(write_yaml(tmp_path, text))


# --------------------------------------------------------------------------- #
# Duplicate keys and unsafe YAML
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "train_imgsz: 320\ntrain_imgsz: 640\n",
        "split: {train: 0.8, val: 0.1, test: 0.1, seed: 1, seed: 2}\n",
        "dedup:\n  box_match_iou: 0.9\n  box_match_iou: 0.8\n",
        "split: {train: 0.8, val: 0.1, test: 0.1}\nsplit: {train: 0.7, val: 0.2, test: 0.1}\n",
    ],
)
def test_duplicate_yaml_keys_are_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="duplicate key"):
        load_config(write_yaml(tmp_path, text))


def test_unsafe_yaml_tags_are_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "train_imgsz: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_invalid_yaml_syntax_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write_yaml(tmp_path, "split: {train: 0.8\n"))


# --------------------------------------------------------------------------- #
# Direct construction, immutability
# --------------------------------------------------------------------------- #
def test_directly_constructed_invalid_config_is_rejected() -> None:
    with pytest.raises(ConfigError):
        PipelineConfig(train_imgsz=0)
    with pytest.raises(ConfigError):
        PipelineConfig(dhash_max_distance=True)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        PipelineConfig(dedup="x")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        PipelineConfig(split=None)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        SplitConfig(train=0.0, val=0.5, test=0.5)
    with pytest.raises(ConfigError):
        DedupConfig(box_match_iou=2)


def test_replace_revalidates() -> None:
    with pytest.raises(ConfigError):
        dataclasses.replace(default_config(), train_imgsz=-1)


def test_config_is_immutable_and_hashable() -> None:
    cfg = default_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.train_imgsz = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.split.seed = 1  # type: ignore[misc]
    assert hash(cfg) == hash(default_config())


# --------------------------------------------------------------------------- #
# Canonical numbers and hashing
# --------------------------------------------------------------------------- #
def test_numeric_spellings_are_canonical() -> None:
    ints = parse_config(
        {"min_projected_side_px": 16, "oob_tolerance_px": 2, "max_group_fraction": 1,
         "dedup": {"box_match_iou": 1}}
    )
    floats = parse_config(
        {"min_projected_side_px": 16.0, "oob_tolerance_px": 2.0, "max_group_fraction": 1.0,
         "dedup": {"box_match_iou": 1.0}}
    )
    assert ints == floats
    assert hash(ints) == hash(floats)
    assert config_hash(ints) == config_hash(floats)
    assert isinstance(ints.min_projected_side_px, float)
    assert isinstance(ints.split.train, float)
    assert isinstance(ints.train_imgsz, int)


def test_default_hash_equals_hash_of_explicit_integer_spelling() -> None:
    explicit = parse_config({"min_projected_side_px": 16, "oob_tolerance_px": 2})
    assert config_hash(explicit) == config_hash(default_config())


def test_negative_zero_is_normalised() -> None:
    assert config_hash(parse_config({"oob_tolerance_px": -0.0})) == config_hash(
        parse_config({"oob_tolerance_px": 0})
    )


def test_split_ratio_spellings_hash_identically() -> None:
    a = parse_config({"split": {"train": 0.5, "val": 0.25, "test": 0.25}})
    b = parse_config({"split": {"train": 0.50, "val": 0.250, "test": 0.25}})
    assert config_hash(a) == config_hash(b)


def test_config_hash_is_stable_and_value_sensitive() -> None:
    base = default_config()
    assert config_hash(base) == config_hash(default_config())
    assert len(config_hash(base)) == 64
    for override in (
        {"split": {"seed": 43}},
        {"dedup": {"box_match_iou": 0.9}},
        {"train_imgsz": 320},
        {"projection": "stretch"},
        {"min_projected_side_px": 17},
        {"dhash_max_distance": 7},
        {"max_group_fraction": 0.06},
        {"oob_tolerance_px": 3},
    ):
        assert config_hash(parse_config(override)) != config_hash(base), override


def test_operational_backup_dir_does_not_affect_build_identity() -> None:
    base = config_hash(default_config())
    for spelling in ("E:/x", "E:\\x", "e:/x/", "D:/somewhere/else", None):
        cfg = parse_config({"ledger": {"backup_dir": spelling}})
        assert config_hash(cfg) == base
    assert parse_config({"ledger": {"backup_dir": "E:/x"}}) != default_config()


def test_ledger_backup_dir_accepts_null_or_path() -> None:
    assert parse_config({"ledger": {"backup_dir": None}}).ledger.backup_dir is None
    assert parse_config({"ledger": {"backup_dir": "D:/somewhere"}}).ledger.backup_dir == Path("D:/somewhere")
    with pytest.raises(ConfigError):
        parse_config({"ledger": {"backup_dir": "a\x00b"}})


# --------------------------------------------------------------------------- #
# M3: ambiguous YAML 1.1 numeric forms are rejected, never silently reinterpreted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "train_imgsz: 010\n",
        "train_imgsz: 0640\n",
        "train_imgsz: -010\n",
        "train_imgsz: +010\n",
        "train_imgsz: 00\n",
        "train_imgsz: 0_7\n",
        "dhash_max_distance: 06\n",
        "split: {train: 0.8, val: 0.1, test: 0.1, seed: 010}\n",
        "train_imgsz: 6:40\n",
        "train_imgsz: 1:30\n",
        "train_imgsz: -6:40\n",
        "dhash_max_distance: 9:30\n",
        "min_projected_side_px: 1:30.5\n",
        "oob_tolerance_px: 1:30:15.25\n",
    ],
)
def test_ambiguous_legacy_yaml_numbers_are_rejected_with_a_clear_error(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="ambiguous legacy YAML 1.1"):
        load_config(write_yaml(tmp_path, text))


def test_ambiguous_number_error_names_the_literal_and_the_alternative(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_config(write_yaml(tmp_path, "train_imgsz: 0640\n"))
    message = str(info.value)
    assert "'0640'" in message and "octal" in message and "plain decimal" in message
    with pytest.raises(ConfigError) as info:
        load_config(write_yaml(tmp_path, "train_imgsz: 6:40\n"))
    assert "'6:40'" in str(info.value) and "sexagesimal" in str(info.value)


@pytest.mark.parametrize(
    ("text", "attribute", "expected"),
    [
        ("train_imgsz: 640\n", "train_imgsz", 640),
        ("train_imgsz: 1_000\n", "train_imgsz", 1000),
        ("train_imgsz: 0x280\n", "train_imgsz", 640),
        ("train_imgsz: 32\n", "train_imgsz", 32),
        ("dhash_max_distance: 0\n", "dhash_max_distance", 0),
        ("dhash_max_distance: 6\n", "dhash_max_distance", 6),
        ("min_projected_side_px: 16\n", "min_projected_side_px", 16.0),
        ("min_projected_side_px: 010.5\n", "min_projected_side_px", 10.5),
        ("min_projected_side_px: 0.5\n", "min_projected_side_px", 0.5),
        ("oob_tolerance_px: 0\n", "oob_tolerance_px", 0.0),
    ],
)
def test_normal_numbers_are_unchanged(tmp_path: Path, text: str, attribute: str, expected) -> None:
    value = getattr(load_config(write_yaml(tmp_path, text)), attribute)
    assert value == expected
    assert type(value) is type(expected)


def test_seed_zero_and_negative_seed_are_still_accepted(tmp_path: Path) -> None:
    assert load_config(write_yaml(tmp_path, "split: {seed: 0}\n")).split.seed == 0
    assert load_config(write_yaml(tmp_path, "split: {seed: -5}\n")).split.seed == -5


def test_quoted_strings_are_untouched_by_the_number_check(tmp_path: Path) -> None:
    cfg = load_config(write_yaml(tmp_path, "ledger: {backup_dir: '6:40'}\n"))
    assert cfg.ledger.backup_dir == Path("6:40")
    with pytest.raises(ConfigError):  # a quoted number is a string, so still a type error
        load_config(write_yaml(tmp_path, "train_imgsz: '010'\n"))


@pytest.mark.parametrize("text", ["train_imgsz: 08\n", "train_imgsz: 09\n", "dhash_max_distance: 0:30\n"])
def test_forms_that_yaml_reads_as_strings_are_still_rejected_as_type_errors(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="must be an integer"):
        load_config(write_yaml(tmp_path, text))


def test_ambiguous_numbers_are_rejected_inside_nested_sections_and_keys(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="ambiguous legacy YAML 1.1"):
        load_config(write_yaml(tmp_path, "dedup: {box_match_iou: 1:0}\n"))
    with pytest.raises(ConfigError, match="ambiguous legacy YAML 1.1"):
        load_config(write_yaml(tmp_path, "split: {train: 0.8, val: 0.1, test: 0.1}\nledger: {backup_dir: x}\n010: 1\n"))
