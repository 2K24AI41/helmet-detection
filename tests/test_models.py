"""Tests for helmet_data.models."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re

import pytest

from helmet_data.models import (
    REASON_CODE_DOCS,
    AnnotationStatus,
    BoxSemantics,
    Decision,
    Group,
    InvalidBoxError,
    OutputRecord,
    PixelBox,
    ReasonCode,
    SourceAnnotation,
    SourceRecord,
    Split,
    TargetClass,
    source_record_id,
    validate_dataset_name,
    validate_relative_posix_path,
    validate_sha256,
)

A = "a" * 64
B = "b" * 64

#: The exact, deliberately reviewed set of reason codes. Adding or removing a code
#: must be a conscious change to this test, to REASON_CODE_DOCS and to the design.
APPROVED_REASON_CODES = frozenset(
    {
        "small_projected_box",
        "out_of_bounds_minor",
        "out_of_bounds_major",
        "invalid_box",
        "invalid_class_id",
        "contains_rejected_box",
        "has_excluded_annotation",
        "exact_duplicate",
        "exact_duplicate_conflict",
        "exact_duplicate_unresolved",
        "exif_orientation",
        "unsupported_format",
        "corrupt_image",
        "possible_multi_head",
        "class_conflict",
        "oversized_group",
        "low_variance_image",
        "unknown_semantics",
        "unlabeled",
        "rider_without_head_box",
        "occupant_count_mismatch",
        "possible_non_occupant",
        "linked_to_frozen_test",
        "ledger_conflict",
        "excluded_unreviewed",
        "iscrowd",
    }
)


def record(path: str = "train/a.jpg", sha: str = A, dataset: str = "ds") -> SourceRecord:
    return SourceRecord(dataset=dataset, relative_path=path, sha256=sha, size_bytes=10)


# --------------------------------------------------------------------------- #
# Enums and reason codes
# --------------------------------------------------------------------------- #
def test_target_class_ids_are_fixed() -> None:
    assert TargetClass.WITH_HELMET == 0
    assert TargetClass.WITHOUT_HELMET == 1
    assert TargetClass.WITH_HELMET.label == "with_helmet"
    assert TargetClass.WITHOUT_HELMET.label == "without_helmet"
    assert len(TargetClass) == 2


def test_enum_values_are_stable() -> None:
    assert [d.value for d in Decision] == ["accept", "review", "reject"]
    assert [s.value for s in Split] == ["train", "val", "test"]
    assert [s.value for s in AnnotationStatus] == ["mapped", "excluded", "rejected", "review"]
    assert {b.value for b in BoxSemantics} == {
        "head_region_single",
        "helmet_only",
        "whole_rider",
        "vehicle",
        "other",
        "unknown",
    }


@pytest.mark.parametrize("member", [*Decision, *Split, *AnnotationStatus, *BoxSemantics, *ReasonCode])
def test_string_enums_serialise_as_their_value_everywhere(member) -> None:
    assert str(member) == member.value
    assert format(member) == member.value
    assert f"{member}" == member.value
    assert f"{member:>40}" == f"{member.value:>40}"
    assert json.dumps({"k": member}) == json.dumps({"k": member.value})


def test_reason_code_set_is_pinned_exactly() -> None:
    assert {r.value for r in ReasonCode} == APPROVED_REASON_CODES
    assert len(ReasonCode) == 26


def test_every_reason_code_is_documented() -> None:
    assert set(REASON_CODE_DOCS) == set(ReasonCode)
    assert all(isinstance(text, str) and len(text) > 20 for text in REASON_CODE_DOCS.values())


def test_reason_codes_are_unique_lower_snake_case() -> None:
    values = [r.value for r in ReasonCode]
    assert len(values) == len(set(values))
    assert all(re.fullmatch(r"[a-z]+(_[a-z]+)*", v) for v in values)


def test_exact_duplicate_codes_are_distinguished_in_their_documentation() -> None:
    assert "Outcome label" in REASON_CODE_DOCS[ReasonCode.EXACT_DUPLICATE]
    assert "NOT equivalent" in REASON_CODE_DOCS[ReasonCode.EXACT_DUPLICATE_CONFLICT]
    assert "ARE equivalent" in REASON_CODE_DOCS[ReasonCode.EXACT_DUPLICATE_UNRESOLVED]


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("good", ["a", "ds", "helmet-set_2", "0abc", "a" * 64])
def test_valid_dataset_names(good: str) -> None:
    assert validate_dataset_name(good) == good


@pytest.mark.parametrize("bad", ["", "Upper", "a b", "a/b", "a\\b", "-x", "_x", "a" * 65, "caf\u00e9", "a\nb", None, 5])
def test_invalid_dataset_names(bad) -> None:
    with pytest.raises(ValueError):
        validate_dataset_name(bad)


@pytest.mark.parametrize("bad", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, None, 5, b"a" * 64])
def test_invalid_sha256(bad) -> None:
    with pytest.raises(ValueError):
        validate_sha256(bad)


def test_valid_sha256() -> None:
    assert validate_sha256(hashlib.sha256(b"x").hexdigest())


@pytest.mark.parametrize("good", ["a.jpg", "train/a.jpg", "a b/caf\u00e9.png", ".hidden"])
def test_valid_relative_paths(good: str) -> None:
    assert validate_relative_posix_path(good) == good


@pytest.mark.parametrize(
    "bad", ["", "/a", "a/", "a//b", "../a", "a/../b", "./a", "a/./b", "a\\b", "C:/a", "a:b", "a\x00b", "a\nb", None, 5]
)
def test_invalid_relative_paths(bad) -> None:
    with pytest.raises(ValueError):
        validate_relative_posix_path(bad)


# --------------------------------------------------------------------------- #
# Source identity (H1)
# --------------------------------------------------------------------------- #
def test_identical_bytes_at_different_paths_get_different_ids() -> None:
    first = record("train/a.jpg")
    second = record("train/copy_of_a.jpg")
    assert first.sha256 == second.sha256
    assert first.record_id != second.record_id
    assert first != second


def test_same_path_and_bytes_in_different_datasets_get_different_ids() -> None:
    assert record(dataset="one").record_id != record(dataset="two").record_id


def test_same_path_with_different_bytes_gets_different_ids() -> None:
    assert record(sha=A).record_id != record(sha=B).record_id


def test_source_record_id_is_deterministic_full_hex_and_matches_documented_encoding() -> None:
    value = source_record_id("ds", "train/a.jpg", A)
    assert value == source_record_id("ds", "train/a.jpg", A)
    assert re.fullmatch(r"[0-9a-f]{64}", value)
    payload = json.dumps(
        ["helmet_data.source_record.v1", "ds", "train/a.jpg", A], ensure_ascii=True, separators=(",", ":")
    )
    assert value == hashlib.sha256(payload.encode("ascii")).hexdigest()


def test_source_record_id_encoding_is_unambiguous() -> None:
    # Shifting characters between fields must never produce the same id.
    assert source_record_id("ab", "c/d", A) != source_record_id("a", "bc/d", A)
    assert source_record_id("a", "b/c", A) != source_record_id("a", "b", A)


@pytest.mark.parametrize(
    "args",
    [("Bad", "a.jpg", A), ("ds", "../a.jpg", A), ("ds", "a.jpg", "xyz"), ("ds", "", A), ("", "a.jpg", A)],
)
def test_source_record_id_validates_inputs(args) -> None:
    with pytest.raises(ValueError):
        source_record_id(*args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size_bytes": -1},
        {"size_bytes": True},
        {"size_bytes": "10"},
        {"width": 0},
        {"height": -3},
        {"width": 1.5},
        {"dhash": -1},
        {"dhash": 2**64},
        {"exif_orientation": 0},
        {"exif_orientation": 9},
        {"pixel_hash": "nope"},
        {"image_format": ""},
        {"sha256": "short"},
        {"dataset": "Bad Name"},
        {"relative_path": "../escape.jpg"},
    ],
)
def test_source_record_validates_fields(kwargs) -> None:
    base = {"dataset": "ds", "relative_path": "a.jpg", "sha256": A, "size_bytes": 10}
    base.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        SourceRecord(**base)


def test_source_record_accepts_full_valid_data_and_is_hashable() -> None:
    full = SourceRecord(
        "ds", "a.jpg", A, 10, width=640, height=480, pixel_hash=B, dhash=2**64 - 1,
        exif_orientation=1, image_format="JPEG",
    )
    assert hash(full) == hash(dataclasses.replace(full))
    assert len({full, dataclasses.replace(full)}) == 1


# --------------------------------------------------------------------------- #
# PixelBox (L7)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf"), "1", None, True, [1], 10**400]
)
def test_pixel_box_rejects_non_finite_or_non_numeric(bad) -> None:
    for position in range(4):
        values = [0, 0, 10, 10]
        values[position] = bad
        with pytest.raises(InvalidBoxError):
            PixelBox(*values)


def test_pixel_box_normalises_ints_and_negative_zero() -> None:
    box = PixelBox(1, 2, 3, 4)
    assert all(isinstance(v, float) for v in dataclasses.astuple(box))
    assert PixelBox(-0.0, 0, 1, 1) == PixelBox(0, 0, 1, 1)
    assert hash(PixelBox(-0.0, 0, 1, 1)) == hash(PixelBox(0, 0, 1, 1))
    assert str(PixelBox(-0.0, 0, 1, 1).left) == "0.0"


def test_pixel_box_equality_and_hash_are_deterministic() -> None:
    assert PixelBox(1.0, 2.0, 3.0, 4.0) == PixelBox(1, 2, 3, 4)
    assert hash(PixelBox(1.0, 2.0, 3.0, 4.0)) == hash(PixelBox(1, 2, 3, 4))
    assert len({PixelBox(1, 2, 3, 4), PixelBox(1.0, 2.0, 3.0, 4.0)}) == 1


def test_pixel_box_is_frozen() -> None:
    box = PixelBox(1, 2, 3, 4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        box.left = 5.0  # type: ignore[misc]


def test_pixel_box_allows_finite_degenerate_geometry_for_later_validators() -> None:
    zero_width = PixelBox(5, 5, 5, 9)
    assert zero_width.right - zero_width.left == 0
    assert PixelBox(10, 10, 5, 5)  # inverted but finite: validators reject it later


# --------------------------------------------------------------------------- #
# Annotations, output records, groups
# --------------------------------------------------------------------------- #
def annotation(**overrides) -> SourceAnnotation:
    values = dict(
        annotation_id="ann-1", record_id=A, source_class="helmet", box=PixelBox(0, 0, 10, 10),
        status=AnnotationStatus.MAPPED, target_class=TargetClass.WITH_HELMET,
    )
    values.update(overrides)
    return SourceAnnotation(**values)


def test_source_annotation_is_hashable_and_validated() -> None:
    first = annotation()
    assert hash(first) == hash(annotation())
    for bad in (
        {"annotation_id": ""},
        {"record_id": "short"},
        {"source_class": ""},
        {"box": (0, 0, 1, 1)},
        {"status": "mapped"},
        {"reason": "iscrowd"},
        {"target_class": 0},
    ):
        with pytest.raises((ValueError, TypeError)):
            annotation(**bad)


def test_output_record_requires_tuple_of_valid_unique_nonempty_ids() -> None:
    assert OutputRecord("x.jpg", (A, B), Split.TRAIN)
    with pytest.raises(ValueError):
        OutputRecord("x.jpg", ())
    with pytest.raises(TypeError):
        OutputRecord("x.jpg", [A])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OutputRecord("x.jpg", ("not-a-hash",))
    with pytest.raises(ValueError):
        OutputRecord("x.jpg", (123,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OutputRecord("x.jpg", (A, A))


@pytest.mark.parametrize("name", ["", "a/b.jpg", "a\\b.jpg", ".", ".."])
def test_output_record_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError):
        OutputRecord(name, (A,))


def test_output_record_is_hashable_immutable_and_type_checks_split() -> None:
    first = OutputRecord("x.jpg", (A, B), Split.TRAIN)
    assert hash(first) == hash(OutputRecord("x.jpg", (A, B), Split.TRAIN))
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.split = Split.TEST  # type: ignore[misc]
    with pytest.raises(TypeError):
        OutputRecord("x.jpg", (A,), split="train")  # type: ignore[arg-type]


def test_group_validates_element_types_and_is_hashable() -> None:
    group = Group("g1", (A, B), ("dhash",))
    assert hash(group) == hash(Group("g1", (A, B), ("dhash",)))
    with pytest.raises(TypeError):
        Group("g2", [A])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Group("g3", (A,), ["x"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Group("g4", (A,), (5,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Group("g5", ("zzz",))
    with pytest.raises(ValueError):
        Group("", (A,))
    with pytest.raises(ValueError):
        Group("g6", ())


# --------------------------------------------------------------------------- #
# Strict whole-string validation (a '$' anchor would accept a trailing newline)
# --------------------------------------------------------------------------- #
UNEXPECTED_EXTRAS = ["\n", "\r", "\r\n", "\n\n", " ", "\t", "\x00", "\x0b", "\x0c", " ", "\u0085", " ", "​"]


@pytest.mark.parametrize("extra", UNEXPECTED_EXTRAS)
def test_dataset_name_rejects_leading_or_trailing_extra_characters(extra: str) -> None:
    assert validate_dataset_name("ds") == "ds"
    for bad in ("ds" + extra, extra + "ds", "d" + extra + "s"):
        with pytest.raises(ValueError):
            validate_dataset_name(bad)


@pytest.mark.parametrize("extra", UNEXPECTED_EXTRAS)
def test_sha256_rejects_leading_or_trailing_extra_characters(extra: str) -> None:
    assert validate_sha256(A) == A
    for bad in (A + extra, extra + A, A[:-1] + extra):
        with pytest.raises(ValueError):
            validate_sha256(bad)


def test_sha256_rejects_wrong_lengths_and_look_alike_characters() -> None:
    for bad in ("a" * 65, "a" * 63, "٣" * 64, "ａ" * 64, "a" * 63 + "á", "A" * 64):
        with pytest.raises(ValueError):
            validate_sha256(bad)
    for bad in ("ds٣", "dｓ", "dsé"):
        with pytest.raises(ValueError):
            validate_dataset_name(bad)


@pytest.mark.parametrize("extra", ["\n", "\r", "\r\n", " ", "\t", "\x00"])
def test_every_consumer_of_the_validators_rejects_noncanonical_values(extra: str) -> None:
    with pytest.raises(ValueError):
        source_record_id("ds", "a.jpg", A + extra)
    with pytest.raises(ValueError):
        source_record_id("ds" + extra, "a.jpg", A)
    with pytest.raises(ValueError):
        SourceRecord("ds", "a.jpg", A + extra, 1)
    with pytest.raises(ValueError):
        SourceRecord("ds" + extra, "a.jpg", A, 1)
    with pytest.raises(ValueError):
        SourceRecord("ds", "a.jpg", A, 1, pixel_hash=B + extra)
    with pytest.raises(ValueError):
        OutputRecord("x.jpg", (A + extra,))
    with pytest.raises(ValueError):
        Group("g", (A + extra,))
    with pytest.raises(ValueError):
        annotation_with_record_id(A + extra)


def annotation_with_record_id(record_id: str) -> SourceAnnotation:
    return SourceAnnotation(
        annotation_id="a1", record_id=record_id, source_class="helmet", box=PixelBox(0, 0, 1, 1),
        status=AnnotationStatus.MAPPED,
    )
