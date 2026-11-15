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
    FINDING_STATUSES,
    OUTPUT_NAME_HASH_HEX_CHARS,
    SOURCE_ID_NAMESPACES,
    validate_source_id_namespace,
    MAX_OUTPUT_FILENAME_LENGTH,
    MAX_RAW_GEOMETRY_CHARS,
    OUTPUT_IMAGE_EXTENSIONS,
    PENDING_MAPPING_STATUSES,
    TRAINING_CANDIDATE_STATUSES,
    bounded_raw_text,
    source_annotation_id,
    validate_output_filename,
    validate_source_class,
    validate_source_locator,
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
        # added in Milestone 1.2.0
        "malformed_annotation",
        "unsupported_annotation",
        "duplicate_source_id",
        "orphan_annotation",
        "ambiguous_pairing",
        "image_size_mismatch",
        "unsafe_reference",
        "limit_exceeded",
        "duplicate_box",
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
    assert [s.value for s in AnnotationStatus] == ["mapped", "excluded", "rejected", "review", "unmapped"]
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
    assert len(ReasonCode) == 35 == len(APPROVED_REASON_CODES)


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
        record_id=A, source_file="labels/a.txt", source_locator="line:1", source_class="helmet",
        status=AnnotationStatus.MAPPED, box=PixelBox(0, 0, 10, 10), target_class=TargetClass.WITH_HELMET,
    )
    values.update(overrides)
    return SourceAnnotation(**values)


def test_source_annotation_is_hashable_and_validated() -> None:
    first = annotation()
    assert hash(first) == hash(annotation())
    for bad in (
        {"record_id": "short"},
        {"source_file": "../x.txt"},
        {"source_locator": "line:01"},
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
UNEXPECTED_EXTRAS = ["\n", "\r", "\r\n", "\n\n", " ", "\t", "\x00", "\x0b", "\x0c", "\u2028", "\u0085", "\u00a0", "\u200b"]


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
    for bad in ("a" * 65, "a" * 63, "\u0663" * 64, "\uff41" * 64, "a" * 63 + "\u00e1", "A" * 64):
        with pytest.raises(ValueError):
            validate_sha256(bad)
    for bad in ("ds\u0663", "d\uff53", "ds\u00e9"):
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
        record_id=record_id, source_file="labels/a.txt", source_locator="line:1",
        source_class="helmet", status=AnnotationStatus.UNMAPPED, box=PixelBox(0, 0, 1, 1),
    )


# =========================================================================== #
# Milestone 1.2.0
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Dataset names: reserved Windows device names
# --------------------------------------------------------------------------- #
RESERVED = ["con", "prn", "aux", "nul", *[f"com{i}" for i in range(1, 10)], *[f"lpt{i}" for i in range(1, 10)]]


@pytest.mark.parametrize("name", RESERVED)
def test_reserved_device_names_are_refused_as_dataset_names(name: str) -> None:
    with pytest.raises(ValueError, match="reserved Windows device name"):
        validate_dataset_name(name)


@pytest.mark.parametrize("name", ["NUL", "Con", "COM1", "Lpt9", "AUX", "nUl"])
def test_reserved_device_names_are_refused_in_any_case(name: str) -> None:
    with pytest.raises(ValueError, match="reserved Windows device name"):
        validate_dataset_name(name)


@pytest.mark.parametrize("name", ["nul.txt", "CON.jpg", "com1.x", "aux.", "nul ", "con  ", "prn.tar.gz", "COM9.TXT"])
def test_reserved_device_names_are_refused_with_extensions_or_trailing_spaces(name: str) -> None:
    with pytest.raises(ValueError):
        validate_dataset_name(name)


@pytest.mark.parametrize("name", ["console", "nullable", "com10", "com0", "lpt0", "aux2", "con-1", "nul_", "xcon", "com1a", "my-com1"])
def test_names_that_merely_resemble_reserved_names_are_still_valid(name: str) -> None:
    assert validate_dataset_name(name) == name


def test_existing_strict_dataset_name_rules_are_preserved() -> None:
    for good in ("a", "ds", "helmet-set_2", "0abc", "a" * 64):
        assert validate_dataset_name(good) == good
    for bad in ("", "Upper", "a b", "a/b", "a\\b", "-x", "_x", "a" * 65, "caf\u00e9", "a\nb", "ds\n", None, 5):
        with pytest.raises(ValueError):
            validate_dataset_name(bad)


def test_reserved_dataset_names_cannot_reach_source_records_or_ids() -> None:
    with pytest.raises(ValueError):
        SourceRecord("nul", "a.jpg", A, 1)
    with pytest.raises(ValueError):
        source_record_id("con", "a.jpg", A)


# --------------------------------------------------------------------------- #
# Output file names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name",
    ["a.jpg", "A.JPG", "ds__0123456789abcdef.png", "x-1_2.webp", "img.v2.jpeg", "0.tif", "a.TIFF", "a.Bmp", "a" * 124 + ".jpg"],
)
def test_valid_output_filenames(name: str) -> None:
    assert validate_output_filename(name) == name
    assert OutputRecord(name, (A,)).output_name == name


@pytest.mark.parametrize(
    "name",
    [
        "", ".", "..", "...", ".jpg", "a", "a.", "a ", " a.jpg", "a.jpg ", "a.jpg.", "a..jpg",
        "a/b.jpg", "a\\b.jpg", "/a.jpg", "\\a.jpg", "../a.jpg", "..\\a.jpg", "C:a.jpg", "a:b.jpg", "a.jpg:stream",
        "a\nb.jpg", "a\r.jpg", "a\tb.jpg", "a\x00.jpg", "a\x1f.jpg", "a\x7f.jpg",
        'a"b.jpg', "a<b.jpg", "a>b.jpg", "a|b.jpg", "a?b.jpg", "a*b.jpg", "a b.jpg", "a,b.jpg", "a+b.jpg", "a;b.jpg",
        "caf\u00e9.jpg", "\u00e1.jpg", "\uff41.jpg", "a\u202eb.jpg", "a\u200b.jpg", "a\u2028b.jpg",
        "nul.jpg", "NUL.png", "con.jpg", "Con.JPG", "prn.png", "aux.webp", "com1.jpg", "COM9.png", "lpt1.bmp", "lpt9.tif",
        "a.exe", "a.txt", "a.gif", "a.jpg.exe", "a.svg", "a.jpgx", "a.tar", "a.JPGG",
        "a" * 125 + ".jpg", "a" * 200 + ".jpg",
        None, 5, b"a.jpg", ["a.jpg"],
    ],
)
def test_unsafe_or_unsuitable_output_filenames_are_rejected(name) -> None:
    with pytest.raises(ValueError):
        validate_output_filename(name)
    with pytest.raises((ValueError, TypeError)):
        OutputRecord(name, (A,))


def test_output_filename_length_limit_is_in_characters_of_the_whole_name() -> None:
    assert MAX_OUTPUT_FILENAME_LENGTH == 128
    assert len(validate_output_filename("a" * (128 - 4) + ".jpg")) == 128  # includes the extension
    with pytest.raises(ValueError, match="limit is 128"):
        validate_output_filename("a" * (129 - 4) + ".jpg")
    # ASCII only, so characters equal bytes
    assert len(("a" * 124 + ".jpg").encode("ascii")) == 128


def test_output_filename_extension_is_checked_case_insensitively_but_never_changed() -> None:
    for extension in OUTPUT_IMAGE_EXTENSIONS:
        assert validate_output_filename("a" + extension) == "a" + extension
        assert validate_output_filename("a" + extension.upper()) == "a" + extension.upper()
    assert OUTPUT_IMAGE_EXTENSIONS == (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")


def test_reserved_names_are_refused_even_inside_a_longer_looking_name() -> None:
    for name in ("nul.png", "NUL.PNG", "nul.tar.png", "com1.v2.jpg"):
        with pytest.raises(ValueError):
            validate_output_filename(name)
    for name in ("nul_1.png", "console.png", "com10.jpg"):
        assert validate_output_filename(name) == name


# --------------------------------------------------------------------------- #
# Annotation ids
# --------------------------------------------------------------------------- #
def test_annotation_id_is_documented_full_sha256_of_a_versioned_json_array() -> None:
    value = source_annotation_id(A, "labels/a.txt", "line:12")
    payload = json.dumps(
        ["helmet_data.source_annotation.v1", A, "labels/a.txt", "line:12"], ensure_ascii=True, separators=(",", ":")
    )
    assert value == hashlib.sha256(payload.encode("ascii")).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", value)
    assert annotation().annotation_id == source_annotation_id(A, "labels/a.txt", "line:1")


def test_identical_locators_in_different_files_get_different_ids() -> None:
    ids = {
        source_annotation_id(A, "labels/a.txt", "line:1"),
        source_annotation_id(A, "labels/b.txt", "line:1"),
        source_annotation_id(A, "other/labels/a.txt", "line:1"),
    }
    assert len(ids) == 3


def test_identical_locators_for_different_records_get_different_ids() -> None:
    assert source_annotation_id(A, "labels/a.txt", "line:1") != source_annotation_id(B, "labels/a.txt", "line:1")
    first = SourceRecord("ds", "images/a.jpg", A, 1)
    second = SourceRecord("ds", "images/copy.jpg", A, 1)  # identical bytes, different path
    one = annotation(record_id=first.record_id)
    two = annotation(record_id=second.record_id)
    assert one.annotation_id != two.annotation_id


def test_different_locators_in_one_file_get_different_ids() -> None:
    locators = ["line:1", "line:2", "line:10", "object:1", "index:1/id:1", "index:1/id:2", "index:2/id:1"]
    ids = {source_annotation_id(A, "labels/a.txt", loc) for loc in locators}
    assert len(ids) == len(locators)


def test_annotation_id_encoding_is_unambiguous() -> None:
    # moving characters between fields can never reproduce an id
    assert source_annotation_id(A, "a/b.txt", "line:1") != source_annotation_id(A, "a/b", "line:1")
    assert source_annotation_id(A, "a.txt", "line:1") != source_annotation_id(A, "a.txt", "line:1/id:1")
    assert source_annotation_id(A, "a/line", "line:1") != source_annotation_id(A, "a", "line:1")


def test_annotation_id_is_independent_of_class_geometry_status_and_flags() -> None:
    base = annotation().annotation_id
    assert annotation(source_class="other").annotation_id == base
    assert annotation(box=PixelBox(1, 1, 5, 5)).annotation_id == base
    assert annotation(flags=("difficult",)).annotation_id == base
    unmapped = annotation(status=AnnotationStatus.UNMAPPED, target_class=None)
    assert unmapped.annotation_id == base


def test_annotation_id_is_stable_across_instances_and_pythonhashseed() -> None:
    assert annotation().annotation_id == annotation().annotation_id
    # Python's hash() is not used: a pinned value would break if it were (salted per process)
    assert source_annotation_id(A, "labels/a.txt", "line:1") == (
        hashlib.sha256(
            b'["helmet_data.source_annotation.v1","' + A.encode() + b'","labels/a.txt","line:1"]'
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "args",
    [
        ("short", "a.txt", "line:1"),
        (A, "/abs/a.txt", "line:1"),
        (A, "C:/abs/a.txt", "line:1"),
        (A, "..\\a.txt", "line:1"),
        (A, "../a.txt", "line:1"),
        (A, "a\\b.txt", "line:1"),
        (A, "", "line:1"),
        (A, "a.txt", ""),
        (A, "a.txt", "line"),
        (A, "a.txt", "line:"),
        (A, "a.txt", "line:01"),
        (A, "a.txt", "line:-1"),
        (A, "a.txt", "line:1.5"),
        (A, "a.txt", "Line:1"),
        (A, "a.txt", "line:1 "),
        (A, "a.txt", "line:1\n"),
        (A, "a.txt", "line:1/"),
        (A, "a.txt", "/line:1"),
        (A, "a.txt", "line:1//id:2"),
        (A, "a.txt", "line:1234567890123456789"),
        (A, "a.txt", "line:" + "9" * 40),
        (A + "\n", "a.txt", "line:1"),
        (None, "a.txt", "line:1"),
        (A, 5, "line:1"),
        (A, "a.txt", 1),
    ],
)
def test_annotation_id_rejects_unsafe_or_noncanonical_inputs(args) -> None:
    with pytest.raises((ValueError, TypeError)):
        source_annotation_id(*args)


@pytest.mark.parametrize("locator", ["line:1", "line:0", "object:0", "index:0/id:17", "index:42/id:9", "line:" + "9" * 18])
def test_valid_source_locators(locator: str) -> None:
    assert validate_source_locator(locator) == locator


def test_absolute_paths_timestamps_and_random_values_never_enter_the_id() -> None:
    # the only inputs are three validated strings; absolute-looking inputs are refused
    for bad_path in ("D:/Projects/x/labels/a.txt", "/home/u/a.txt", "\\\\server\\share\\a.txt", "C:\\a.txt"):
        with pytest.raises(ValueError):
            source_annotation_id(A, bad_path, "line:1")
    assert len({annotation().annotation_id for _ in range(5)}) == 1


# --------------------------------------------------------------------------- #
# SourceAnnotation: valid geometry versus bounded raw geometry
# --------------------------------------------------------------------------- #
def rejected(**overrides) -> SourceAnnotation:
    values = dict(
        record_id=A, source_file="labels/a.txt", source_locator="line:3", source_class="helmet",
        status=AnnotationStatus.REJECTED, box=None, raw_geometry="0.1 nan 0.2 0.3",
        reason=ReasonCode.INVALID_BOX,
    )
    values.update(overrides)
    return SourceAnnotation(**values)


def test_invalid_geometry_is_stored_as_raw_text_never_as_a_box() -> None:
    item = rejected()
    assert item.box is None
    assert item.raw_geometry == "0.1 nan 0.2 0.3"
    assert item.status is AnnotationStatus.REJECTED
    assert item.annotation_id == source_annotation_id(A, "labels/a.txt", "line:3")
    assert hash(item) == hash(rejected())


def test_a_nan_box_cannot_be_smuggled_in_as_a_valid_box() -> None:
    with pytest.raises(InvalidBoxError):
        PixelBox(float("nan"), 0, 1, 1)
    with pytest.raises(TypeError):
        annotation(box=(float("nan"), 0, 1, 1))
    with pytest.raises(TypeError):
        annotation(box="0 0 1 1")


def test_box_and_raw_geometry_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="only for annotations without a valid box"):
        annotation(raw_geometry="0 0 1 1")
    with pytest.raises(ValueError, match="needs non-empty raw_geometry"):
        rejected(raw_geometry=None)
    with pytest.raises(ValueError, match="needs non-empty raw_geometry"):
        rejected(raw_geometry="")


@pytest.mark.parametrize("status", [AnnotationStatus.MAPPED, AnnotationStatus.UNMAPPED, AnnotationStatus.REVIEW])
def test_only_rejected_or_excluded_annotations_may_lack_a_box(status) -> None:
    with pytest.raises(ValueError):
        rejected(status=status, reason=ReasonCode.INVALID_BOX if status is AnnotationStatus.REVIEW else None)


def test_excluded_annotations_may_carry_a_box_or_raw_geometry() -> None:
    with_box = SourceAnnotation(
        A, "ann.json", "index:3/id:17", "person", AnnotationStatus.EXCLUDED, box=PixelBox(0, 0, 5, 5),
        reason=ReasonCode.ISCROWD, flags=("iscrowd",),
    )
    without = SourceAnnotation(
        A, "ann.json", "index:4/id:18", "person", AnnotationStatus.EXCLUDED, raw_geometry="[1, 2, 3]",
        reason=ReasonCode.ISCROWD, flags=("iscrowd",),
    )
    assert with_box.box is not None and without.box is None


def test_raw_geometry_must_be_bounded_printable_ascii() -> None:
    assert rejected(raw_geometry="x" * MAX_RAW_GEOMETRY_CHARS).raw_geometry == "x" * MAX_RAW_GEOMETRY_CHARS
    for bad in ("x" * (MAX_RAW_GEOMETRY_CHARS + 1), "a\nb", "a\x00b", "caf\u00e9", "a\x7f", "a\tb", "\u202e"):
        with pytest.raises(ValueError):
            rejected(raw_geometry=bad)


def test_bounded_raw_text_escapes_and_truncates_deterministically() -> None:
    assert bounded_raw_text("0.1 0.2") == "0.1 0.2"
    assert bounded_raw_text("a\nb\x00c") == "a\\nb\\x00c"
    assert bounded_raw_text("caf\u00e9") == "caf\\xe9"
    assert bounded_raw_text("\u202e") == "\\u202e"
    assert bounded_raw_text("back\\slash") == "back\\\\slash"
    assert bounded_raw_text("\ud800") == "\\ud800"  # lone surrogate cannot crash
    long = bounded_raw_text("z" * 10_000)
    assert len(long) == MAX_RAW_GEOMETRY_CHARS and long.endswith("...[truncated]")
    assert bounded_raw_text("z" * 10_000) == long
    assert len(bounded_raw_text("\u00e9" * 1000, 40)) == 40
    for hostile in ("\x00" * 5000, "\u202e" * 5000, "\n" * 5000, "a" * 100_000):
        assert len(bounded_raw_text(hostile)) <= MAX_RAW_GEOMETRY_CHARS
        # the result is always acceptable to the model
        assert rejected(raw_geometry=bounded_raw_text(hostile))
    with pytest.raises(TypeError):
        bounded_raw_text(b"bytes")  # type: ignore[arg-type]
    for bad_limit in (0, 5, 14, -1, True, 2.5):
        with pytest.raises(ValueError):
            bounded_raw_text("x", bad_limit)  # type: ignore[arg-type]


def test_source_file_locator_and_flags_are_preserved() -> None:
    item = annotation(source_file="annotations/instances.json", source_locator="index:7/id:99", flags=("truncated", "difficult"))
    assert item.source_file == "annotations/instances.json"
    assert item.source_locator == "index:7/id:99"
    assert item.flags == ("difficult", "truncated")  # canonical order
    assert annotation(flags=("b", "a")) == annotation(flags=("a", "b"))


@pytest.mark.parametrize(
    "flags", [["difficult"], ("Difficult",), ("1x",), ("a b",), ("",), ("difficult", "difficult"), ("x" * 33,), tuple(f"f{i}" for i in range(17)), (5,)]
)
def test_invalid_flags_are_rejected(flags) -> None:
    with pytest.raises((ValueError, TypeError)):
        annotation(flags=flags)


@pytest.mark.parametrize(
    "name", ["helmet", "Helmet", " helmet ", "without helmet", "id:3", "\u5934\u76d4", "class-1", "a" * 256, "x\u00e9"]
)
def test_source_classes_are_kept_verbatim(name: str) -> None:
    assert validate_source_class(name) == name
    assert annotation(source_class=name).source_class == name


@pytest.mark.parametrize(
    "name", ["", " ", "\t", "\n", "a\nb", "a\x00", "a\u202eb", "a\u200bb", "a\u2028b", "a\u2029b", "\ud800", "a" * 257, None, 5, b"x"]
)
def test_unsafe_source_classes_are_rejected(name) -> None:
    with pytest.raises(ValueError):
        validate_source_class(name)


# --------------------------------------------------------------------------- #
# Status rules and UNMAPPED
# --------------------------------------------------------------------------- #
def unmapped(**overrides) -> SourceAnnotation:
    values = dict(status=AnnotationStatus.UNMAPPED, target_class=None)
    values.update(overrides)
    return annotation(**values)


def test_unmapped_is_a_parsed_valid_annotation_with_no_reason_and_no_target() -> None:
    item = unmapped()
    assert item.status is AnnotationStatus.UNMAPPED
    assert item.box is not None and item.reason is None and item.target_class is None
    with pytest.raises(ValueError, match="not a finding"):
        unmapped(reason=ReasonCode.UNKNOWN_SEMANTICS)
    with pytest.raises(ValueError, match="only a MAPPED"):
        unmapped(target_class=TargetClass.WITH_HELMET)


def test_unmapped_is_not_accepted_for_training_and_not_a_finding() -> None:
    assert AnnotationStatus.UNMAPPED not in TRAINING_CANDIDATE_STATUSES
    assert AnnotationStatus.UNMAPPED not in FINDING_STATUSES
    assert AnnotationStatus.UNMAPPED in PENDING_MAPPING_STATUSES
    assert TRAINING_CANDIDATE_STATUSES == {AnnotationStatus.MAPPED}
    assert FINDING_STATUSES == {AnnotationStatus.REVIEW, AnnotationStatus.REJECTED, AnnotationStatus.EXCLUDED}
    assert PENDING_MAPPING_STATUSES == {AnnotationStatus.UNMAPPED}
    # the three groupings are disjoint
    assert not (FINDING_STATUSES & PENDING_MAPPING_STATUSES)
    assert not (FINDING_STATUSES & TRAINING_CANDIDATE_STATUSES)
    assert not (PENDING_MAPPING_STATUSES & TRAINING_CANDIDATE_STATUSES)
    assert set(AnnotationStatus) == FINDING_STATUSES | PENDING_MAPPING_STATUSES | TRAINING_CANDIDATE_STATUSES


def test_unmapped_value_serialises_stably() -> None:
    assert AnnotationStatus.UNMAPPED.value == "unmapped"
    assert str(AnnotationStatus.UNMAPPED) == "unmapped" == f"{AnnotationStatus.UNMAPPED}"


def test_mapped_requires_a_target_and_has_no_reason() -> None:
    assert annotation().target_class is TargetClass.WITH_HELMET
    with pytest.raises(ValueError, match="needs a target_class"):
        annotation(target_class=None)
    with pytest.raises(ValueError, match="has no reason"):
        annotation(reason=ReasonCode.SMALL_PROJECTED_BOX)


@pytest.mark.parametrize("status", sorted(FINDING_STATUSES, key=lambda s: s.value))
def test_finding_statuses_require_a_reason_and_forbid_a_target(status) -> None:
    ok = annotation(status=status, target_class=None, reason=ReasonCode.SMALL_PROJECTED_BOX)
    assert ok.reason is ReasonCode.SMALL_PROJECTED_BOX
    with pytest.raises(ValueError, match="needs a reason"):
        annotation(status=status, target_class=None)
    with pytest.raises(ValueError, match="only a MAPPED"):
        annotation(status=status, target_class=TargetClass.WITH_HELMET, reason=ReasonCode.SMALL_PROJECTED_BOX)


# --------------------------------------------------------------------------- #
# Whole-model compatibility checks
# --------------------------------------------------------------------------- #
def test_every_status_can_be_constructed_in_a_valid_shape() -> None:
    shapes = {
        AnnotationStatus.MAPPED: dict(target_class=TargetClass.WITHOUT_HELMET),
        AnnotationStatus.UNMAPPED: dict(target_class=None),
        AnnotationStatus.REVIEW: dict(target_class=None, reason=ReasonCode.OUT_OF_BOUNDS_MINOR),
        AnnotationStatus.REJECTED: dict(target_class=None, reason=ReasonCode.OUT_OF_BOUNDS_MAJOR),
        AnnotationStatus.EXCLUDED: dict(target_class=None, reason=ReasonCode.ISCROWD),
    }
    assert set(shapes) == set(AnnotationStatus)
    for status, extra in shapes.items():
        assert annotation(status=status, **extra).status is status


def test_the_approved_35_code_set_is_the_original_26_plus_nine_new_codes() -> None:
    new_codes = {
        "malformed_annotation", "unsupported_annotation", "duplicate_source_id", "orphan_annotation",
        "ambiguous_pairing", "image_size_mismatch", "unsafe_reference", "limit_exceeded", "duplicate_box",
    }
    assert len(new_codes) == 9 and new_codes <= APPROVED_REASON_CODES
    assert len(APPROVED_REASON_CODES - new_codes) == 26
    assert {r.value for r in ReasonCode} == APPROVED_REASON_CODES


@pytest.mark.parametrize("name", ["a/b.jpg", "a\\b.jpg", "/a.jpg", "\\a.jpg", "../a.jpg", "..\\a.jpg", "dir/"])
def test_output_filename_separators_get_a_specific_error(name: str) -> None:
    with pytest.raises(ValueError, match="path separators"):
        validate_output_filename(name)

# =========================================================================== #
# Milestone 1.2.0 finalization: the nine new reason codes, joiners, hash suffix
# =========================================================================== #
NEW_CODES = {
    ReasonCode.MALFORMED_ANNOTATION: "REJECT",
    ReasonCode.UNSUPPORTED_ANNOTATION: "REJECT",
    ReasonCode.DUPLICATE_SOURCE_ID: "REVIEW",
    ReasonCode.ORPHAN_ANNOTATION: "REJECT",
    ReasonCode.AMBIGUOUS_PAIRING: "REVIEW",
    ReasonCode.IMAGE_SIZE_MISMATCH: "REVIEW",
    ReasonCode.UNSAFE_REFERENCE: "REJECT",
    ReasonCode.LIMIT_EXCEEDED: "REJECT",
    ReasonCode.DUPLICATE_BOX: "REVIEW",
}


def doc(code: ReasonCode) -> str:
    return REASON_CODE_DOCS[code]


def test_there_are_exactly_nine_new_codes_each_with_a_documented_default_decision() -> None:
    assert len(NEW_CODES) == 9
    for code, decision in NEW_CODES.items():
        assert doc(code).startswith(decision), (code, doc(code)[:30])
        assert len(doc(code)) > 120  # a real definition, not a label


def test_every_reason_code_has_a_unique_definition() -> None:
    texts = [REASON_CODE_DOCS[code] for code in ReasonCode]
    assert len(set(texts)) == len(texts)
    assert set(REASON_CODE_DOCS) == set(ReasonCode)
    for code in ReasonCode:
        assert doc(code).split()[0].rstrip(".,(") in {
            "REVIEW", "REJECT", "BLOCK", "EXCLUDE", "EXCLUDED", "Outcome"
        }, code


def test_each_new_code_states_what_distinguishes_it_from_its_neighbours() -> None:
    assert "invalid_box" in doc(ReasonCode.MALFORMED_ANNOTATION)
    assert "unsupported_annotation" in doc(ReasonCode.MALFORMED_ANNOTATION)
    assert "never converted" in doc(ReasonCode.UNSUPPORTED_ANNOTATION)
    assert "duplicate_box" in doc(ReasonCode.DUPLICATE_SOURCE_ID)
    assert "unlabeled" in doc(ReasonCode.ORPHAN_ANNOTATION)
    assert "No pairing is ever guessed" in doc(ReasonCode.AMBIGUOUS_PAIRING)
    assert "never rescaled" in doc(ReasonCode.IMAGE_SIZE_MISMATCH)
    assert "never used to open any file" in doc(ReasonCode.UNSAFE_REFERENCE)
    assert "nothing is read partially" in doc(ReasonCode.LIMIT_EXCEEDED)
    assert "class_conflict" in doc(ReasonCode.DUPLICATE_BOX)


def test_malformed_annotation_distinguishes_file_level_from_annotation_level() -> None:
    text = doc(ReasonCode.MALFORMED_ANNOTATION)
    assert "ANNOTATION-level" in text and "FILE-level" in text
    assert "exactly one of" in text and "state its scope" in text
    for file_level_cause in ("undecodable", "invalid XML/JSON", "forbidden DTD", "duplicate JSON key", "NaN constant"):
        assert file_level_cause in text
    for annotation_level_cause in ("wrong column count", "missing or wrongly typed required field", "no bounding-box"):
        assert annotation_level_cause in text


def test_duplicate_source_id_names_its_identifier_namespace() -> None:
    text = doc(ReasonCode.DUPLICATE_SOURCE_ID)
    assert "ONE identifier namespace" in text and "must name the namespace" in text
    for namespace in SOURCE_ID_NAMESPACES:
        assert namespace in text
    assert SOURCE_ID_NAMESPACES == (
        "coco.annotation_id", "coco.category_id", "coco.category_name", "coco.image_id", "yolo.class_id", "yolo.class_name"
    )
    assert "different namespaces never conflict" in text


def test_source_id_namespace_validation() -> None:
    for namespace in SOURCE_ID_NAMESPACES:
        assert validate_source_id_namespace(namespace) == namespace
    for bad in ("", "coco", "COCO.image_id", "coco.image_id ", "coco.image_id\n", "voc.object_name", "image_id", None, 5):
        with pytest.raises(ValueError):
            validate_source_id_namespace(bad)


def test_duplicate_box_is_review_only_and_never_removes_annotations() -> None:
    text = doc(ReasonCode.DUPLICATE_BOX)
    assert text.startswith("REVIEW only")
    assert "kept" in text and "none is silently removed, merged or dropped" in text
    assert "REJECT" not in text.split(".")[0]


def test_unknown_semantics_is_a_class_level_block_condition() -> None:
    text = doc(ReasonCode.UNKNOWN_SEMANTICS)
    assert text.startswith("BLOCK (class-level, never per annotation)")
    assert "REVIEW/BLOCK" not in text
    assert "stay UNMAPPED" in text and "never inferred from the class name" in text


def test_corrupt_image_in_1_2_concerns_header_inspection_not_full_decoding() -> None:
    text = doc(ReasonCode.CORRUPT_IMAGE)
    assert "header inspection only" in text
    assert "does NOT mean the pixel data was fully decoded" in text
    assert "later milestone (1.4)" in text
    assert "fails full decoding" not in text


def test_invalid_box_covers_non_numeric_non_finite_and_wrong_arity_geometry() -> None:
    text = doc(ReasonCode.INVALID_BOX)
    for phrase in ("non-numeric", "non-finite", "NaN or infinity", "wrong arity", "exactly 4 numbers", "degenerate"):
        assert phrase in text
    assert "raw_geometry" in text and "never as a PixelBox" in text
    assert "malformed_annotation" in text and "out_of_bounds_" in text


def test_new_codes_are_usable_on_source_annotations_with_the_right_status() -> None:
    for code, decision in NEW_CODES.items():
        status = AnnotationStatus.REJECTED if decision == "REJECT" else AnnotationStatus.REVIEW
        item = annotation(status=status, target_class=None, reason=code)
        assert item.reason is code
        raw = rejected(reason=code)  # a box-less annotation may carry any reason
        assert raw.reason is code


def test_unmapped_finding_and_training_status_sets_are_still_disjoint_and_complete() -> None:
    assert not (FINDING_STATUSES & PENDING_MAPPING_STATUSES)
    assert not (FINDING_STATUSES & TRAINING_CANDIDATE_STATUSES)
    assert not (PENDING_MAPPING_STATUSES & TRAINING_CANDIDATE_STATUSES)
    assert set(AnnotationStatus) == FINDING_STATUSES | PENDING_MAPPING_STATUSES | TRAINING_CANDIDATE_STATUSES
    assert AnnotationStatus.UNMAPPED in PENDING_MAPPING_STATUSES
    assert AnnotationStatus.UNMAPPED not in FINDING_STATUSES | TRAINING_CANDIDATE_STATUSES
    with pytest.raises(ValueError):
        unmapped(reason=ReasonCode.UNKNOWN_SEMANTICS)  # still not a finding, even with a class-level code


# --------------------------------------------------------------------------- #
# Output-name hash suffix (recorded only; the writer is not implemented)
# --------------------------------------------------------------------------- #
def test_the_approved_output_name_hash_suffix_is_sixteen_hex_characters() -> None:
    assert OUTPUT_NAME_HASH_HEX_CHARS == 16
    digest = hashlib.sha256(b"image bytes").hexdigest()
    suffix = digest[:OUTPUT_NAME_HASH_HEX_CHARS]
    assert len(suffix) == 16 and re.fullmatch(r"[0-9a-f]{16}", suffix)


def test_the_longest_planned_output_name_is_valid_and_within_the_length_limit() -> None:
    worst = "a" * 64 + "__" + "f" * OUTPUT_NAME_HASH_HEX_CHARS + ".tiff"
    assert len(worst) == 64 + 2 + 16 + 5 == 87 <= MAX_OUTPUT_FILENAME_LENGTH
    assert validate_output_filename(worst) == worst
    assert OutputRecord(worst, (A,)).output_name == worst
    for extension in OUTPUT_IMAGE_EXTENSIONS:
        name = "d" * 64 + "__" + "0" * OUTPUT_NAME_HASH_HEX_CHARS + extension.upper()
        assert validate_output_filename(name) == name


def test_no_name_generation_function_exists_yet() -> None:
    import helmet_data.models as models

    assert not any(name.startswith(("make_output", "generate_output", "build_output")) for name in dir(models))


# --------------------------------------------------------------------------- #
# Class names: U+200C / U+200D allowed only in safe positions
# --------------------------------------------------------------------------- #
ZWNJ, ZWJ = "\u200c", "\u200d"
PERSIAN = "\u0645\u06cc" + ZWNJ + "\u062e\u0648\u0627\u0647\u0645"  # ZWNJ between letters
INDIC = "\u0915\u094d" + ZWJ + "\u0937"  # ZWJ between a virama (mark) and a letter
EMOJI = "\U0001F468" + ZWJ + "\U0001F469"  # ZWJ between two symbols
EMOJI_FAMILY = "\U0001F468" + ZWJ + "\U0001F469" + ZWJ + "\U0001F467"


@pytest.mark.parametrize("name", [PERSIAN, INDIC, EMOJI, EMOJI_FAMILY, "a" + ZWNJ + "b", "a" + ZWJ + "b", "1" + ZWNJ + "2", "x" + ZWNJ + "\u0301"])
def test_joiners_between_letters_marks_numbers_or_symbols_are_allowed(name: str) -> None:
    assert validate_source_class(name) == name
    assert annotation(source_class=name).source_class == name  # kept verbatim, not normalised


@pytest.mark.parametrize(
    "name",
    [
        ZWNJ, ZWJ, ZWNJ + ZWJ, ZWNJ + ZWNJ,                      # alone / only joiners
        ZWNJ + "a", ZWJ + "a", "a" + ZWNJ, "a" + ZWJ,            # at the start or end
        "a" + ZWNJ + ZWNJ + "b", "a" + ZWNJ + ZWJ + "b", "a" + ZWJ + ZWNJ + "b",  # adjacent joiners
        "a " + ZWNJ + "b", "a" + ZWNJ + " b", "a\t" + ZWNJ + "b",  # next to whitespace
        "a-" + ZWNJ + "b", "a" + ZWJ + "-b", "a." + ZWNJ + "b", "a_" + ZWNJ + "b",  # next to punctuation
        "a" + ZWNJ + "\x00", "a" + ZWNJ + "\u200b" + "b",           # next to other controls / zero-width space
    ],
)
def test_joiners_in_unsafe_positions_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        validate_source_class(name)


@pytest.mark.parametrize(
    "char",
    [
        "\u200b",  # ZERO WIDTH SPACE
        "\u200e", "\u200f",  # LRM, RLM
        "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidi embeddings and overrides
        "\u2066", "\u2067", "\u2068", "\u2069",  # bidi isolates
        "\u061c",  # ARABIC LETTER MARK
        "\u2060", "\u2061", "\u2062", "\u2063", "\u2064",  # WORD JOINER and invisible operators
        "\ufeff",  # BOM / ZERO WIDTH NO-BREAK SPACE
        "\u00ad",  # SOFT HYPHEN
        "\u180e",  # MONGOLIAN VOWEL SEPARATOR
        "\ufff9", "\ufffa", "\ufffb",  # interlinear annotation
        "\U000E0041", "\U000E007F",  # tag characters
        "\u2028", "\u2029",  # line / paragraph separators
        "\x00", "\x07", "\x1b", "\x7f", "\x85",  # control characters
        "\ud800",  # lone surrogate
    ],
)
def test_dangerous_formatting_and_control_characters_stay_rejected_even_between_letters(char: str) -> None:
    for name in ("a" + char + "b", char + "ab", "ab" + char, "a" + ZWNJ + "b" + char):
        with pytest.raises(ValueError):
            validate_source_class(name)
    with pytest.raises(ValueError):
        annotation(source_class="a" + char + "b")


def test_allowing_joiners_does_not_weaken_the_other_class_name_rules() -> None:
    assert validate_source_class(" helmet ") == " helmet "
    assert validate_source_class("\u5934\u76d4") == "\u5934\u76d4"
    for bad in ("", " ", "\t", "a" * 257, None, 5):
        with pytest.raises(ValueError):
            validate_source_class(bad)
    long_ok = ("a" + ZWNJ) * 127 + "ab"  # 256 characters including joiners
    assert len(long_ok) == 256 and validate_source_class(long_ok) == long_ok
    with pytest.raises(ValueError):
        validate_source_class(long_ok + "c")


def test_a_name_with_a_joiner_is_a_different_key_from_the_name_without_it() -> None:
    plain, joined = "ab", "a" + ZWNJ + "b"
    assert validate_source_class(plain) != validate_source_class(joined)  # no silent normalisation
    assert annotation(source_class=plain) != annotation(source_class=joined)


# --------------------------------------------------------------------------- #
# Source hygiene: no hidden Unicode ("Trojan Source") anywhere in the project's files
# --------------------------------------------------------------------------- #
_HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})


def _project_text_files():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    suffixes = {".py", ".yaml", ".yml", ".toml", ".txt", ".md", ".cfg", ".ini"}
    found = []
    for folder in ("src", "tests", "configs"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
                found.append(path)
    found.extend(sorted(p for p in root.glob("*") if p.is_file() and p.suffix in suffixes))
    return root, found


def test_no_file_contains_invisible_bidirectional_or_control_characters() -> None:
    import unicodedata

    root, files = _project_text_files()
    assert len(files) > 10
    offenders = []
    for path in files:
        text = path.read_bytes().decode("utf-8")
        for line_number, line in enumerate(text.split("\n"), 1):
            for char in line:
                if char in "\t\r":
                    continue
                category = unicodedata.category(char)
                if category in _HIDDEN_CATEGORIES or (category == "Zs" and char != " "):
                    offenders.append(f"{path.relative_to(root)}:{line_number}: U+{ord(char):04X}")
    assert offenders == []


def test_library_source_is_pure_ascii_with_explicit_escapes() -> None:
    root, files = _project_text_files()
    library = [p for p in files if p.suffix == ".py" and "src" in p.relative_to(root).parts]
    assert library
    for path in library:
        text = path.read_bytes().decode("utf-8")
        assert all(ord(char) < 128 for char in text), path.name


def test_models_source_spells_the_joiners_as_escapes_not_literal_characters() -> None:
    import pathlib

    text = pathlib.Path(__file__).resolve().parent.parent.joinpath("src", "helmet_data", "models.py").read_text(encoding="utf-8")
    line = next(l for l in text.split("\n") if l.startswith("_ALLOWED_JOINERS = "))
    backslash = chr(92)
    assert backslash + "u200c" in line and backslash + "u200d" in line
    assert chr(0x200C) not in text and chr(0x200D) not in text


# --------------------------------------------------------------------------- #
# Blank-looking characters are refused in every position
# --------------------------------------------------------------------------- #
BLANK_LOOKING = [chr(0x3164), chr(0x115F), chr(0x034F)]  # Hangul Filler, Choseong Filler, CGJ
BLANK_IDS = ["U+3164", "U+115F", "U+034F"]


@pytest.mark.parametrize("char", BLANK_LOOKING, ids=BLANK_IDS)
@pytest.mark.parametrize(
    "template",
    [
        "{c}helmet",          # beginning
        "helmet{c}",          # end
        "hel{c}met",          # between ordinary letters
        "{c}",                # alone
        "{c}{c}",             # repeated
        "a{c}{c}b",           # repeated between letters
        "with {c}helmet",     # after a space
        "helmet{c} ",         # before trailing space
        "he" + chr(0x200C) + "{c}lmet",  # next to an allowed joiner
    ],
    ids=["start", "end", "between-letters", "alone", "repeated", "repeated-between", "after-space", "before-space", "next-to-joiner"],
)
def test_blank_looking_characters_are_rejected_in_every_position(char: str, template: str) -> None:
    name = template.replace("{c}", char)
    with pytest.raises(ValueError, match="blank-looking character"):
        validate_source_class(name)


@pytest.mark.parametrize("char", BLANK_LOOKING, ids=BLANK_IDS)
@pytest.mark.parametrize("template", ["{c}helmet", "helmet{c}", "hel{c}met"], ids=["start", "end", "between-letters"])
def test_source_annotation_rejects_blank_looking_class_names(char: str, template: str) -> None:
    name = template.replace("{c}", char)
    with pytest.raises(ValueError):
        annotation(source_class=name)
    with pytest.raises(ValueError):
        unmapped(source_class=name)
    with pytest.raises(ValueError):
        rejected(source_class=name)


@pytest.mark.parametrize("char", BLANK_LOOKING, ids=BLANK_IDS)
def test_the_clean_name_is_accepted_and_only_the_blank_character_makes_the_difference(char: str) -> None:
    assert validate_source_class("helmet") == "helmet"
    assert annotation(source_class="helmet").source_class == "helmet"
    with pytest.raises(ValueError):
        validate_source_class("helmet" + char)


def test_the_three_characters_are_exactly_hangul_filler_choseong_filler_and_cgj() -> None:
    import unicodedata

    assert [hex(ord(c)) for c in BLANK_LOOKING] == ["0x3164", "0x115f", "0x34f"]
    assert unicodedata.name(BLANK_LOOKING[0]) == "HANGUL FILLER"
    assert unicodedata.name(BLANK_LOOKING[1]) == "HANGUL CHOSEONG FILLER"
    assert unicodedata.name(BLANK_LOOKING[2]) == "COMBINING GRAPHEME JOINER"


def test_ordinary_combining_marks_are_still_allowed() -> None:
    names = [
        "e" + chr(0x0301),                       # e + combining acute
        "a" + chr(0x0308) + "b",                 # a + combining diaeresis
        chr(0x0915) + chr(0x094D) + chr(0x0937),  # Devanagari conjunct (virama is a mark)
        chr(0x0645) + chr(0x064E) + chr(0x062F),  # Arabic letter + fatha
        "x" + chr(0x0300) + chr(0x0301) + chr(0x0302),  # several stacked marks
        chr(0x1F44D) + chr(0xFE0F),              # emoji + variation selector-16
    ]
    for name in names:
        assert validate_source_class(name) == name
        assert annotation(source_class=name).source_class == name


def test_joiner_rules_and_legitimate_examples_are_unchanged_by_the_blank_character_rule() -> None:
    for name in (PERSIAN, INDIC, EMOJI, EMOJI_FAMILY, "a" + ZWNJ + "b", "a" + ZWJ + "b", "1" + ZWNJ + "2"):
        assert validate_source_class(name) == name
    for name in (ZWNJ, ZWJ, ZWNJ + "a", "a" + ZWJ, "a" + ZWNJ + ZWJ + "b", "a " + ZWNJ + "b", "a-" + ZWNJ + "b"):
        with pytest.raises(ValueError):
            validate_source_class(name)
    for char in (0x200B, 0x200E, 0x202E, 0x2060, 0xFEFF, 0x00AD, 0x180E, 0x2028):
        with pytest.raises(ValueError):
            validate_source_class("a" + chr(char) + "b")
