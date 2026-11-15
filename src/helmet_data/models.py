"""Shared enums, validators and immutable data types.

Design rules:
* All dataclasses are frozen and use only immutable field types (tuples, strings,
  numbers, enums, None), validated on construction, so every instance is hashable.
* Invalid numeric box values (NaN, infinity, non-numbers) are rejected at
  construction time (``InvalidBoxError``). Readers must catch that error and record
  the annotation as rejected with ``ReasonCode.INVALID_BOX`` and the raw text, so
  equality and hashing are always well defined.
* String enums serialise as their ``.value`` under ``str()``, ``format()`` and
  f-strings, on every supported Python version. Use ``.value`` explicitly when
  writing JSON/CSV anyway.

Requirements deferred to Milestone 1.2 (recorded here deliberately; not yet implemented):
* Dataset names: also reject Windows reserved device names (``nul``, ``con``, ``prn``,
  ``aux``, ``com1``-``com9``, ``lpt1``-``lpt9``), because a dataset name becomes the
  directory ``data/raw/<name>``.
* ``OutputRecord.output_name``: validate it with the same portable file-name rules as
  ``paths.check_segment`` (reserved names, trailing dots/spaces, forbidden characters).
* ``SourceAnnotation.annotation_id``: define a deterministic id scoped to its source
  record (for example derived from ``record_id`` plus the annotation's position or
  source id), so ids stay unique across images for provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Optional

# Strict whole-string patterns, always used with ``fullmatch``. A ``$`` anchor would
# accept a trailing newline, so ``\A``/``\Z`` are used and nothing else may follow.
_DATASET_NAME = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_SOURCE_ID_VERSION = "helmet_data.source_record.v1"


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #
def validate_dataset_name(name: Any) -> str:
    """Dataset names are lowercase ``[a-z0-9_-]``, start alphanumeric, max 64 chars."""
    if not isinstance(name, str) or not _DATASET_NAME.fullmatch(name):
        raise ValueError(f"invalid dataset name: {name!r} (expected [a-z0-9][a-z0-9_-]{{0,63}})")
    return name


def validate_sha256(value: Any, what: str = "sha256") -> str:
    """A SHA-256 is exactly 64 lowercase hexadecimal characters."""
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"invalid {what}: {value!r} (expected 64 lowercase hex characters)")
    return value


def validate_relative_posix_path(path: Any) -> str:
    """A safe relative path using ``/`` separators.

    Rejects absolute paths, backslashes, ``:``, control characters, empty segments
    and ``.``/``..`` segments.
    """
    if not isinstance(path, str) or path == "":
        raise ValueError(f"invalid relative path: {path!r}")
    if path.startswith("/") or "\\" in path or ":" in path:
        raise ValueError(f"relative path must be POSIX-style and not absolute: {path!r}")
    if any(ord(ch) < 32 for ch in path):
        raise ValueError(f"relative path contains control characters: {path!r}")
    for piece in path.split("/"):
        if piece in ("", ".", ".."):
            raise ValueError(f"unsafe segment {piece!r} in relative path: {path!r}")
    return path


def _require_int(name: str, value: Any, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _require_nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class _ValueStrEnum(str, Enum):
    """String enum whose ``str()`` and ``format()`` are the plain value."""

    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, format_spec: str) -> str:
        return format(str(self.value), format_spec)


class Decision(_ValueStrEnum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


class AnnotationStatus(_ValueStrEnum):
    MAPPED = "mapped"
    EXCLUDED = "excluded"
    REJECTED = "rejected"
    REVIEW = "review"


class Split(_ValueStrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class BoxSemantics(_ValueStrEnum):
    """What a source class's boxes actually represent."""

    HEAD_REGION_SINGLE = "head_region_single"
    HELMET_ONLY = "helmet_only"
    WHOLE_RIDER = "whole_rider"
    VEHICLE = "vehicle"
    OTHER = "other"
    UNKNOWN = "unknown"


class TargetClass(IntEnum):
    """YOLO class ids. Serialise with ``int(x)`` or ``x.label``, never ``str(x)``."""

    WITH_HELMET = 0
    WITHOUT_HELMET = 1

    @property
    def label(self) -> str:
        return self.name.lower()


class ReasonCode(_ValueStrEnum):
    """Why a sample, image or annotation was reviewed, rejected or excluded.

    The exact set is pinned by tests; adding a code requires a deliberate review.
    Every code is described in ``REASON_CODE_DOCS``.
    """

    SMALL_PROJECTED_BOX = "small_projected_box"
    OUT_OF_BOUNDS_MINOR = "out_of_bounds_minor"
    OUT_OF_BOUNDS_MAJOR = "out_of_bounds_major"
    INVALID_BOX = "invalid_box"
    INVALID_CLASS_ID = "invalid_class_id"
    CONTAINS_REJECTED_BOX = "contains_rejected_box"
    HAS_EXCLUDED_ANNOTATION = "has_excluded_annotation"
    EXACT_DUPLICATE = "exact_duplicate"
    EXACT_DUPLICATE_CONFLICT = "exact_duplicate_conflict"
    EXACT_DUPLICATE_UNRESOLVED = "exact_duplicate_unresolved"
    EXIF_ORIENTATION = "exif_orientation"
    UNSUPPORTED_FORMAT = "unsupported_format"
    CORRUPT_IMAGE = "corrupt_image"
    POSSIBLE_MULTI_HEAD = "possible_multi_head"
    CLASS_CONFLICT = "class_conflict"
    OVERSIZED_GROUP = "oversized_group"
    LOW_VARIANCE_IMAGE = "low_variance_image"
    UNKNOWN_SEMANTICS = "unknown_semantics"
    UNLABELED = "unlabeled"
    RIDER_WITHOUT_HEAD_BOX = "rider_without_head_box"
    OCCUPANT_COUNT_MISMATCH = "occupant_count_mismatch"
    POSSIBLE_NON_OCCUPANT = "possible_non_occupant"
    LINKED_TO_FROZEN_TEST = "linked_to_frozen_test"
    LEDGER_CONFLICT = "ledger_conflict"
    EXCLUDED_UNREVIEWED = "excluded_unreviewed"
    ISCROWD = "iscrowd"


REASON_CODE_DOCS: dict[ReasonCode, str] = {
    ReasonCode.SMALL_PROJECTED_BOX: (
        "REVIEW. The box's projected short side (at train_imgsz) is below "
        "min_projected_side_px. Never an automatic rejection."
    ),
    ReasonCode.OUT_OF_BOUNDS_MINOR: (
        "REVIEW. The box extends outside the image by at most oob_tolerance_px. "
        "Only an explicit 'approve_clip' decision may clip it; it is never clipped silently."
    ),
    ReasonCode.OUT_OF_BOUNDS_MAJOR: (
        "REJECT (a reviewer may override). The box extends outside the image by more "
        "than oob_tolerance_px."
    ),
    ReasonCode.INVALID_BOX: (
        "REJECT. The box is malformed: non-numeric, NaN/infinite, or zero/negative size."
    ),
    ReasonCode.INVALID_CLASS_ID: "REJECT. The class id is outside the source dataset's class list.",
    ReasonCode.CONTAINS_REJECTED_BOX: (
        "REVIEW the whole image. At least one of its boxes was rejected, so the image may "
        "contain an unlabeled head. Only an explicit 'drop_box_keep_image' decision keeps it."
    ),
    ReasonCode.HAS_EXCLUDED_ANNOTATION: (
        "REVIEW the whole image. It has an annotation excluded for a reason other than a "
        "deliberately ignored class (for example a COCO crowd region)."
    ),
    ReasonCode.EXACT_DUPLICATE: (
        "Outcome label (not a review item). This source record has the same file bytes "
        "(SHA-256) as a canonical record and was merged into it in the training output. "
        "The source record itself is always preserved. Logged in dedup_log.csv."
    ),
    ReasonCode.EXACT_DUPLICATE_CONFLICT: (
        "REVIEW. Exact-duplicate images whose mapped annotations are NOT equivalent "
        "(different box counts, classes, or IoU below dedup.box_match_iou)."
    ),
    ReasonCode.EXACT_DUPLICATE_UNRESOLVED: (
        "REVIEW. Exact-duplicate images whose annotations ARE equivalent, but merging is not "
        "yet allowed (licenses not shown compatible, or another review flag is open). "
        "Both stay in the same split until a human decides. Disjoint from "
        "EXACT_DUPLICATE_CONFLICT."
    ),
    ReasonCode.EXIF_ORIENTATION: (
        "REVIEW. The EXIF orientation is not 1, so boxes may be rotated relative to the "
        "image that OpenCV/Ultralytics will load."
    ),
    ReasonCode.UNSUPPORTED_FORMAT: "REVIEW. The file extension/format is not readable by Ultralytics.",
    ReasonCode.CORRUPT_IMAGE: "REJECT. The image is unreadable, truncated, or fails full decoding.",
    ReasonCode.POSSIBLE_MULTI_HEAD: (
        "REVIEW. The box may contain more than one head (shape, size or overlap heuristics)."
    ),
    ReasonCode.CLASS_CONFLICT: (
        "REVIEW. Heavily overlapping boxes carry different target classes."
    ),
    ReasonCode.OVERSIZED_GROUP: (
        "REVIEW. A duplicate/frame group exceeds max_group_fraction of all images."
    ),
    ReasonCode.LOW_VARIANCE_IMAGE: (
        "REVIEW. The image is nearly blank, so its dHash is unreliable for near-duplicate grouping."
    ),
    ReasonCode.UNKNOWN_SEMANTICS: (
        "REVIEW/BLOCK. A source class has no approved box_semantics, so it cannot be mapped."
    ),
    ReasonCode.UNLABELED: (
        "REJECT from the output. The annotation file/entry is missing, or empty without the "
        "required dataset-level confirmation that empty means background."
    ),
    ReasonCode.RIDER_WITHOUT_HEAD_BOX: (
        "REVIEW. A source rider box has no head box inside it (occupant-coverage cross-check)."
    ),
    ReasonCode.OCCUPANT_COUNT_MISMATCH: (
        "REVIEW. Fewer head boxes than visible motorcycle occupants (occupant-coverage cross-check)."
    ),
    ReasonCode.POSSIBLE_NON_OCCUPANT: (
        "REVIEW. A head box may belong to a pedestrian/cyclist rather than a motorcycle occupant."
    ),
    ReasonCode.LINKED_TO_FROZEN_TEST: (
        "EXCLUDE from the current build. A new image is linked to a frozen test group; it can "
        "only join the test split through an approved test_membership_amendment."
    ),
    ReasonCode.LEDGER_CONFLICT: (
        "BLOCK the build. New duplicate/frame links would connect groups previously assigned "
        "to different splits."
    ),
    ReasonCode.EXCLUDED_UNREVIEWED: (
        "EXCLUDE. The image still had unresolved review items and the build ran with "
        "--allow-unreviewed; such images are never included in training."
    ),
    ReasonCode.ISCROWD: (
        "EXCLUDED annotation. A COCO annotation with iscrowd=1; recorded in "
        "excluded_annotations.csv and the image goes to REVIEW (has_excluded_annotation)."
    ),
}


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
class InvalidBoxError(ValueError):
    """Raised when box coordinates are not finite numbers."""


@dataclass(frozen=True)
class PixelBox:
    """A box in pixel-edge coordinates (origin at the image's top-left corner).

    Coordinates must be finite real numbers; they are stored as floats (negative zero
    normalised to zero), so equality and hashing are deterministic. Geometric validity
    (ordering, size, image bounds) is checked later by the box validators.
    """

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidBoxError(f"{name} must be a real number, got {value!r}")
            try:
                number = float(value)
            except OverflowError as exc:
                raise InvalidBoxError(f"{name} is too large: {value!r}") from exc
            if not math.isfinite(number):
                raise InvalidBoxError(f"{name} must be finite, got {value!r}")
            object.__setattr__(self, name, number + 0.0)


def source_record_id(dataset: str, relative_path: str, sha256: str) -> str:
    """Deterministic id from dataset, relative path and file hash.

    The inputs are validated and encoded as a versioned JSON array, so the encoding is
    unambiguous. Identical bytes at different paths get different ids.
    """
    payload = json.dumps(
        [
            _SOURCE_ID_VERSION,
            validate_dataset_name(dataset),
            validate_relative_posix_path(relative_path),
            validate_sha256(sha256),
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SourceRecord:
    """One raw source image. Never deleted or merged."""

    dataset: str
    relative_path: str
    sha256: str
    size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    pixel_hash: Optional[str] = None
    dhash: Optional[int] = None
    exif_orientation: Optional[int] = None
    image_format: Optional[str] = None

    def __post_init__(self) -> None:
        validate_dataset_name(self.dataset)
        validate_relative_posix_path(self.relative_path)
        validate_sha256(self.sha256)
        _require_int("size_bytes", self.size_bytes, minimum=0)
        if self.width is not None:
            _require_int("width", self.width, minimum=1)
        if self.height is not None:
            _require_int("height", self.height, minimum=1)
        if self.pixel_hash is not None:
            validate_sha256(self.pixel_hash, "pixel_hash")
        if self.dhash is not None:
            _require_int("dhash", self.dhash, minimum=0, maximum=2**64 - 1)
        if self.exif_orientation is not None:
            _require_int("exif_orientation", self.exif_orientation, minimum=1, maximum=8)
        if self.image_format is not None:
            _require_nonempty_str("image_format", self.image_format)

    @property
    def record_id(self) -> str:
        return source_record_id(self.dataset, self.relative_path, self.sha256)


@dataclass(frozen=True)
class SourceAnnotation:
    """One raw annotation, with a stable id and an outcome status."""

    annotation_id: str
    record_id: str
    source_class: str
    box: PixelBox
    status: AnnotationStatus
    reason: Optional[ReasonCode] = None
    target_class: Optional[TargetClass] = None

    def __post_init__(self) -> None:
        _require_nonempty_str("annotation_id", self.annotation_id)
        validate_sha256(self.record_id, "record_id")
        _require_nonempty_str("source_class", self.source_class)
        if not isinstance(self.box, PixelBox):
            raise TypeError("box must be a PixelBox")
        if not isinstance(self.status, AnnotationStatus):
            raise TypeError("status must be an AnnotationStatus")
        if self.reason is not None and not isinstance(self.reason, ReasonCode):
            raise TypeError("reason must be a ReasonCode or None")
        if self.target_class is not None and not isinstance(self.target_class, TargetClass):
            raise TypeError("target_class must be a TargetClass or None")


def _validate_id_tuple(name: str, value: Any) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(value) == 0:
        raise ValueError(f"{name} must not be empty")
    for item in value:
        validate_sha256(item, f"{name} entry")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicates")


@dataclass(frozen=True)
class OutputRecord:
    """One image in the training output, linked to one or more source records."""

    output_name: str
    source_record_ids: tuple[str, ...]
    split: Optional[Split] = None
    group_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_nonempty_str("output_name", self.output_name)
        if "/" in self.output_name or "\\" in self.output_name or self.output_name in (".", ".."):
            raise ValueError(f"output_name must be a plain file name: {self.output_name!r}")
        _validate_id_tuple("source_record_ids", self.source_record_ids)
        if self.split is not None and not isinstance(self.split, Split):
            raise TypeError("split must be a Split or None")
        if self.group_id is not None:
            _require_nonempty_str("group_id", self.group_id)


@dataclass(frozen=True)
class Group:
    """A set of source records that must stay in the same split."""

    group_id: str
    member_record_ids: tuple[str, ...]
    link_types: tuple[str, ...] = ()
    split: Optional[Split] = None

    def __post_init__(self) -> None:
        _require_nonempty_str("group_id", self.group_id)
        _validate_id_tuple("member_record_ids", self.member_record_ids)
        if not isinstance(self.link_types, tuple):
            raise TypeError("link_types must be a tuple")
        for item in self.link_types:
            _require_nonempty_str("link_types entry", item)
        if self.split is not None and not isinstance(self.split, Split):
            raise TypeError("split must be a Split or None")
