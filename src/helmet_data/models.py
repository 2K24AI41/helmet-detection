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

Changes in Milestone 1.2.0 (BREAKING where noted):
* ``validate_dataset_name`` additionally rejects Windows reserved device names in any
  case (``nul``, ``CON``, ``com1``-``com9``, ``lpt1``-``lpt9``), because a dataset name
  becomes the directory ``data/raw/<name>``. BREAKING: names such as ``nul`` that were
  valid in 1.1 are now refused.
* ``validate_output_filename`` (new) defines the portable output-file-name rules and
  ``OutputRecord.output_name`` uses it. BREAKING: names must be ASCII, at most 128
  characters, with an allowed image extension.
* ``SourceAnnotation`` was redesigned. BREAKING: ``annotation_id`` is no longer a
  constructor argument; it is a derived property (like ``SourceRecord.record_id``) built
  from ``record_id``, ``source_file`` and ``source_locator``. ``box`` is optional and a
  bounded ``raw_geometry`` text holds geometry that could not become a valid
  ``PixelBox``. ``source_file``, ``source_locator`` and ``flags`` are new; the argument
  order changed. Nothing in 1.1 consumed ``SourceAnnotation`` beyond its own tests.
* ``AnnotationStatus.UNMAPPED`` (new): parsed successfully, class not yet mapped. It is
  neither accepted for training nor a review finding.
* ``ReasonCode`` now has 35 codes (26 from 1.1 plus nine reader/source-validation codes);
  the definitions are in ``REASON_CODE_DOCS`` and the exact set is pinned by tests.
* Class names may contain U+200C/U+200D only between two letters/marks/numbers/symbols;
  all other format characters stay refused.
* ``OUTPUT_NAME_HASH_HEX_CHARS = 16`` records the approved hash-prefix length for future
  output names; the writer itself is not implemented yet.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Optional

from .paths import RESERVED_DEVICE_NAMES, check_segment

# Strict whole-string patterns, always used with ``fullmatch``. A ``$`` anchor would
# accept a trailing newline, so ``\A``/``\Z`` are used and nothing else may follow.
_DATASET_NAME = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_SOURCE_ID_VERSION = "helmet_data.source_record.v1"
_ANNOTATION_ID_VERSION = "helmet_data.source_annotation.v1"

#: Output image file names: ASCII only, alphanumeric first character.
_OUTPUT_NAME = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")
#: Source locators: ``key:number`` parts joined by ``/``; canonical decimal numbers only
#: (no leading zeros), so one position has exactly one spelling.
_LOCATOR = re.compile(r"\A[a-z]+:(?:0|[1-9][0-9]{0,17})(?:/[a-z]+:(?:0|[1-9][0-9]{0,17}))*\Z")
_FLAG = re.compile(r"\A[a-z][a-z0-9_]{0,31}\Z")

#: Extensions allowed for output images (compared case-insensitively; the original
#: spelling is preserved). These are formats Ultralytics and Pillow can both read.
OUTPUT_IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")
#: Maximum output file name length, measured in characters of the whole name including the
#: extension. Names are ASCII-only, so this equals bytes and UTF-16 units. 128 leaves room
#: for ``processed/<64-char build id>/images/<split>/`` below Windows' 260-character path
#: limit when the data root is under ~60 characters; it is deliberately conservative.
MAX_OUTPUT_FILENAME_LENGTH = 128
#: Bounds for text copied from untrusted annotation files into records and reports.
MAX_RAW_GEOMETRY_CHARS = 256
MAX_SOURCE_CLASS_CHARS = 256
MAX_LOCATOR_CHARS = 200
MAX_FLAGS = 16
_TRUNCATION_MARKER = "...[truncated]"
#: Unicode categories refused in class names: control, format (bidi controls, zero-width
#: space, word joiner, BOM, soft hyphen, tag characters, ...), surrogate, and
#: line/paragraph separators.
_FORBIDDEN_CLASS_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
#: The only format characters allowed in class names: ZERO WIDTH NON-JOINER (U+200C) and
#: ZERO WIDTH JOINER (U+200D), which are required by Persian, Indic and emoji text. They
#: are allowed only between two letters/marks/numbers/symbols (see ``validate_source_class``).
_ALLOWED_JOINERS = frozenset({"\u200c", "\u200d"})
_JOINER_NEIGHBOUR_CATEGORIES = frozenset("LMNS")  # first letter of the Unicode category
#: Characters that render as blank or invisible although they are not format characters, so
#: two class names differing only by one of them would look identical. Refused in any
#: position (start, end, between letters): U+3164 HANGUL FILLER, U+115F HANGUL CHOSEONG
#: FILLER and U+034F COMBINING GRAPHEME JOINER. Built with chr() so the source file itself
#: contains no invisible character. Combining marks in general are NOT refused.
_BLANK_LOOKING_CHARACTERS = frozenset({chr(0x3164), chr(0x115F), chr(0x034F)})

#: Length of the SHA-256 hexadecimal prefix used in generated output file names. APPROVED
#: for the future writer (Milestone 1.7), which is NOT implemented yet. The planned name is
#: ``<dataset>__<first 16 hex characters of the image's SHA-256><original extension>``.
#: Eight characters (32 bits) were rejected: chance collisions become likely at roughly
#: 77,000 images. Longest planned name: 64 + 2 + 16 + 5 = 87 characters, within
#: ``MAX_OUTPUT_FILENAME_LENGTH``. The writer must still detect collisions explicitly.
OUTPUT_NAME_HASH_HEX_CHARS = 16


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #
def validate_dataset_name(name: Any) -> str:
    """Dataset names are lowercase ``[a-z0-9_-]``, start alphanumeric, max 64 chars.

    Windows reserved device names (``nul``, ``con``, ``prn``, ``aux``, ``com1``-``com9``,
    ``lpt1``-``lpt9``) are refused in any case, also with an extension or trailing
    spaces (``NUL.txt``, ``Con ``), because the name becomes a directory name.
    """
    if isinstance(name, str) and name.split(".")[0].rstrip(" ").lower() in RESERVED_DEVICE_NAMES:
        raise ValueError(f"dataset name is a reserved Windows device name: {name!r}")
    if not isinstance(name, str) or not _DATASET_NAME.fullmatch(name):
        raise ValueError(f"invalid dataset name: {name!r} (expected [a-z0-9][a-z0-9_-]{{0,63}})")
    return name


def validate_output_filename(name: Any) -> str:
    """Portable Windows-compatible name of an output image file.

    Rules: a non-empty ``str`` of at most ``MAX_OUTPUT_FILENAME_LENGTH`` characters
    (counted as ``len(name)``, i.e. including the extension; names are ASCII so this is
    also the byte length); only ``[A-Za-z0-9._-]`` and an alphanumeric first character
    (so no separators, control characters, spaces or non-ASCII); no ``..``; no
    trailing dot or space; not a Windows reserved device name even with an extension;
    and an extension from ``OUTPUT_IMAGE_EXTENSIONS`` (any letter case) after a
    non-empty stem. The extension spelling is preserved, never changed.
    """
    if not isinstance(name, str) or name == "":
        raise ValueError(f"output file name must be a non-empty string, got {name!r}")
    if len(name) > MAX_OUTPUT_FILENAME_LENGTH:
        raise ValueError(
            f"output file name is {len(name)} characters; the limit is {MAX_OUTPUT_FILENAME_LENGTH}"
        )
    if "/" in name or "\\" in name:
        raise ValueError(f"output file name must not contain path separators: {name!r}")
    check_segment(name)  # control characters, <>:"|?*, trailing dot/space, reserved names
    if ".." in name or not _OUTPUT_NAME.fullmatch(name):
        raise ValueError(f"output file name is not portable ASCII [A-Za-z0-9._-]: {name!r}")
    stem, dot, extension = name.rpartition(".")
    if not dot or stem == "":
        raise ValueError(f"output file name needs a stem and an extension: {name!r}")
    if "." + extension.lower() not in OUTPUT_IMAGE_EXTENSIONS:
        raise ValueError(
            f"output file extension .{extension} is not one of {OUTPUT_IMAGE_EXTENSIONS}: {name!r}"
        )
    return name


def validate_source_class(name: Any) -> str:
    """A class key copied from an untrusted file: bounded, printable, kept verbatim.

    Control, format (for example bidi overrides, zero-width space, word joiner, BOM, soft
    hyphen, tag characters), surrogate and line/paragraph-separator characters are
    refused; whitespace-only names are refused; case and surrounding spaces are preserved
    because mapping must match them exactly.

    Exception: ZERO WIDTH NON-JOINER (U+200C) and ZERO WIDTH JOINER (U+200D), which
    Persian, Indic and emoji text need, are allowed, but only when the character before
    and the character after each one is a letter, mark, number or symbol (Unicode
    category L*, M*, N* or S*). A joiner at the start or end, next to whitespace or
    punctuation, next to another joiner, or alone is refused, so invisible characters
    cannot be used to make two visually identical names differ.

    Also refused in any position: U+3164 HANGUL FILLER, U+115F HANGUL CHOSEONG FILLER and
    U+034F COMBINING GRAPHEME JOINER, which are blank or invisible but are not format
    characters. Other combining marks remain allowed.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError(f"source class must be a non-blank string, got {name!r}")
    if len(name) > MAX_SOURCE_CLASS_CHARS:
        raise ValueError(f"source class is longer than {MAX_SOURCE_CLASS_CHARS} characters")
    for index, char in enumerate(name):
        if char in _BLANK_LOOKING_CHARACTERS:
            raise ValueError(f"source class contains a blank-looking character U+{ord(char):04X}")
        if char in _ALLOWED_JOINERS:
            before = name[index - 1] if index > 0 else ""
            after = name[index + 1] if index + 1 < len(name) else ""
            if not (
                before
                and after
                and unicodedata.category(before)[0] in _JOINER_NEIGHBOUR_CATEGORIES
                and unicodedata.category(after)[0] in _JOINER_NEIGHBOUR_CATEGORIES
            ):
                raise ValueError(
                    f"U+{ord(char):04X} is only allowed between two letters, marks, numbers or symbols"
                )
        elif unicodedata.category(char) in _FORBIDDEN_CLASS_CATEGORIES:
            raise ValueError(f"source class contains a forbidden character U+{ord(char):04X}")
    return name


#: Identifier namespaces in which a source file can declare the same identifier twice.
#: A ``duplicate_source_id`` finding must name exactly one of these.
SOURCE_ID_NAMESPACES = (
    "coco.annotation_id",
    "coco.category_id",
    "coco.category_name",
    "coco.image_id",
    "yolo.class_id",
    "yolo.class_name",
)


def validate_source_id_namespace(namespace: Any) -> str:
    """The identifier namespace a ``duplicate_source_id`` finding refers to."""
    if not isinstance(namespace, str) or namespace not in SOURCE_ID_NAMESPACES:
        raise ValueError(f"unknown source identifier namespace: {namespace!r}; expected one of {SOURCE_ID_NAMESPACES}")
    return namespace


def validate_source_locator(locator: Any) -> str:
    """A stable position inside an annotation file, for example ``line:12``.

    Grammar: ``key:number`` parts joined by ``/`` where ``key`` is lowercase letters and
    ``number`` is a canonical decimal (``0`` or no leading zero), at most 18 digits. Formats
    use: YOLO ``line:<1-based line>``, VOC ``object:<0-based index>``, COCO
    ``index:<0-based array index>/id:<annotation id>``.
    """
    if not isinstance(locator, str) or len(locator) > MAX_LOCATOR_CHARS or not _LOCATOR.fullmatch(locator):
        raise ValueError(f"invalid source locator: {locator!r}")
    return locator


def bounded_raw_text(text: Any, limit: int = MAX_RAW_GEOMETRY_CHARS) -> str:
    """Escape and bound untrusted text so it is safe to store, print and serialise.

    The result is printable ASCII only (non-ASCII and control characters become
    ``\\xNN``/``\\uNNNN`` escapes, a backslash becomes ``\\\\``) and at most ``limit``
    characters; longer text is cut and ends with ``...[truncated]``. Deterministic.
    """
    if not isinstance(text, str):
        raise TypeError("raw text must be a string")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= len(_TRUNCATION_MARKER):
        raise ValueError(f"limit must be an integer larger than {len(_TRUNCATION_MARKER)}")
    escaped = text.encode("unicode_escape").decode("ascii")
    if len(escaped) <= limit:
        return escaped
    return escaped[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


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
    """Outcome of an annotation.

    * ``UNMAPPED``: parsed successfully, geometry valid, but its source class has no
      approved mapping yet. It is NOT accepted for training and it is NOT a review
      finding: it is simply pending the (human-approved) class mapping.
    * ``MAPPED``: mapped to a target class (still subject to the review gate).
    * ``REVIEW``: needs a human decision; carries a reason.
    * ``REJECTED``: unusable as given; carries a reason; never silently dropped.
    * ``EXCLUDED``: deliberately not used (for example COCO ``iscrowd``); carries a reason.
    """

    MAPPED = "mapped"
    EXCLUDED = "excluded"
    REJECTED = "rejected"
    REVIEW = "review"
    UNMAPPED = "unmapped"


#: Statuses that are findings: each requires a ``ReasonCode`` and, directly or through
#: its image, a human decision or an audit entry. ``UNMAPPED`` is deliberately absent.
FINDING_STATUSES = frozenset(
    {AnnotationStatus.REVIEW, AnnotationStatus.REJECTED, AnnotationStatus.EXCLUDED}
)
#: Statuses that are waiting for a mapping, not for a decision about a problem.
PENDING_MAPPING_STATUSES = frozenset({AnnotationStatus.UNMAPPED})
#: Only a mapped annotation can ever feed training. This is necessary, not sufficient:
#: the review gate still applies. ``UNMAPPED`` is never in this set.
TRAINING_CANDIDATE_STATUSES = frozenset({AnnotationStatus.MAPPED})


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
    # Added in Milestone 1.2.0 (reader and source-validation outcomes):
    MALFORMED_ANNOTATION = "malformed_annotation"
    UNSUPPORTED_ANNOTATION = "unsupported_annotation"
    DUPLICATE_SOURCE_ID = "duplicate_source_id"
    ORPHAN_ANNOTATION = "orphan_annotation"
    AMBIGUOUS_PAIRING = "ambiguous_pairing"
    IMAGE_SIZE_MISMATCH = "image_size_mismatch"
    UNSAFE_REFERENCE = "unsafe_reference"
    LIMIT_EXCEEDED = "limit_exceeded"
    DUPLICATE_BOX = "duplicate_box"


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
        "REJECT. Geometry is present but unusable: non-numeric values, non-finite values "
        "(NaN or infinity), the wrong number of values (wrong arity, for example a COCO bbox "
        "that does not have exactly 4 numbers), or a degenerate box (zero or negative size, "
        "inverted edges). The text is kept as bounded raw_geometry, never as a PixelBox. "
        "Geometry missing entirely is malformed_annotation; a valid box outside the image "
        "is out_of_bounds_*."
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
    ReasonCode.CORRUPT_IMAGE: (
        "REJECT. In Milestone 1.2 this concerns header inspection only: the image header is "
        "unreadable, unrecognised, truncated, or declares impossible dimensions. It does NOT "
        "mean the pixel data was fully decoded and found bad; full-decode checking is a later "
        "milestone (1.4) and will reuse this code."
    ),
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
        "BLOCK (class-level, never per annotation). A source class has no approved "
        "box_semantics or mapping, so no annotation of that class can be mapped and a build "
        "that includes the class is refused until a human approves it. The class's "
        "annotations stay UNMAPPED; semantics are never inferred from the class name."
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
    ReasonCode.MALFORMED_ANNOTATION: (
        "REJECT. Source text that cannot be interpreted as an annotation. Every finding must "
        "state its scope, which is exactly one of: ANNOTATION-level, where a single record is "
        "unusable (wrong column count, missing or wrongly typed required field, no bounding-box "
        "element, unusable class field) and only that annotation is rejected; or FILE-level, "
        "where an annotation or class-names file cannot be read or parsed at all (undecodable, "
        "invalid XML/JSON, forbidden DTD or entity, duplicate JSON key, NaN constant) so every "
        "annotation it would have supplied is unavailable. Present-but-unusable geometry is "
        "invalid_box; a well-formed non-box shape is unsupported_annotation."
    ),
    ReasonCode.UNSUPPORTED_ANNOTATION: (
        "REJECT (kept as raw text, never converted). A well-formed annotation that is not an "
        "axis-aligned bounding box: a YOLO segmentation polygon, a COCO annotation with only "
        "a segmentation, or a VOC polygon or other shape. It is never turned into a box, "
        "because that would be an inference."
    ),
    ReasonCode.DUPLICATE_SOURCE_ID: (
        "REVIEW. The source file itself declares the same identifier twice within ONE "
        "identifier namespace; the finding must name the namespace, one of "
        "SOURCE_ID_NAMESPACES (coco.image_id, coco.annotation_id, coco.category_id, "
        "coco.category_name, yolo.class_id, yolo.class_name), and the duplicated value. "
        "Identifiers from different namespaces never conflict. This is not about this "
        "project's derived annotation ids (unique by construction) and not about two "
        "identical boxes (duplicate_box)."
    ),
    ReasonCode.ORPHAN_ANNOTATION: (
        "REJECT. An annotation (or annotation file) refers to an image that does not exist in "
        "the dataset: a COCO image_id with no matching image entry or file, or a label/XML "
        "file with no matching image. The opposite case, an image with no annotation, is "
        "unlabeled."
    ),
    ReasonCode.AMBIGUOUS_PAIRING: (
        "REVIEW (the image). The link between an image and its annotation cannot be decided "
        "without guessing: the same file stem with several image extensions, names that "
        "differ only by letter case, several annotation files for one image, or two entries "
        "naming one image file. No pairing is ever guessed."
    ),
    ReasonCode.IMAGE_SIZE_MISMATCH: (
        "REVIEW (the image). The width/height declared by the annotation source (VOC <size>, "
        "COCO width/height) differs from the image header or is not positive. Boxes are never "
        "rescaled to fit."
    ),
    ReasonCode.UNSAFE_REFERENCE: (
        "REJECT (the annotation, which becomes an orphan). A path-like reference inside "
        "annotation content is unsafe: parent traversal, absolute, drive, UNC, device or "
        "extended-length path, alternate data stream, backslash separator, reserved Windows "
        "name, or control character. It is never used to open any file."
    ),
    ReasonCode.LIMIT_EXCEEDED: (
        "REJECT (the whole file or image that exceeded the limit; nothing is read partially "
        "or truncated silently). A configured resource limit was exceeded: file size, line "
        "length, annotation count, nesting depth, or image pixel count (including a "
        "decompression-bomb warning)."
    ),
    ReasonCode.DUPLICATE_BOX: (
        "REVIEW only (annotation-level). Two or more annotations of one image have exactly "
        "identical box coordinates and the same source class. Every one of them is kept and "
        "reported for a human decision; none is silently removed, merged or dropped. The same "
        "box under different classes is class_conflict, a later check."
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


def source_annotation_id(record_id: str, source_file: str, source_locator: str) -> str:
    """Deterministic, collision-resistant id of one annotation.

    SHA-256 of a versioned JSON array ``[version, record_id, source_file, locator]``
    (ASCII-escaped, compact separators), returned as 64 lowercase hex characters. The
    inputs are the image's ``record_id`` (which already encodes dataset, path and file
    hash), the dataset-relative POSIX path of the annotation file, and a stable locator
    inside that file. Never used: absolute paths, timestamps, random values or Python's
    ``hash()``. The same locator in a different file or for a different record yields a
    different id, and moving content between files changes the id.
    """
    payload = json.dumps(
        [
            _ANNOTATION_ID_VERSION,
            validate_sha256(record_id, "record_id"),
            validate_relative_posix_path(source_file),
            validate_source_locator(source_locator),
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SourceAnnotation:
    """One raw annotation, kept apart from any geometry that could not be parsed.

    Identity: ``annotation_id`` is derived from ``record_id``, ``source_file`` and
    ``source_locator`` (see ``source_annotation_id``); it is not stored or passed in.

    Geometry is either valid or raw, never both and never disguised:
    * ``box`` is a ``PixelBox`` whose coordinates are finite numbers; or
    * ``box`` is ``None`` and ``raw_geometry`` holds the bounded, escaped ASCII text
      of what the source said (``bounded_raw_text``), for annotations whose geometry
      could not become a ``PixelBox``. Such annotations must be ``REJECTED`` or
      ``EXCLUDED`` with a ``reason``; they are recorded, never dropped.

    Status rules: ``MAPPED`` needs ``target_class`` and a box and has no reason;
    ``UNMAPPED`` needs a box and has neither reason nor target (it is not a finding);
    ``REVIEW``/``REJECTED``/``EXCLUDED`` need a ``reason`` and have no target. ``reason`` is
    the primary reason only; the full list lives in the findings. ``flags`` are
    source-provided markers (for example ``difficult``, ``truncated``, ``iscrowd``),
    kept sorted and unique.
    """

    record_id: str
    source_file: str
    source_locator: str
    source_class: str
    status: AnnotationStatus
    box: Optional[PixelBox] = None
    raw_geometry: Optional[str] = None
    reason: Optional[ReasonCode] = None
    target_class: Optional[TargetClass] = None
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_sha256(self.record_id, "record_id")
        validate_relative_posix_path(self.source_file)
        validate_source_locator(self.source_locator)
        validate_source_class(self.source_class)
        if not isinstance(self.status, AnnotationStatus):
            raise TypeError("status must be an AnnotationStatus")
        if self.reason is not None and not isinstance(self.reason, ReasonCode):
            raise TypeError("reason must be a ReasonCode or None")
        if self.target_class is not None and not isinstance(self.target_class, TargetClass):
            raise TypeError("target_class must be a TargetClass or None")

        if self.box is not None:
            if not isinstance(self.box, PixelBox):
                raise TypeError("box must be a PixelBox or None")
            if self.raw_geometry is not None:
                raise ValueError("raw_geometry is only for annotations without a valid box")
        else:
            if not isinstance(self.raw_geometry, str) or self.raw_geometry == "":
                raise ValueError("an annotation without a valid box needs non-empty raw_geometry")
            if len(self.raw_geometry) > MAX_RAW_GEOMETRY_CHARS or not all(
                32 <= ord(char) < 127 for char in self.raw_geometry
            ):
                raise ValueError("raw_geometry must be bounded printable ASCII (use bounded_raw_text)")
            if self.status not in (AnnotationStatus.REJECTED, AnnotationStatus.EXCLUDED):
                raise ValueError("an annotation without a valid box must be REJECTED or EXCLUDED")

        if self.status is AnnotationStatus.MAPPED:
            if self.target_class is None:
                raise ValueError("a MAPPED annotation needs a target_class")
            if self.reason is not None:
                raise ValueError("a MAPPED annotation has no reason")
        else:
            if self.target_class is not None:
                raise ValueError("only a MAPPED annotation may have a target_class")
        if self.status is AnnotationStatus.UNMAPPED and self.reason is not None:
            raise ValueError("an UNMAPPED annotation is not a finding and has no reason")
        if self.status in FINDING_STATUSES and self.reason is None:
            raise ValueError(f"a {self.status.value} annotation needs a reason")

        if not isinstance(self.flags, tuple):
            raise TypeError("flags must be a tuple")
        if len(self.flags) > MAX_FLAGS:
            raise ValueError(f"at most {MAX_FLAGS} flags are allowed")
        for flag in self.flags:
            if not isinstance(flag, str) or not _FLAG.fullmatch(flag):
                raise ValueError(f"invalid flag: {flag!r}")
        if len(set(self.flags)) != len(self.flags):
            raise ValueError("flags contain duplicates")
        object.__setattr__(self, "flags", tuple(sorted(self.flags)))

    @property
    def annotation_id(self) -> str:
        return source_annotation_id(self.record_id, self.source_file, self.source_locator)


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
        validate_output_filename(self.output_name)
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
