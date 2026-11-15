"""Pipeline configuration: strict loading, validation and hashing.

* Unknown keys, wrong types, out-of-range values, duplicate YAML keys, and empty or
  mistyped nested sections are errors. Missing keys take defaults.
* YAML is parsed with a ``SafeLoader`` subclass (no arbitrary object construction).
* Numeric values are canonicalised (floats stay floats, integers stay integers, -0.0
  becomes 0.0), so equivalent spellings such as ``16`` and ``16.0`` are equal and hash
  identically.
* ``config_hash`` covers build-affecting settings only. Operational settings (the
  ``ledger`` section, e.g. ``backup_dir``) are excluded from build identity.
* All configuration objects are frozen and hashable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml
from yaml.constructor import ConstructorError

PROJECTIONS = ("letterbox", "stretch")
_RATIO_TOLERANCE = 1e-9

#: Sections excluded from the config hash because they do not affect build output.
OPERATIONAL_SECTIONS = ("ledger",)


class ConfigError(ValueError):
    """Raised when a configuration file or value is invalid."""


# --------------------------------------------------------------------------- #
# Value helpers
# --------------------------------------------------------------------------- #
def _int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    return value


def _float(name: str, value: Any) -> float:
    """A finite real number, returned as a float with -0.0 normalised to 0.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ConfigError(f"{name} is too large: {value!r}") from exc
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be a finite number, got {value!r}")
    return number + 0.0


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DedupConfig:
    box_match_iou: float = 0.95

    def __post_init__(self) -> None:
        iou = _float("dedup.box_match_iou", self.box_match_iou)
        if not 0 < iou <= 1:
            raise ConfigError("dedup.box_match_iou must be in (0, 1]")
        _set(self, "box_match_iou", iou)


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        total = 0.0
        for name in ("train", "val", "test"):
            ratio = _float(f"split.{name}", getattr(self, name))
            if ratio <= 0:
                raise ConfigError(f"split.{name} must be > 0")
            _set(self, name, ratio)
            total += ratio
        if abs(total - 1.0) > _RATIO_TOLERANCE:
            raise ConfigError(f"split ratios must sum to 1, got {total}")
        _int("split.seed", self.seed)


@dataclass(frozen=True)
class LedgerConfig:
    """Operational settings; not part of build identity."""

    backup_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.backup_dir is not None and not isinstance(self.backup_dir, Path):
            raise ConfigError("ledger.backup_dir must be a path or null")


@dataclass(frozen=True)
class PipelineConfig:
    train_imgsz: int = 640
    projection: str = "letterbox"
    min_projected_side_px: float = 16.0
    oob_tolerance_px: float = 2.0
    dhash_max_distance: int = 6
    max_group_fraction: float = 0.05
    dedup: DedupConfig = field(default_factory=DedupConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)

    def __post_init__(self) -> None:
        if _int("train_imgsz", self.train_imgsz) <= 0:
            raise ConfigError("train_imgsz must be > 0")
        if not isinstance(self.projection, str) or self.projection not in PROJECTIONS:
            raise ConfigError(f"projection must be one of {PROJECTIONS}, got {self.projection!r}")

        side = _float("min_projected_side_px", self.min_projected_side_px)
        if side <= 0:
            raise ConfigError("min_projected_side_px must be > 0")
        _set(self, "min_projected_side_px", side)

        tolerance = _float("oob_tolerance_px", self.oob_tolerance_px)
        if tolerance < 0:
            raise ConfigError("oob_tolerance_px must be >= 0")
        _set(self, "oob_tolerance_px", tolerance)

        distance = _int("dhash_max_distance", self.dhash_max_distance)
        if not 0 <= distance <= 64:
            raise ConfigError("dhash_max_distance must be between 0 and 64")

        fraction = _float("max_group_fraction", self.max_group_fraction)
        if not 0 < fraction <= 1:
            raise ConfigError("max_group_fraction must be in (0, 1]")
        _set(self, "max_group_fraction", fraction)

        for name, expected in (("dedup", DedupConfig), ("split", SplitConfig), ("ledger", LedgerConfig)):
            if not isinstance(getattr(self, name), expected):
                raise ConfigError(f"{name} must be a {expected.__name__}")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _check_keys(section: str, data: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
    unknown = sorted(set(map(str, data)) - set(allowed))
    if unknown:
        where = section or "top level"
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(unknown)}")


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """A nested section: absent means defaults; present must be a non-empty mapping."""
    if key not in data:
        return {}
    value = data[key]
    if not isinstance(value, Mapping) or len(value) == 0:
        raise ConfigError(f"{key} must be a non-empty mapping, got {value!r}")
    return value


def parse_config(data: Mapping[str, Any]) -> PipelineConfig:
    """Build a validated PipelineConfig from a plain mapping."""
    if not isinstance(data, Mapping):
        raise ConfigError("configuration must be a mapping")
    _check_keys(
        "",
        data,
        (
            "train_imgsz",
            "projection",
            "min_projected_side_px",
            "oob_tolerance_px",
            "dhash_max_distance",
            "max_group_fraction",
            "dedup",
            "split",
            "ledger",
        ),
    )
    defaults = PipelineConfig()

    dedup_data = _section(data, "dedup")
    _check_keys("dedup", dedup_data, ("box_match_iou",))
    split_data = _section(data, "split")
    _check_keys("split", split_data, ("train", "val", "test", "seed"))
    ledger_data = _section(data, "ledger")
    _check_keys("ledger", ledger_data, ("backup_dir",))

    backup_raw = ledger_data.get("backup_dir")
    if backup_raw is None:
        backup: Optional[Path] = None
    elif isinstance(backup_raw, str) and backup_raw.strip() and "\x00" not in backup_raw:
        backup = Path(backup_raw)
    else:
        raise ConfigError("ledger.backup_dir must be a non-empty string or null")

    return PipelineConfig(
        train_imgsz=data.get("train_imgsz", defaults.train_imgsz),
        projection=data.get("projection", defaults.projection),
        min_projected_side_px=data.get("min_projected_side_px", defaults.min_projected_side_px),
        oob_tolerance_px=data.get("oob_tolerance_px", defaults.oob_tolerance_px),
        dhash_max_distance=data.get("dhash_max_distance", defaults.dhash_max_distance),
        max_group_fraction=data.get("max_group_fraction", defaults.max_group_fraction),
        dedup=DedupConfig(box_match_iou=dedup_data.get("box_match_iou", defaults.dedup.box_match_iou)),
        split=SplitConfig(
            train=split_data.get("train", defaults.split.train),
            val=split_data.get("val", defaults.split.val),
            test=split_data.get("test", defaults.split.test),
            seed=split_data.get("seed", defaults.split.seed),
        ),
        ledger=LedgerConfig(backup_dir=backup),
    )


def default_config() -> PipelineConfig:
    return PipelineConfig()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last one."""


def _construct_unique_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False):
    loader.flatten_mapping(node)
    seen: set[Any] = set()
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping", node.start_mark, "found an unhashable key", key_node.start_mark
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)

# YAML 1.1 (which PyYAML implements) silently reads some plain scalars as different
# numbers: ``010`` and ``0640`` are octal (8 and 416), and ``6:40`` is sexagesimal (400).
# Such forms are rejected; write plain decimal numbers instead.
_LEGACY_OCTAL = re.compile(r"[-+]?0[0-7_]+")


def _reject_ambiguous_number(loader: _StrictLoader, node: yaml.ScalarNode, kind: str) -> None:
    text = str(node.value)
    reason = None
    if ":" in text:
        reason = "sexagesimal (base-60)"
    elif kind == "integer" and _LEGACY_OCTAL.fullmatch(text):
        reason = "octal (leading zero)"
    if reason:
        raise ConstructorError(
            "while constructing a number",
            node.start_mark,
            f"ambiguous legacy YAML 1.1 {kind} {text!r} would be read as {reason}; "
            "write a plain decimal number instead",
            node.start_mark,
        )


def _construct_safe_int(loader: _StrictLoader, node: yaml.ScalarNode) -> int:
    _reject_ambiguous_number(loader, node, "integer")
    return yaml.SafeLoader.construct_yaml_int(loader, node)


def _construct_safe_float(loader: _StrictLoader, node: yaml.ScalarNode) -> float:
    _reject_ambiguous_number(loader, node, "float")
    return yaml.SafeLoader.construct_yaml_float(loader, node)


_StrictLoader.add_constructor("tag:yaml.org,2002:int", _construct_safe_int)
_StrictLoader.add_constructor("tag:yaml.org,2002:float", _construct_safe_float)


def load_yaml_strict(path: Union[str, Path], *, what: str = "configuration file") -> Any:
    """Parse a YAML file with the project's strict, safe loader and return the raw data.

    Strictness: no arbitrary object construction (``SafeLoader`` subclass), duplicate
    mapping keys are rejected, and ambiguous legacy YAML 1.1 numbers (leading-zero octal
    such as ``010`` and sexagesimal such as ``6:40``) are rejected. An empty file returns
    ``None``. A missing file, unreadable or undecodable file, or invalid YAML raises
    ``ConfigError``. ``what`` (default ``"configuration file"``) names the kind of file in
    exactly two messages, ``"<what> not found: <path>"`` and ``"cannot read <what> <path>:
    <reason>"``. The ``"invalid YAML in <path>: <reason>"`` message does not use it. With
    the default, all three messages are exactly those ``load_config`` has always produced.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"{what} not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            # _StrictLoader is a SafeLoader subclass: no arbitrary object construction.
            return yaml.load(handle, Loader=_StrictLoader)  # noqa: S506
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read {what} {config_path}: {exc}") from exc


def load_config(path: Union[str, Path]) -> PipelineConfig:
    """Load and validate a YAML configuration file.

    A missing or unreadable file is a ``ConfigError``. An empty file yields the defaults.
    """
    data = load_yaml_strict(path)
    if data is None:
        data = {}
    return parse_config(data)


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def config_hash(config: PipelineConfig) -> str:
    """SHA-256 of the canonical JSON form of the build-affecting configuration."""
    payload = asdict(config)
    for section in OPERATIONAL_SECTIONS:
        payload.pop(section, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
