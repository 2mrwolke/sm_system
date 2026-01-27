from __future__ import annotations

import pprint
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np


def _now_str(tz: str = "Europe/Berlin") -> str:
    """Return a stable, human-readable timestamp string.
    """
    return datetime.now(ZoneInfo(tz)).strftime("%Y/%m/%d_%H:%M:%S")


def _coerce_yaml_value(v: Any) -> Any:
    """Convert numpy-ish values to plain Python types suitable for YAML."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v


@dataclass(kw_only=True, slots=True)
class Configuration:

    """
    Experiment configuration with YAML save/load
    and derived parameter recomputation.
    """


    # -----------------
    # Public / metadata
    # -----------------
    comment: str = field(default_factory=str)
    tags: list[str] = field(default_factory=list)

    # Internal bookkeeping: not part of the user-editable data payload.
    _last_accessed: str = field(default_factory=_now_str)
    path: str = field(default_factory=str)

    # -----------------
    # Probe / FFT setup
    # -----------------
    filter_tabs: int = 1024
    probes_n: int = 68
    probes_fbase: int = 200
    probes_fvoice: int = 16_000
    probes_batches: int = 8
    probes_resolution: int = 1000
    probes_signaltime: float = 0.01
    probes_transient_time: float = 0.01
    probes_fft_n: int = field(init=False)
    probes_fft_delta: float = field(init=False)

    # -------------------
    # Simulation settings
    # -------------------
    seed: int = 42
    samplerate: int = 96_000
    active_sm_paths: list[int] = field(default_factory=list)
    max_order: int = 3
    dut: str = field(default_factory=str)
    snr: float = field(default_factory=float)
    hardware_latency: int = field(default_factory=int)
    delay: float = 0.0

    # -------
    # Filters
    # -------
    filter1_type: str = "-- NONE --"
    filter1_fc: int = 440
    filter1_q: float = 0.707
    filter1_zeros: list[float] = field(default_factory=list)
    filter1_poles: list[float] = field(default_factory=list)
    filter1_gain: float = 1.0
    filter1_tabs: int = 128

    filter21_type: str = "-- NONE --"
    filter21_fc: int = 440
    filter21_q: float = 0.707
    filter21_lag: float = 45.0
    filter21_zeros: list[float] = field(default_factory=list)
    filter21_poles: list[float] = field(default_factory=list)
    filter21_gain: float = 1.0
    filter21_tabs: int = 128

    filter22_type: str = "-- NONE --"
    filter22_fc: int = 440
    filter22_q: float = 0.707
    filter22_lag: float = 45.0
    filter22_zeros: list[float] = field(default_factory=list)
    filter22_poles: list[float] = field(default_factory=list)
    filter22_gain: float = 1.0
    filter22_tabs: int = 128

    filter31_type: str = "-- NONE --"
    filter31_fc: int = 440
    filter31_q: float = 0.707
    filter31_lag: float = 45.0
    filter31_zeros: list[float] = field(default_factory=list)
    filter31_poles: list[float] = field(default_factory=list)
    filter31_gain: float = 1.0
    filter31_tabs: int = 128

    filter32_type: str = "-- NONE --"
    filter32_fc: int = 440
    filter32_q: float = 0.707
    filter32_lag: float = 45.0
    filter32_zeros: list[float] = field(default_factory=list)
    filter32_poles: list[float] = field(default_factory=list)
    filter32_gain: float = 1.0
    filter32_tabs: int = 128

    # -----------
    # Constraints
    # -----------
    constraint_amp_s2: str = "ONE"
    constraint_amp_s3: str = "ONE"
    constraint_phase_s2: str = "ONE"
    constraint_phase_s3: str = "ONE"

    # ----------------
    # Internal fields
    # ----------------
    _keys: list[str] = field(init=False, repr=False)

    # YAML wrapper schema version (integer, for migration / compatibility).
    YAML_SCHEMA_VERSION: int = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        # Public (user-configurable) keys that are serialized to/from YAML.
        # Excludes derived values and internal bookkeeping.
        self._keys = [
            f.name
            for f in fields(self)
            if f.init
            and f.name not in {"path", "_keys"}
            and not f.name.startswith("_")
        ]
        self._recalc()

    # -----------------
    # YAML serialization
    # -----------------
    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable dict of public configuration fields."""

        return {k: _coerce_yaml_value(getattr(self, k)) for k in self._keys}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, strict: bool = True) -> "Configuration":
        """Build a Configuration from a mapping (typically parsed from YAML).

        Parameters
        ----------
        data:
            Mapping with configuration keys.
        strict:
            If True (default), unknown keys raise ValueError to surface typos.
        """

        init_keys = {
            f.name
            for f in fields(cls)
            if f.init
            and f.name not in {"path", "_keys"}
            and not f.name.startswith("_")
        }
        unknown = sorted(set(data.keys()) - init_keys)
        if strict and unknown:
            raise ValueError(
                "Unknown configuration key(s): "
                + ", ".join(unknown)
                + ". Remove them or pass strict=False."
            )

        kwargs = {k: data[k] for k in init_keys if k in data}
        cfg = cls(**kwargs)
        cfg._set_time()
        return cfg

    # -----
    # I/O
    # -----
    def save(self, path2file: str | Path = "config.yaml") -> Path:
        """Save configuration as human-readable YAML.

        The YAML file uses a wrapper object:

        - schema (string identifier)
        - schema_version (integer)
        - saved_at (timestamp)
        - data (the actual user-configurable config mapping)
        """

        path = Path(path2file).expanduser().resolve()
        if path.suffix.lower() not in {".yml", ".yaml"}:
            path = path.with_suffix(".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)

        self.path = str(path)
        self._set_time()

        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "YAML support requires PyYAML. Install it with: pip install pyyaml"
            ) from e

        payload = {
            "schema": "sm_system.Configuration",
            "schema_version": int(self.YAML_SCHEMA_VERSION),
            "saved_at": self._last_accessed,
            "data": self.to_dict(),
        }

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                payload,
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

        return path

    def load(self, path: str | Path, *, strict: bool = True) -> None:
        """Load configuration from YAML into this instance (in-place)."""

        path = Path(path).expanduser().resolve()

        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "YAML support requires PyYAML. Install it with: pip install pyyaml"
            ) from e

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("Invalid YAML config: top-level object must be a mapping/dict.")

        # Accept either a wrapped structure (recommended) or a plain mapping.
        if "data" in raw and isinstance(raw.get("data"), dict):
            data = raw["data"]
            schema_version = raw.get("schema_version")
            if schema_version is not None:
                try:
                    schema_version_i = int(schema_version)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid schema_version {schema_version!r}: expected an integer."
                    )
                if schema_version_i > int(self.YAML_SCHEMA_VERSION):
                    raise ValueError(
                        "Unsupported config schema_version "
                        f"{schema_version_i}; this code supports up to {int(self.YAML_SCHEMA_VERSION)}."
                    )
        else:
            # Plain mapping (legacy / permissive)
            data = raw

        loaded = Configuration.from_dict(data, strict=strict)
        for k in self.__slots__:
            setattr(self, k, getattr(loaded, k))

        self.path = str(path)
        self._set_time()
        self._recalc()

    # -----------------
    # Public convenience
    # -----------------
    def update(self, key: str, value: Any) -> None:
        setattr(self, key, value)
        self._set_time()
        self._recalc()

    # ---------
    # Internals
    # ---------
    def _recalc(self) -> None:
        n_bins = np.log(self.probes_signaltime * self.samplerate) / np.log(2)
        fft_exp = int(np.ceil(n_bins))
        self.probes_fft_n = 1 << fft_exp

        # Legacy (non-idempotent) formulation kept for reproducibility:
        # self.probes_fft_n = int(2 ** np.round(n_bins + 0.5))

        self.probes_signaltime = self.probes_fft_n / self.samplerate
        self.probes_fft_delta = self.samplerate / self.probes_fft_n

        transient_samples = self.probes_transient_time * self.samplerate
        transient_samples = np.round(transient_samples)
        self.probes_transient_time = float(transient_samples / self.samplerate)

    def _set_time(self) -> None:
        self._last_accessed = _now_str()

    def __str__(self) -> str:
        # Pretty-print public, YAML-backed fields by default.
        return pprint.pformat(self.to_dict())
