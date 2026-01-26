"""Top-level package for *sm-system*.

The public surface is explicitly defined (no wildcard re-exports) to keep imports
stable, readable, and free of incidental symbols such as ``np`` / ``scipy``.

Plotting utilities are intentionally *not* imported by default. Install the
optional extra and import from :mod:`sm_system.plots` when needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Core configuration / design helpers
from .config import Configuration
from .filter_design import FIR_Filter_Freq_Sampling, compute_minimum_phase

# Linear algebra / identification helpers
from .lse import (
    build_coef_matrix_order2,
    build_coef_matrix_order3,
    calc_signed_indices,
    check_full_rank,
    clc_lhs_order_2,
    clc_lhs_order_3,
    ensure_fullrank,
    extract_levels,
    least_squares,
)

# Systems / simulation
from .lti import LTI
from .probe import HarmonicProbeIterator
from .sim import Simulation
from .sm import SM_System

# Utilities
from .utils import (
    change_nan_inf,
    expspace,
    from_level,
    noise_like,
    permute_matrix,
    reduce_idx,
    rms,
    save2wave,
    snr_db,
    to_level,
)

# Version is derived from installed package metadata (single source of truth).
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("sm-system")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"


__all__ = [
    # Version
    "__version__",
    # Core
    "Configuration",
    "compute_minimum_phase",
    "FIR_Filter_Freq_Sampling",
    # LSE
    "least_squares",
    "build_coef_matrix_order2",
    "build_coef_matrix_order3",
    "clc_lhs_order_2",
    "clc_lhs_order_3",
    "check_full_rank",
    "ensure_fullrank",
    "calc_signed_indices",
    "extract_levels",
    # Systems
    "LTI",
    "SM_System",
    "HarmonicProbeIterator",
    "Simulation",
    # Utils
    "change_nan_inf",
    "to_level",
    "from_level",
    "expspace",
    "reduce_idx",
    "permute_matrix",
    "save2wave",
    "rms",
    "snr_db",
    "noise_like",
]


if TYPE_CHECKING:  # pragma: no cover
    # For type checkers only; runtime import is intentionally avoided.
    from .plots import plot_bode_measurement as plot_bode_measurement


def __getattr__(name: str) -> Any:  # pragma: no cover
    """Lazy optional imports.

    This keeps the default import lightweight while still allowing convenient
    access to optional functionality *when installed*.
    """

    if name == "plot_bode_measurement":
        # Import-on-first-use to avoid importing matplotlib unless requested.
        from .plots import plot_bode_measurement

        globals()[name] = plot_bode_measurement
        return plot_bode_measurement

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(__all__) + ["plot_bode_measurement"])
