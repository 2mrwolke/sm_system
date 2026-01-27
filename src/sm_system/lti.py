from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import scipy
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class LTI:
    """Linear time-invariant (continuous-time) transfer-function helpers.

    These helpers define *analog* (continuous-time, s-domain) transfer functions. Functions like
    `scipy.signal.lsim` and `scipy.signal.impulse` simulate the continuous-time system; if you apply
    the result to sampled signal you must provide an appropriate time vector (e.g. `T = np.arange(N)/sr`)
    and be mindful of units/scaling.

    For discrete-time filtering, convert the system with
    `TransferFunction.to_discrete(dt=1/sr, method=...)` and operate in the z-domain.
    """


    def __init__(self) -> None:
        # Defining the types of filters and their corresponding methods
        self.system: Dict[str, Callable[..., Any]] = {
            "Custom": self.custom,
            "Low-pass 1st": self.lp_1,
            "Low-pass 2nd": self.lp_2,
            "High-pass 1st": self.hp_1,
            "High-pass 2nd": self.hp_2,
            "Band-pass 2nd": self.bp_2,
            "Notch 2nd": self.notch_2,
            "All-pass 1st": self.ap_1,
            "All-pass 2nd": self.ap_2,
            "-- NONE --": self.none,
        }

    def _get_times(self, signal: NDArray, sr: float) -> NDArray:
        """Generate a series of time values for an input signal with sample rate *sr*."""
        return np.arange(0, signal.shape[-1]) / sr

    @staticmethod
    def to_callable(tf: Any, sr: float) -> Callable[[NDArray], NDArray]:
        """Return a callable performing time-domain simulation (lsim) at sample rate *sr*."""

        def convolve(inputs: NDArray) -> NDArray:
            times = np.arange(0, inputs.shape[-1]) / sr
            _, result, _ = scipy.signal.lsim(tf, U=inputs, T=times, interp=True)
            return result

        return convolve

    @staticmethod
    def _require(kwargs: Dict[str, Any], *names: str) -> Tuple[Any, ...]:
        missing = [name for name in names if name not in kwargs]
        if missing:
            raise ValueError(f"Missing parameter(s): {', '.join(missing)}")
        return tuple(kwargs[name] for name in names)

    @staticmethod
    def to_scipy_tf(func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator: transform the given system into a scipy TransferFunction."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            kw = dict(kwargs)  # avoid mutating caller kwargs

            is_callable = bool(kw.pop("is_callable", False))
            verbose = bool(kw.pop("verbose", False))
            sr = kw.pop("sr", None)

            system = func(*args, **kw)
            tf = scipy.signal.TransferFunction(system[0], system[1])

            if is_callable:
                if sr is None:
                    raise ValueError("Missing parameter: sr (required when is_callable=True)")
                return LTI.to_callable(tf, sr)

            if verbose:
                logger.info("LTI-instance returned: scipy.signal.TransferFunctionContinuous")

            return tf

        return wrapper

    @to_scipy_tf
    def custom(self, **kwargs: Any):
        zeros, poles, gain = self._require(kwargs, "zeros", "poles", "gain")
        system = scipy.signal.ZerosPolesGain(zeros, poles, gain)
        tf = system.to_tf()
        return [tf.num, tf.den]

    @to_scipy_tf
    def lp_1(self, **kwargs: Any):
        (wc,) = self._require(kwargs, "wc")
        return [[1], [1 / wc, 1]]

    @to_scipy_tf
    def lp_2(self, **kwargs: Any):
        wc, q = self._require(kwargs, "wc", "q")
        return [[wc * wc], [1, wc / q, wc * wc]]

    @to_scipy_tf
    def hp_1(self, **kwargs: Any):
        (wc,) = self._require(kwargs, "wc")
        return [[1, 0], [1, wc]]

    @to_scipy_tf
    def hp_2(self, **kwargs: Any):
        wc, q = self._require(kwargs, "wc", "q")
        return [[1, 0, 0], [1, wc / q, wc * wc]]

    @to_scipy_tf
    def bp_2(self, **kwargs: Any):
        wc, q = self._require(kwargs, "wc", "q")
        return [[wc / q, 0], [1, wc / q, wc * wc]]

    @to_scipy_tf
    def notch_2(self, **kwargs: Any):
        wc, q = self._require(kwargs, "wc", "q")
        return [[1, 0, wc * wc], [1, wc / q, wc * wc]]

    @to_scipy_tf
    def ap_1(self, **kwargs: Any):
        (wc,) = self._require(kwargs, "wc")
        return [[1, -wc], [1, wc]]

    @to_scipy_tf
    def ap_2(self, **kwargs: Any):
        wc, q = self._require(kwargs, "wc", "q")
        return [[1, -wc / q, wc * wc], [1, wc / q, wc * wc]]

    @to_scipy_tf
    def none(self, **kwargs: Any):
        return [[1], [1]]

    def to_FIR(self, system_tf: Any, tabs: int = 256, sr: int = 48_000) -> NDArray:
        system_tf = system_tf.to_discrete(dt=1 / sr)
        _, fir = scipy.signal.dimpulse(system_tf, n=tabs)
        return np.asarray(fir[0])[:, 0]

    def convolve(self, signal: NDArray, filter_name: str = "Low-pass 1st", **kwargs: Any) -> NDArray:
        try:
            tf = self.system[filter_name](**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Unknown filter_name {filter_name!r}. Available: {list(self.system.keys())}"
            ) from e

        t, yout = scipy.signal.impulse(tf)
        if isinstance(yout, (tuple, list)):
            h = np.asarray(yout[0]).ravel()
        else:
            h = np.asarray(yout).ravel()

        return np.convolve(h, np.asarray(signal), mode="full")

    def __str__(self) -> str:
        return str(self.system.keys())
