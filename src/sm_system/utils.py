import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Sequence, Union, Tuple


def change_nan_inf(
    x: ArrayLike,
    nan: Optional[float] = None,
    inf: Optional[float] = None,
    dtype: Union[str, np.dtype] = "d",
) -> NDArray:
    """Replace NaN and +/-Inf values in *x*.

    Parameters
    ----------
    x:
        Input array.
    nan:
        Replacement value for NaNs. Defaults to the smallest finite value
        representable by *dtype*.
    inf:
        Replacement value for +/-Inf. Defaults to the largest finite value
        representable by *dtype*.
    dtype:
        Floating dtype used for finfo defaults and output casting.

    Returns
    -------
    numpy.ndarray
        Array of dtype *dtype* with NaN/Inf replaced.
    """
    info = np.finfo(dtype)
    if nan is None:
        nan = info.min
    if inf is None:
        inf = info.max

    x_arr = np.asarray(x, dtype=dtype)
    x_arr = np.where(np.isnan(x_arr), nan, x_arr)
    x_arr = np.where(np.isinf(x_arr), inf, x_arr)
    return x_arr


def to_level(x: ArrayLike, norm: Optional[float] = None) -> NDArray:
    """Convert a linear magnitude array to dB (20*log10).

    If *norm* is not provided, uses the first dimension length of *x*.
    """
    x_arr = np.asarray(x)
    if norm is None:
        norm = int(x_arr.shape[0])
    return 20 * np.log10(x_arr / norm)


def from_level(level: ArrayLike, norm: Optional[float] = None) -> NDArray:
    """Convert a dB array (20*log10) back to linear magnitude.

    If *norm* is not provided, uses the first dimension length of *level*.
    """
    lvl = np.asarray(level)
    if norm is None:
        norm = int(lvl.shape[0])
    x = lvl * np.log(10) / 20
    return np.exp(x) * norm


def expspace(min_val: float, max_val: float, n: int = 100, base: float = 10) -> NDArray:
    """Logarithmically-spaced values from *min_val* to *max_val* (inclusive)."""
    min_log = np.log(min_val) / np.log(base)
    max_log = np.log(max_val) / np.log(base)
    rnge = np.linspace(min_log, max_log, n)
    return np.power(base, rnge)


def reduce_idx(
    x: NDArray,
    indices: Sequence[int],
    dims: Optional[Sequence[int]] = None,
) -> NDArray:
    """Remove indices along selected dimensions.

    Parameters
    ----------
    x:
        Array that gets reduced.
    indices:
        One index per dimension to be reduced.
    dims:
        Dimensions to reduce. Defaults to [0].

    Returns
    -------
    numpy.ndarray
        Reduced array.
    """
    if dims is None:
        dims = [0]

    if len(indices) != len(dims):
        raise ValueError("Length of indices and dims must be the same")

    x_out = x
    for idx, dim in zip(indices, dims):
        x_out = np.delete(x_out, idx, dim)

    return x_out


def permute_matrix(d: int, v: int = 2) -> NDArray:
    n = v**d

    def _fun(i: int) -> NDArray:
        x = np.arange(n)
        arg = n / (v**i)
        x = np.mod(x, arg)
        x = np.sort(x)
        x = np.mod(x, v)
        return x

    return np.array([_fun(i) for i in range(d)]).T


def save2wave(
    x: ArrayLike,
    title: str = "new_wav",
    normalize: bool = True,
    sr: int = 96_000,
    peak: float = -0.5,
    dtype: np.dtype = np.float32,
) -> NDArray:
    """Normalize *x* and save it to WAV format.

    Parameters
    ----------
    x:
        Input signal.
    title:
        Output file name without extension.
    normalize:
        If True, normalize to 0 dBFS and scale to the given *peak* (dB).
    sr:
        Sample rate.
    peak:
        Target peak level in dB.
    dtype:
        Target dtype for file writing.

    Returns
    -------
    numpy.ndarray
        Signal cast to *dtype* (after optional normalization/scaling).
    """
    allowed_types = [np.float16, np.float32, np.uint8, np.int16, np.int32]
    if dtype not in allowed_types:
        raise TypeError(
            f"dtype must be one of {allowed_types}, got {dtype!r}"
        )

    x_arr = np.asarray(x)

    if normalize:
        x_arr = x_arr.astype(np.float128)
        norm = np.max(np.abs(x_arr))
        if norm != 0:
            x_arr /= norm
            x_arr *= from_level(np.array([peak]))

        # Only apply integer full-scale mapping for integer dtypes.
        try:
            mn, mx = np.iinfo(dtype).min, np.iinfo(dtype).max
            ctr = 0.5 * (mn + mx) + 0.5
            hlf = 0.5 * (mx - mn - 1)
            x_arr = x_arr * hlf + ctr
        except (TypeError, ValueError):
            # Float dtypes: keep as normalized float in [-1, 1] (or scaled by peak).
            pass

    x_arr = x_arr.astype(dtype)
    scipy.io.wavfile.write(title + ".wav", sr, x_arr)
    return x_arr


def rms(x: ArrayLike, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> NDArray:
    """Root-mean-square along *axis*."""
    x_arr = np.asarray(x)
    return np.sqrt(np.mean(np.square(x_arr), axis=axis))


def snr_db(
    s: ArrayLike,
    n: ArrayLike,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> NDArray:
    """Signal-to-noise ratio in dB (20*log10(rms(s)/rms(n)))."""
    snr = rms(s, axis) / rms(n, axis)
    return 20 * np.log10(snr)


def noise_like(x: ArrayLike, db: float) -> NDArray:
    """Generate zero-mean Gaussian noise with power level in dB.

    Parameters
    ----------
    x:
        Array whose shape will be matched.
    db:
        Noise power level in decibel.
    """
    var = 10 ** (db / 10)  # power (linear)
    std = np.sqrt(var)  # RMS (linear)
    return np.random.normal(size=np.asarray(x).shape, scale=std)
