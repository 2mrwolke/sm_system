from __future__ import annotations

from math import pi
from typing import Optional, Self

import numpy as np
from numpy.typing import NDArray

from .utils import expspace, reduce_idx

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover

    def tqdm(iterable, *args, **kwargs):
        return iterable


from functools import lru_cache

PI = pi


class HarmonicProbeIterator:
    """
    Iterator that returns batches of frequency pairs.
    The frequency pairs are chosen to avoid interference
    of nonlinearities up to order 3.
    """

    def __init__(
        self,
        f_min: int = 20,
        f_max: int = 16_000,
        n_max: int = 36,
        samplerate: int = 96_000,
        batches: int = 1,
        resolution: int = 1000,
        signal_time: float = 0.01,
        transient_time: float = 0.0,
        sync_signal: Optional[NDArray[np.floating]] = None,
        iscyclic: bool = False,
        order: int = 3,
        seed: int = 42,
    ):
        """
        Initialize HarmonicProbeIterator

        Parameters:
        f_min (int):          Lowerbounds Probe-Frequencies
        f_max(int):           Upperbounds Probe-Frequencies
        n_max (int):          Defines the upperbound of: number of unique
                                                         frequency pairs
                                                         per batch.
        nyquist (int):        Defines the upperbound of: frequency.
        resolution (int):     Defines the rounding error of forbidden fractions
                              that define interference.
        batches (int):        Defines number of batches

        signal_time (float):  Signal length in seconds.
        batch_size (int):     The size of the batch to be returned.
        iscyclic (bool):      Defines whether the iterator cycles.
        seed (int):           Seed for numpy's random number generator.
        """

        self.n_max = n_max
        self.f_min = f_min
        self.f_max = f_max
        self.sample_rate = samplerate
        self.nyquist = int(samplerate / 2)
        self.resolution = resolution
        self.batches = batches
        self.signal_time = signal_time
        self.transient_time = transient_time
        self.sync_signal = sync_signal
        self.iscyclic = iscyclic
        self.order = order
        self.seed = seed
        self.index = 0
        self.forbidden_fractions: NDArray[np.floating] = np.array(
            [0, 1 / 2, 2 / 3, 1 / 3, 1 / 5, 1 / 4, 1, 2, 3 / 2, 3, 5, 4]
        )
        self.freq_pairs: NDArray[np.floating] = self.build_batches()
        self.wav_shape: Optional[tuple[int, ...]] = None

        if self.transient_time == -1:
            self.transient_time = self.signal_time

        if not iscyclic:
            self.__update_index = lambda: self.index + 1
        else:
            self.__update_index = lambda: np.mod(
                self.index + 1, len(self.freq_pairs)
            )

    @lru_cache(maxsize=None)
    def build_batches(self) -> NDArray[np.floating]:
        original_seed = self.seed
        batches = [
            self._create_freq_pairs(
                self.f_min,
                self.f_max,
                self.n_max,
                self.signal_time,
                self.forbidden_fractions,
            )
            for i in range(self.batches)
        ]

        min_size = np.min([b.shape[0] for b in batches])
        batches = np.array([b[:min_size] for b in batches])
        self.seed = original_seed
        return batches

    def _create_freq_pairs(
        self,
        freq_min: int,
        freq_max: int,
        n: int,
        signal_time: float,
        fractions: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Calculate random frequency pairs.
        s.t.: Every frequency_pair is unique.
              Every single freqency appears exactly two times.
              Interference up to power: 3 is avoided.

        Parameters:
        freq_min (int):       Lowerbound for freq_pairs.
        freq_max (int):       Upperbound for freq_pairs. <- this is a lie!
        n (int):              Upperbound for how many freq_pairs are returned.
                              For higher n, the less logarithmic is the freq.
                              distribution.
        signal_time (float):  Ensures that all frequencies are exactly
                              on bins of fft with signal_time - length.
        fractions (list):     Defines forbidden fractions of freqency pairs,
                              that would cause interferrence.

        Returns:
        numpy.ndarray: Array of frequency pairs.
        """

        np.random.seed(self.seed)
        self.seed += 7

        freq = expspace(min=freq_min, max=freq_max - 1, n=n)

        freq = np.round(freq * signal_time) / signal_time
        freq = np.where(freq < freq_max, freq, 0)
        freq = np.where(freq > freq_min, freq, 0)
        freq = freq[freq != 0]
        freq = np.unique(freq)

        # NOTE (intentional trade-off): 
        # Vectorized 3D meshgrid => O(N^3) intermediate storage where N = len(freq). We allocate
        # multiple N×N×N arrays (e.g., f1/f2/f3 plus boolean masks), so peak RAM can grow to several
        # × (8*N^3) bytes (float64) + mask overhead. This will exceed RAM for embedded systems. 
        #
        # We keep this approach because it reduces branching and keeps the constraint logic fully
        # vectorized, which makes the path faster to iterate on during development. 
        #
        # For production measurement setups with a fixed frequency plan: precompute the resulting
        # frequency triplets offline and load them as a static artifact (rather than generating them
        # on-device / at runtime).
        freq_grid = np.meshgrid(freq, freq, freq)
        x, y, z = freq_grid
        grid = np.stack([x, y, z], axis=-1)
        mask = np.array([x / y, x / z, y / z])
        mask = np.round(mask * self.resolution) / self.resolution
        fractions = np.round(fractions * self.resolution) / self.resolution
        mask = ~np.isin(mask, fractions)
        mask = mask[0] * mask[1] * mask[2]
        freq_pairs = list([])
        for i, f in tqdm(enumerate(freq)):
            try:
                tmp = grid[0][mask[0]]
                rdm_idx = np.random.choice(len(tmp), 1, replace=False)

                a, b, c = np.sort(tmp[rdm_idx])[0]
                freq_pairs.append(np.array([[a, c], [a, b], [b, c]]))
                grid = reduce_idx(grid, indices=[0] * 3, dims=[0, 1, 2])
                mask = reduce_idx(mask, indices=[0] * 3, dims=[0, 1, 2])
                m = ~np.isin(grid, [b, c])
                mask = mask * m[:, :, :, 0] * m[:, :, :, 1] * m[:, :, :, 2]
            except BaseException:
                grid = reduce_idx(grid, indices=[0] * 3, dims=[0, 1, 2])
                mask = reduce_idx(mask, indices=[0] * 3, dims=[0, 1, 2])

        freq_pairs = np.array(freq_pairs).reshape((-1, 2))
        return freq_pairs

    def __iter__(self) -> Self:
        """
        Returns the iterator object (self).
        """
        return self

    def __next__(self) -> NDArray[np.floating]:
        """
        Fetch the next items in frequency pairs iteratively.

        Returns:
        numpy.ndarray: Next batch of frequency pairs.

        Raises:
        StopIteration: When all batches have been accessed,
                       and if iterator is not cyclic.
        """
        if self.index >= len(self.freq_pairs):
            raise StopIteration

        result = self.freq_pairs[self.index]
        self.index = self.__update_index()
        return result

    def __call__(self) -> NDArray[np.floating]:
        """
        Returns:
        numpy.ndarray: Sum of 2 cosine signals for every pair in <freq_pairs>.
        """
        w_pairs = 2 * PI * self.freq_pairs
        time_t = (
            np.arange(0, self.transient_time, 1 / self.sample_rate)[::-1]
            + 1 / self.sample_rate
        )
        time_s = np.arange(0, self.signal_time, 1 / self.sample_rate)
        time = np.concatenate([-time_t, time_s])
        self.time = time
        args = np.einsum("...bp, t -> ...bpt", w_pairs, time)
        signals = np.cos(args)
        signals = np.sum(signals, axis=-2)
        return signals
