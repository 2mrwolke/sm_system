from __future__ import annotations


from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover

    def tqdm(iterable, *args, **kwargs):
        return iterable


from .config import Configuration
from .lti import LTI
from .probe import HarmonicProbeIterator


class Simulation:

    cfg: Configuration
    lti: LTI
    harmonic_probes: HarmonicProbeIterator
    sm: Optional[Callable[[NDArray[np.floating]], NDArray[np.floating]]]

    def __init__(
        self,
        path2config: Optional[str] = None,
        rm_dc: bool = True,
        rm_transients: bool = True,
    ) -> None:
        self.cfg = Configuration()
        if path2config is not None:
            self.cfg.load(path2config)
        self.lti = LTI()
        self.build_harmonic_probes()
        self.rm_dc = rm_dc
        self.rm_transients = rm_transients

    def build_harmonic_probes(self) -> None:
        self.harmonic_probes = HarmonicProbeIterator(
            f_min=self.cfg.probes_fbase,
            f_max=self.cfg.probes_fvoice,
            n_max=self.cfg.probes_n,
            samplerate=self.cfg.samplerate,
            batches=self.cfg.probes_batches,
            resolution=self.cfg.probes_resolution,
            signal_time=self.cfg.probes_signaltime,
            transient_time=self.cfg.probes_transient_time,
            iscyclic=False,
            order=self.cfg.max_order,
            seed=self.cfg.seed,
        )

    def set_sm(
        self, sm: Callable[[NDArray[np.floating]], NDArray[np.floating]]
    ) -> None:
        if not callable(sm):
            raise TypeError(
                "Simulation.set_sm(sm): 'sm' must be callable."
            )
        self.sm = sm

    def run(self) -> NDArray[np.floating]:
        if self.sm is None:
            raise RuntimeError(
                "Simulation.run(): Call Simulation.set_sm(sm) before run()."
            )
        sm = self.sm
        probes = self.harmonic_probes()
        n_batches, n_pairs, n_samps = probes.shape[:3]
        result: NDArray[np.floating] = np.empty((n_batches, n_pairs, n_samps))

        for batch_idx, batch in enumerate(probes):
            for pair_idx, signal in tqdm(enumerate(batch)):
                result[batch_idx, pair_idx] = sm(signal)
        if self.rm_transients:
            result = result[
                :, :, -int(self.cfg.probes_signaltime * self.cfg.samplerate) :
            ]
        if self.rm_dc:
            mean = np.mean(result, axis=-1, keepdims=True)
            result = result - mean
        return result
