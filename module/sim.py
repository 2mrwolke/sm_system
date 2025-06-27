import numpy as np
from tqdm import tqdm
from module.config import Configuration
from module.probe import HarmonicProbeIterator
from module.lti import LTI

class Simulation():
  
  def __init__(self, path2config: str = None, rm_dc=True, rm_transients=True):
    self.cfg = Configuration()
    if not path2config == None:
      self.cfg.load(path2config)
    self.lti = LTI()
    self.build_harmonic_probes()
    self.rm_dc = rm_dc
    self.rm_transients = rm_transients

  def build_harmonic_probes(self):
    self.harmonic_probes = HarmonicProbeIterator(
        f_min=self.cfg.probes_fbase,
        f_max=self.cfg.probes_fvoice,
        n_max = self.cfg.probes_n,
        samplerate=self.cfg.samplerate,
        batches=self.cfg.probes_batches,
        resolution=self.cfg.probes_resolution,
        signal_time=self.cfg.probes_signaltime,
        transient_time=self.cfg.probes_transient_time,
        iscyclic=False,
        order=self.cfg.max_order,
        seed=self.cfg.seed)
    
  def set_sm(self, sm):
    self.sm = sm

  def run(self):
    sm = self.sm
    probes = self.harmonic_probes()
    b, n, s = probes.shape[:3]
    result = np.empty((b, n, s))
    for b, batch in enumerate(probes):
      for s, signal in tqdm(enumerate(batch)):
        result[b, s] = sm(signal)
    if self.rm_transients:
        result = result[:,:,-int(self.cfg.probes_signaltime*self.cfg.samplerate):]        
    if self.rm_dc:
        mean = np.mean(result, axis=-1, keepdims=True)
        result = result - mean
    return result