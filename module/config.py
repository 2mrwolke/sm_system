from dataclasses import dataclass, field
from datetime import datetime
import pprint
import pytz
import numpy as np

@dataclass(kw_only=True, slots=True)
class Configuration:

  comment: str = field(default_factory=str)
  tags: list[str] = field(default_factory=list)
  _last_accessed: str = datetime.now(pytz.timezone(
    'Europe/Berlin')).strftime("%Y/%m/%d_%H:%M:%S")
  path: str = field(default_factory=str)#''# field(init=False, repr=False)
  
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
    
  seed: int = 42
  samplerate: int = 96_000
  active_sm_paths: list[int] = field(default_factory=list)
  max_order: int = 3
  dut: str = field(default_factory=str)
  snr: float = field(default_factory=float)
  hardware_latency: int = field(default_factory=int)
  delay: float = 0.

  filter1_type: str = '-- NONE --'
  filter1_fc: int = 440
  filter1_q: float = 0.707
  filter1_zeros: list[float] = field(default_factory=list)
  filter1_poles: list[float] = field(default_factory=list)
  filter1_gain: float = 1.
  filter1_tabs: int = 128
  
  filter21_type: str = '-- NONE --'
  filter21_fc: int = 440
  filter21_q: float = 0.707
  filter21_lag: float = 45.
  filter21_zeros: list[float] = field(default_factory=list)
  filter21_poles: list[float] = field(default_factory=list)
  filter21_gain: float = 1.
  filter21_tabs: int = 128
  
  filter22_type: str = '-- NONE --'
  filter22_fc: int = 440
  filter22_q: float = 0.707
  filter22_lag: float = 45.
  filter22_zeros: list[float] = field(default_factory=list)
  filter22_poles: list[float] = field(default_factory=list)
  filter22_gain: float = 1.    
  filter22_tabs: int = 128
  
  filter31_type: str = '-- NONE --'
  filter31_fc: int = 440
  filter31_q: float = 0.707
  filter31_lag: float = 45.
  filter31_zeros: list[float] = field(default_factory=list)
  filter31_poles: list[float] = field(default_factory=list)
  filter31_gain: float = 1.
  filter31_tabs: int = 128
  
  filter32_type: str = '-- NONE --'
  filter32_fc: int = 440
  filter32_q: float = 0.707
  filter32_lag: float = 45.  
  filter32_zeros: list[float] = field(default_factory=list)
  filter32_poles: list[float] = field(default_factory=list)
  filter32_gain: float = 1.
  filter32_tabs: int = 128

  constraint_amp_s2: str = 'ONE'
  constraint_amp_s3: str = 'ONE'    
  constraint_phase_s2: str = 'ONE'
  constraint_phase_s3: str = 'ONE'
     
  _keys: list[str] = field(init=False, repr=False)
  
  def __post_init__(self):
    self._keys = list(self.__slots__)[:-2]
    self._recalc()
    
  def save(self, path2file='new_config'):
    self.path = 'experiments/'+path2file+'/'
    np.save(path2file, self)
          
  def load(self, path):
    new_config = np.load(path, allow_pickle=True).item()
    for key in self.__slots__:
      setattr(self, key, getattr(new_config, key))
    self._set_time()
      
  def update(self, key, value):
    setattr(self, key, value)
    self._set_time()
    self._recalc()
      
  def _recalc(self):
    n_bins = np.log(self.probes_signaltime*self.samplerate) / np.log(2)
    self.probes_fft_n = int(2**np.round(n_bins + 0.5))
    self.probes_signaltime = self.probes_fft_n/self.samplerate
    self.probes_fft_delta = self.samplerate / self.probes_fft_n
    self.probes_transient_time *= self.samplerate
    self.probes_transient_time = np.round(self.probes_transient_time)
    self.probes_transient_time /= self.samplerate
      
  def _set_time(self):
    time = datetime.now(pytz.timezone('Europe/Berlin'))
    self._last_accessed = time.strftime("%Y/%m/%d_%H:%M:%S")
    
  def __str__(self):
    return pprint.pformat(self)

   
    