import numpy as np
import scipy

def change_nan_inf(x, nan=None, inf=None, dtype='d'):
    info = np.finfo(dtype) 
    if not nan:
        nan = info.min
    if not inf:
        inf = info.max
    x = np.where(x==np.nan, x, nan)
    x = np.where(x==np.inf, x, inf)
    return x


def to_level(x, norm=None):
    if norm is None:
        norm = int(x.shape[0])   
    level = 20 * np.log10(x/norm)
    return level

def from_level(level, norm=None):
    if norm is None:
        norm = int(level.shape[0])
    x = level * np.log(10) / 20
    x = np.exp(x) * norm
    return x


def expspace(min, max, n=100, base=10):
    min = np.log(min)/np.log(base)
    max = np.log(max)/np.log(base)
    rnge = np.linspace(min, max, n)
    return np.power(base, rnge)


def reduce_idx(x, indices, dims=[0]):
    """
    Reduces x by removing indices along dims

    Parameters:
    x (numpy.array): Array that gets reduced
    indices (list): List that defines the index for every dimension to be reduced
    dims (list): List that defines the dimensions that are to be reduced

    Returns:
    numpy.array: x reduced according to indices and dims
    """
    if len(indices) != len(dims):
        raise ValueError("Length of indices and dims must be the same")

    for idx, dim in zip(indices, dims):
        x = np.delete(x, idx, dim)

    return x


def permute_matrix(d, v=2):
    n = v**d
    def _fun(i):
        x = np.arange(n)
        arg = n/(v**i)
        x = np.mod(x, arg)
        x = np.sort(x)
        x = np.mod(x, v)
        return x
    return np.array([_fun(i) for i in range(d)]).T


def save2wave(x, title='new_wav', normalize=True, sr=96_000, peak=-0.5, dtype=np.float32):
    """
    Normalizes x and saves it to WAV format
    
    Parameters
    ----------
    x (np.array): input signal
    title (string): file_name
    sr (int): samplerate
    peak (int) = max allowed peak in dB
    dtype (type): target type
    
    Returns
    -------
    
    x (dtype): normalized x
    """
    
    allowed_types =[np.float16, np.float32, np.uint8, np.int16, np.int32]
    
    assert dtype in allowed_types
    
    if normalize:
        x = x.astype(np.float128)
        norm = np.max(np.abs(x))
        if not norm==0:
            x /= norm
            x *= from_level(np.array([peak]))
        try:
            mn, mx = np.iinfo(dtype).min, np.iinfo(dtype).max
            ctr = 0.5*(mn + mx) + 0.5
            hlf = 0.5*(mx-mn-1)
            x = x*hlf + ctr
        except:
            pass
        
    x = x.astype(dtype)    
    scipy.io.wavfile.write(title+'.wav', sr, x)
    return x


def rms(x, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def snr_db(s, n, axis=None):
    snr = rms(s, axis) / rms(n, axis)
    return 20*np.log10(snr)


def noise_like(x, db):
    """
    db: float = noise power level in decibel
    """
    var = 10**(db/10)  # Power linear
    std = np.sqrt(var) # RMS linear
    noise = np.random.normal(size=x.shape, scale=std)
    return noise