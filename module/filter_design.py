import numpy as np
from module.utils import least_squares

def compute_minimum_phase(magnitude_spectrum, dB=True, degree=True):
    """
    Computes the phase spectrum of a minimum phase system that corresponds to the given magnitude spectrum.
    
    Parameters
    ----------
        magnitude_spectrum : np.array
            The magnitude spectrum of a system in dB (for DC and positive frequencies only).
        dB : bool
            If True, magnitude_spectrum is expected to be given in dB.
        degree : bool
            If True, returns the phase in degree, otherwise in radians.
    
    Returns
    -------
        np.array
            The phase spectrum of the minimum phase system.
    """

    # Convert dB values to linear scale
    if dB:
        magnitude_spectrum = 10**(magnitude_spectrum / 20)
    
    # Create a mirrored version of the magnitude spectrum
    magnitude_spectrum = np.hstack([magnitude_spectrum, np.flip(magnitude_spectrum[1:])])
    
    # Compute the cepstrum
    cepstrum = np.fft.ifft(np.log(magnitude_spectrum))
    
    # Adjust the cepstrum values to ensure causality in the time domain
    cepstrum[1:len(cepstrum)//2] *= 2
    if len(cepstrum) % 2 == 1:
        cepstrum[len(cepstrum)//2] = cepstrum[len(cepstrum)//2].real   
    cepstrum[len(cepstrum)//2+1:] = 0
    
    # Compute the imaginary part of the Fourier transform of the adjusted cepstrum
    phase_spectrum = np.imag(np.fft.rfft(cepstrum))
    
    # If the degree flag is set, convert radians to degree
    if degree:
        phase_spectrum /= np.pi/180
    
    return phase_spectrum, cepstrum


def FIR_Filter_Freq_Sampling(freq_hz, magnitude_db, phase_deg, tabs, samplerate, weights=None):
    
    # Define frequency points and create frequency matrix
    freq_rad = 2*np.pi * freq_hz 
    # Ensure filter is realizable (1)
    freq_rad = np.concatenate([freq_rad, -freq_rad[::-1]])
    
    # Build coef-matrix for LSE
    rnge = np.arange(tabs) / samplerate
    #rnge = np.linspace(0, samplerate/tabs, tabs, endpoint=True)
    freq_matrix = np.exp(-1j * np.outer(freq_rad, rnge))

    # Convert magnitude from dB to amp
    magnitude = 10**(magnitude_db / 20)
    # Convert phase from degree to rad
    phase_rad = np.radians(phase_deg)
    
    # Ensure filter is realizable (2)
    magnitude = np.concatenate([magnitude, magnitude[::-1]])
    phase_rad = np.concatenate([phase_rad, -phase_rad[::-1]])

    # Complex desired frequency response
    desired_freq_resp = magnitude * np.exp(1j * phase_rad)
    
    if weights is None:
        # Get pseudo inverse of frequency matrix
        pseudo_inverse = np.linalg.pinv(freq_matrix)
        # Least squares solution for filter coefficients
        fir_truncated = np.dot(pseudo_inverse, desired_freq_resp).real
        
    else:
        # Ensure filter is realizable (3)
        weights = np.concatenate([weights, weights[::-1]])
        # Calculate weighted least-squares solution
        fir_truncated = least_squares(A=freq_matrix,
                                   b=desired_freq_resp,
                                   w=weights).real
    
    fir = np.zeros(samplerate)
    fir[:tabs] = fir_truncated
    
    bins = np.fft.rfftfreq(samplerate, 1/samplerate)
    
    return bins, fir