import numpy as np

# Goertzel algorithm for computing the DFT at a specific frequency
# bin efficiently. 
def goertzel_single(x, k, hann=True, normalize=True):
    """
    Compute the DFT at frequency bin k using the Goertzel algorithm.

    Parameters
    ----------
    x : array_like
        Input signal (1D array).
    k : int
        Frequency bin to compute.
    N : int
        Length of the input signal.
    hann: bool, optional
        If True, apply a Hann window to the input signal (default is True).
    normalize : bool, optional
        If True, normalize the result by the length of the signal (default is True).

    Returns
    -------
    X_k : complex
        DFT value at frequency k.
    """
    # Initialize variables
    N = x.shape[0]
    s_prev = 0
    s_prev2 = 0
    omega_0 = 2 * np.pi * k / N
      
    if hann:
        # Apply a Hann window to the input signal (verified to be correct)
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
        x = x * window[:, np.newaxis]

    # Compute the Goertzel algorithm by dividing the input signal into frames
    for n in range(N):
        s = x[n] + 2 * np.cos(omega_0) * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    res = s_prev2 - np.exp(1j * omega_0) * s_prev
    if normalize:
        res /= N  # Normalize the result by the length of the signal
    return res


# Goertzel algorithm for computing the DFT at a specific frequency
# bin efficiently. Divides the input signal into frames with the specified
# overlap and applies a Hann window (optional). Returns the DFT values for each frame.
def goertzel(x, k, Nf, overlap=0.5, hann=True, normalize=True):
    """
    Compute the DFT at frequency bin k using the Goertzel algorithm with overlapping frames.

    Parameters
    ----------
    x : array_like
        Input signal (2D array): (samples, channels).
    k : int
        Frequency bin to compute.
    Nf : int
        Length of each frame.
    overlap : float, optional
        Overlap between frames (default is 0.5).
    
    Returns
    -------
    X_k : array_like
        DFT values at frequency k for each frame and channel.
    step : int
        Step size between frames.
    """
    
    if not (0 <= overlap < 1):
        raise ValueError("Overlap must be between 0 and 1.")

    step = int(Nf * (1 - overlap))
    num_frames = (len(x) - Nf) // step + 1
    X_k = np.zeros((num_frames, x.shape[1]), dtype=complex)

    for i in range(num_frames):
        start = i * step
        end = start + Nf
        frame = x[start:end]
        X_k[i] = goertzel_single(frame, k, hann, normalize)

    return X_k, step


def goertzel_multi_bin_with_logging(x, bins, Nf, overlap=0.5, hann=True, normalize=True, logger=None):
    """
    Compute the DFT using the Goertzel algorithm for multiple frequency bins,
    returning individual results per bin. Logs one summary for input/output.

    Parameters
    ----------
    x : ndarray
        Input signal, shape (samples, channels)
    bins : list of int
        Frequency bins to compute
    Nf : int
        Frame length
    overlap : float
        Overlap fraction [0, 1)
    hann : bool
        Apply Hann window
    normalize : bool
        Normalize output by frame length
    logger : SignalLogger or None
        Logger to collect min/max/mean/std info

    Returns
    -------
    X_bins : ndarray
        DFT values per (frame, channel, bin), shape (n_frames, n_channels, n_bins)
    step : int
        Frame step size
    """
    if not (0 <= overlap < 1):
        raise ValueError("Overlap must be between 0 and 1.")

    step = int(Nf * (1 - overlap))
    n_samples, n_ch = x.shape
    n_frames = (n_samples - Nf) // step + 1
    n_bins = len(bins)

    X_bins = np.zeros((n_frames, n_ch, n_bins), dtype=np.complex64)

    if logger:
        logger.log("goertzel/input", x)

    if hann:
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(Nf) / (Nf - 1)))
        window = window[:, np.newaxis]

    for j, bin_k in enumerate(bins):
        omega_0 = 2 * np.pi * bin_k / Nf
        cos_omega = 2 * np.cos(omega_0)
        exp_omega = np.exp(1j * omega_0)

        for i in range(n_frames):
            start = i * step
            frame = x[start:start+Nf]

            if hann:
                frame = frame * window

            s_prev = np.zeros(n_ch)
            s_prev2 = np.zeros(n_ch)

            for n in range(Nf):
                s = frame[n] + cos_omega * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s

            res = s_prev2 - exp_omega * s_prev
            if normalize:
                res /= Nf

            X_bins[i, :, j] = res

    if logger:
        logger.log("goertzel/output", X_bins)

    return X_bins, step
