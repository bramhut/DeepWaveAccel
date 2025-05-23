import numpy as np

# Goertzel algorithm for computing the DFT at a specific frequency
def goertzel(x, k, N, normalize=True):
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

    Returns
    -------
    X_k : complex
        DFT value at frequency k.
    """
    # Initialize variables
    s_prev = 0
    s_prev2 = 0
    omega_0 = 2 * np.pi * k / N

    # Compute the Goertzel algorithm
    for n in range(N):
        s = x[n] + 2 * np.cos(omega_0) * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    res = s_prev2 - np.exp(1j * omega_0) * s_prev
    if normalize:
        res /= N  # Normalize the result by the length of the signal
    return res

# # Parameters
# omega_0 = np.pi / 4  # example frequency
# N = 100              # length of signal
# x = np.random.randn(N)  # example input signal

# # Initialize s[n] with zeros
# s = np.zeros(N)

# # Compute s[n] using equation (2.17)
# for n in range(2, N):
#     s[n] = x[n] + 2 * np.cos(omega_0) * s[n - 1] - s[n - 2]

# # Define s1 and s2 (example values)
# s1 = s[1]
# s2 = s[2]

# # Compute X[k] using equation (2.18)
# X_k = np.exp(1j * omega_0) * s1 - s2
# # or equivalently using the expanded form
# X_k_alt = np.cos(omega_0) * s1 + 1j * np.sin(omega_0) * s1 - s2

# print(f"X[k] = {X_k}")
# print(f"Alternative expression of X[k] = {X_k_alt}")