import numpy as np


import numpy as np

def estimate_max_eigenvalues(S, num_iter):
    """
    Pure NumPy, optimized with batched @ and np.linalg.norm.

    S: (n, n) or (k, n, n)
    num_iter: int

    Returns:
        float if input was (n, n)
        array of shape (k,) if input was (k, n, n)
    """

    # Promote single matrix to batch of size 1
    single_matrix = (S.ndim == 2)
    if single_matrix:
        S = S[None, ...]

    k, n, _ = S.shape

    # Initial vectors for all matrices
    v = np.ones((k, n), dtype=np.complex128) / np.sqrt(n)

    for _ in range(num_iter):
        # Batched mat-vec multiply → (k, n)
        v = (S @ v[..., None])[..., 0]

        # Compute ||v||^2 for each batch member
        norm = np.linalg.norm(v, axis=1)**2
        norm = np.where(norm == 0, 1, norm)

        # Normalize each vector
        v = v / norm[:, None]

    # Final Rayleigh quotient
    Sv = (S @ v[..., None])[..., 0]
    rq = np.sum(np.conj(v) * Sv, axis=1) * norm
    rq = np.real(rq)

    return rq[0] if single_matrix else rq



def normalize_correlation_matrix(S, floor_val, num_iter=10, energy_norm=False):
    """
    Normalize one matrix (n,n) or a batch of matrices (k,n,n)
    by dividing by max(max_eigenvalue, eig_value_floor).

    Returns:
        normalized_S : same shape as S
        norm_factors : float or array (k,)
    """

    if energy_norm:
        # Energy normalization: trace(S) / n
        trace_S = np.trace(S, axis1=1, axis2=2).real
        norm_factors = trace_S / np.sqrt(2)
        norm_factors = np.maximum(norm_factors, floor_val)
        
    else:
        # Compute eigenvalues (scalar or vector)
        max_eigenvalues = estimate_max_eigenvalues(S, num_iter)

        # Normalization factors: max(max_eigenvalue, floor)
        norm_factors = np.maximum(max_eigenvalues, floor_val)

    # Normalize
    normalized_S = S / norm_factors[:, None, None]
    return normalized_S, norm_factors



# Cross-correlation with optional exponential moving average (EMA) smoothing and grouping
def cross_correlation(dft, group_size=1, floor_value=1, num_iter_power=10, energy_norm=False):
    """
    Compute and store cross-correlation matrices (optionally using exponential moving average) for each frame of DFT data.
    This function calculates the cross-correlation matrices for each frame in the input DFT data. Optionally, it applies exponential moving average (EMA) smoothing to the correlation matrices. When EMA is not used (alpha=0), the resulting matrices can be grouped and summed over a specified group size. Each resulting correlation matrix is normalized before being returned.
    Args:
        dft (np.ndarray): Input array of shape (num_frames, num_channels), representing the DFT data for each frame and channel.
        alpha (float, optional): Smoothing factor for EMA (0 <= alpha < 1). If 0, no smoothing is applied. Default is 0.95.
        group_size (int, optional): Number of frames to group and sum the correlation matrices when EMA is not used (alpha=0). Default is 1.
        num_iter_power (int, optional): Number of iterations for power iteration to estimate the maximum eigenvalue for normalization. Default is 10.
    Returns:
        np.ndarray: Array of normalized cross-correlation matrices, shape depends on input and grouping.
    """
    # Calculate and store all correlation matrices using EMA for each frame
    
    N_ch = dft.shape[1]  # Number of channels
    R = np.zeros((N_ch, N_ch), dtype=np.complex128)
    R_all = []  # List to store R at each step

    # Calculate and store all correlation matrices using EMA for each frame
    for frame in dft:
        outer = np.outer(np.conj(frame), frame)
        R = outer
        R_all.append(R.copy())
    R_all = np.stack(R_all)

    # Group and sum every 'group_size' resulting correlation matrices 
    R_all = R_all[:(R_all.shape[0] // group_size) * group_size].reshape(-1, group_size, N_ch, N_ch).sum(axis=1)

    # Normalize each correlation matrix in R_all
    R_all, norms = normalize_correlation_matrix(R_all, floor_val=floor_value, num_iter=num_iter_power, energy_norm=energy_norm)
    return R_all, norms