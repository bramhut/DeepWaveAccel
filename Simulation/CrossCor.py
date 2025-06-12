import numpy as np


# Normalization functions
# Function to estimate the maximum eigenvalue of a matrix using power iteration
def estimate_max_eigenvalue(S, num_iter):
    n = S.shape[0]
    # Fixed deterministic initial vector (normalized ones)
    v = np.ones(n, dtype=np.complex128) / np.sqrt(n)
    for _ in range(num_iter):
        v = S @ v
        L2_norm = np.linalg.norm(v)
        v /= L2_norm
    return np.real(v.conj().T @ S @ v)


# Normalize the correlation matrix S by its maximum eigenvalue
def normalize_correlation_matrix(S, num_iter=10):
    max_eigenvalue = estimate_max_eigenvalue(S, num_iter)
    if max_eigenvalue == 0:
        return S  # Avoid division by zero
    return S / max_eigenvalue


# Cross-correlation identical to the deepwave reference implementation
def cross_correlation_deepwave_ref(dft, num_iter_power=10):
    return cross_correlation(dft, alpha=0, group_size=9, num_iter_power=num_iter_power)

# Cross-correlation with optional exponential moving average (EMA) smoothing and grouping
def cross_correlation(dft, alpha=0.95, group_size=1, num_iter_power=10):
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
        R = (1 - alpha) * outer + alpha * R
        R_all.append(R.copy())
    R_all = np.stack(R_all)

    # Group and sum every 'group_size' resulting correlation matrices (only when not using EMA)
    if alpha == 0:
        R_all = R_all[:(R_all.shape[0] // group_size) * group_size].reshape(-1, group_size, N_ch, N_ch).sum(axis=1)

    # Normalize each correlation matrix in R_all
    R_all = np.array([normalize_correlation_matrix(S, num_iter=num_iter_power) for S in R_all])
    return R_all