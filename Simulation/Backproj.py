import numpy as np

# Backprojection functions
def backproject_hw_accurate(R, B, tau):
    """
    Compute y_i for a single correlation matrix (frame), using unique off-diagonal terms only.

    Parameters
    ----------
    R : ndarray
        Correlation matrix of shape (N_ch, N_ch)
    B : ndarray
        Beamforming matrix of shape (N_ch, N_vec)
    tau : ndarray
        Bias term of shape (N_vec,)

    Returns
    -------
    y : ndarray
        Output values y_i, shape (N_vec,)
    """
    N_ch, N_vec = B.shape
    y = np.zeros(N_vec, dtype=np.float64)

    diag_R = np.real(np.diag(R))

    for i in range(N_vec):
        b = B[:, i]

        # Diagonal term: Σ_jj * |b_ji|^2
        diag_term = np.sum(diag_R * np.abs(b)**2)

        # Off-diagonal term: sum_{j<k} 2 * Re( conj(b_j) * R_jk * b_k )
        offdiag_sum = 0.0
        for j in range(N_ch):
            for k in range(j + 1, N_ch):
                offdiag_sum += np.real(np.conj(b[j]) * R[j, k] * b[k])

        y[i] = diag_term + 2 * offdiag_sum

    return y - tau

def backproject_py_opt(R, B, tau):
    """
    Efficient computation of y_i = b_i^H * R * b_i for all beamformers b_i.

    Parameters
    ----------
    R : ndarray
        Correlation matrix (N_ch, N_ch)
    B : ndarray
        Beamforming matrix (N_ch, N_vec)
    tau : ndarray
        Bias term (N_vec,)

    Returns
    -------
    y : ndarray
        Output values y_i (N_vec,)
    """
    return  np.real(np.einsum('ij,ji->i', np.conj(B.T) @ R, B)) - tau