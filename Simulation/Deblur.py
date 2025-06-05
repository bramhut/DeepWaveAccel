import numpy as np

def chebyshev_conv(L, x, theta):
    """
    Compute y = sum_{k=0}^K theta_k * z_k
    where
    z_0 = x
    z_1 = L @ x
    z_k = 2 * L @ z_{k-1} - z_{k-2} for k >= 2
    
    Parameters:
    -----------
    L : scipy.sparse.dia_matrix
        Normalized (sparse) Laplacian matrix.
    x : np.ndarray
        Input image \mathbf{x}.
    theta : array_like
        Chebyshev coefficients [theta_0, theta_1, ..., theta_K].
    
    Returns:
    --------
    y : np.ndarray
        Output image \mathbf{y}.
    """
    K = len(theta) - 1
    
    z_k_minus_two = x
    y = theta[0] * z_k_minus_two
    
    if K == 0:
        return y
    
    z_k_minus_one = L @ x
    y += theta[1] * z_k_minus_one
    
    for k in range(2, K+1):
        z_k = 2 * L @ z_k_minus_one - z_k_minus_two
        y += theta[k] * z_k
        z_k_minus_two, z_k_minus_one = z_k_minus_one, z_k
    
    return y

import numpy as np

def retanh_activation(x, alpha=1.0):
    """
    Compute beta = alpha / tanh(1)
    Then return elementwise max(0, beta * tanh(x))
    
    Parameters:
    -----------
    x : np.ndarray
        Input array.
    alpha : float
        Scalar alpha.
        
    Returns:
    --------
    result : np.ndarray
        Array after applying the operation.
    """
    beta = alpha / np.tanh(1.0)
    return np.maximum(0, beta * np.tanh(x))

