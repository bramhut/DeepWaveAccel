import numpy as np
import scipy.sparse as sp
import scipy.linalg as linalg
import scipy.spatial as spatial
import scipy.sparse.linalg as splinalg
from scipy.sparse import coo_matrix, dia_matrix, csc_matrix, csr_matrix, issparse
from pygsp.graphs import Graph


def prepare_laplacian(laplacian):
    """Prepare a graph Laplacian to be fed to a graph convolutional layer.
    """
    def estimate_lmax(laplacian, tol=5e-3):
        """Estimate the largest eigenvalue of an operator.
        """
        lmax = sp.linalg.eigsh(laplacian, k=1, tol=tol, ncv=min(laplacian.shape[0], 10), return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1 + 2 * tol  # Be robust to errors.
        return lmax

    def scale_operator(L, lmax, scale=1):
        """Scale the eigenvalues from [0, lmax] to [-scale, scale].
        """
        I = sp.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L *= 2 * scale / lmax
        L -= I
        return L

    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)
    return laplacian


def cvxhull_graph(R: np.ndarray, cheb_normalized: bool = True, compute_differential_operator: bool = True):

    r"""
    Build the convex hull graph of a point set in :math:`\mathbb{R}^3`.
    The graph edges have exponential-decay weighting.
    Definitions of the graph Laplacians:

    .. math::

        L     = I - D^{-1/2} W D^{-1/2},\qquad        L_{n} = (2 / \mu_{\max}) L - I

    Parameters
    ----------
    R : :py:class:`~numpy.ndarray`
        (N,3) Cartesian coordinates of point set with size N. All points must be **distinct**.
    cheb_normalized : bool
        Rescale Laplacian spectrum to [-1, 1].
    compute_differential_operator : bool
        Computes the graph gradient.

    Returns
    -------
    G : :py:class:`~pygsp.graphs.Graph`
        If ``cheb_normalized = True``, ``G.Ln`` is created (Chebyshev Laplacian :math:`L_{n}` above)
        If ``compute_differential_operator = True``, ``G.D`` is created and contains the gradient.
    rho : float
        Scale parameter :math:`\rho` corresponding to the average distance of a point
        on the graph to its nearest neighbors.

    Examples
    --------

    .. plot::

        import numpy as np
        from pycgsp.graph import cvxhull_graph
        from pygsp.plotting import plot_graph
        theta, phi = np.linspace(0,np.pi,6, endpoint=False)[1:], np.linspace(0,2*np.pi,9, endpoint=False)
        theta, phi = np.meshgrid(theta, phi)
        x,y,z = np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)
        R = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
        G, _ = cvxhull_graph(R)
        plot_graph(G)

    Warnings
    --------
    In the newest version of PyGSP (> 0.5.1) the convention is changed: ``Graph.D`` is the divergence operator and
    ``Graph.D.transpose()`` the gradient (see routine `Graph.compute_differential_operator <https://pygsp.readthedocs.io/en/latest/reference/graphs.html#pygsp.graphs.Graph.compute_differential_operator>`_). The code should be adapted when this new version is released.

    """

    # Form convex hull to extract nearest neighbors. Each row in
    # cvx_hull.simplices is a triangle of connected points.
    cvx_hull = spatial.ConvexHull(R.T)
    cols = np.roll(cvx_hull.simplices, shift=1, axis=-1).reshape(-1)
    rows = cvx_hull.simplices.reshape(-1)

    # Form sparse affinity matrix from extracted pairs
    W = sp.coo_matrix((cols * 0 + 1, (rows, cols)),
                      shape=(cvx_hull.vertices.size, cvx_hull.vertices.size))
    # Symmetrize the matrix to obtain an undirected graph.
    extended_row = np.concatenate([W.row, W.col])
    extended_col = np.concatenate([W.col, W.row])
    W.row, W.col = extended_row, extended_col
    W.data = np.concatenate([W.data, W.data])
    W = W.tocsr().tocoo()  # Delete potential duplicate pairs

    # Weight matrix elements according to the exponential kernel
    distance = linalg.norm(cvx_hull.points[W.row, :] -
                           cvx_hull.points[W.col, :], axis=-1)
    rho = np.mean(distance)
    W.data = np.exp(- (distance / rho) ** 2)
    W = W.tocsc()

    G = _graph_laplacian(W, R, compute_differential_operator=compute_differential_operator,
                         cheb_normalized=cheb_normalized)
    return G, rho

def _graph_laplacian(W, R, compute_differential_operator=False, cheb_normalized=False):
    '''
    Form Graph Laplacian
    '''
    G = Graph(W, gtype='undirected', lap_type='normalized', coords=R)
    G.compute_laplacian(lap_type='normalized')  # Stored in G.L, sparse matrix, csc ordering
    if compute_differential_operator is True:
        G.compute_differential_operator()  # stored in G.D, also accessible via G.grad() or G.div() (for the adjoint).
    else:
        pass

    if cheb_normalized:
        D_max = splinalg.eigsh(G.L, k=1, return_eigenvectors=False)
        Ln = (2 / D_max[0]) * G.L - sp.identity(W.shape[0], dtype=np.float64, format='csc')
        G.Ln = Ln
    else:
        pass
    return G

def laplacian_scipy(R):
    """Get the icosahedron laplacian matrix
    Args:
        R : :py:class:`~numpy.ndarray`
        (N,3) Cartesian coordinates of point set with size N. All points must be **distinct**.
    Returns:
        laplacian: `scipy.sparse` laplacian
        rho: float: laplacian order
    """
    G, rho = cvxhull_graph(R)
    laplacian = prepare_laplacian(G.L)
    return laplacian, rho

def sparsify_band_symmetric(matrix, threshold):
    """
    Sparsify a symmetric scipy.sparse matrix by keeping only the main and upper diagonals 
    that have at least one value above a normalized threshold.
    
    Args:
        matrix (scipy.sparse.dia_matrix): Symmetric matrix in DIA format.
        threshold (float): Normalized threshold (between 0 and 1).
    
    Returns:
        numpy.ndarray: A 2D array in banded format where:
                       - The first column contains diagonal offsets (>= 0)
                       - Remaining columns contain diagonal values aligned by row index
    """
    if not issparse(matrix):
        raise TypeError("Input must be a SciPy sparse matrix.")
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1")
    
    matrix = matrix.todia()  # Ensure DIA format
    max_val = np.max(np.abs(matrix.data)) if matrix.data.size > 0 else 0
    cutoff = threshold * max_val

    # Keep only main and upper diagonals (offset >= 0)
    upper_mask = matrix.offsets >= 0
    upper_data = matrix.data[upper_mask]
    upper_offsets = matrix.offsets[upper_mask]

    # Keep diagonals with any value >= cutoff
    keep_mask = np.any(np.abs(upper_data) >= cutoff, axis=1)
    kept_data = upper_data[keep_mask]
    kept_offsets = upper_offsets[keep_mask]

    # Prepare banded output: rows = diagonals, columns = [offset, diag values...]
    banded = np.zeros((kept_data.shape[0], matrix.shape[0] + 1))
    banded[:, 0] = kept_offsets  # first column = diagonal offsets
    banded[:, 1:] = kept_data    # remaining columns = diagonal values

    return banded

