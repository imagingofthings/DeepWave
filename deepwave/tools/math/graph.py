# #############################################################################
# graph.py
# ========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# Author : Matthieu SIMEONI [meo@zurich.ibm.com]
# #############################################################################

"""
Graph Signal Processing Tools.
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import scipy.spatial as spatial


def laplacian_exp(R, normalized=True):
    r"""
    Sparse Graph Laplacian based on exponential-decay metric::

        L     = I - D^{-1/2} W D^{-1/2}  OR
        L_{n} = (2 / \mu_{\max}) L - I

    Parameters
    ----------
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points.
    normalized : bool
        Rescale Laplacian spectrum to [-1, 1].

    Returns
    -------
    L : :py:class:`~scipy.sparse.csr_matrix`
        (N_px, N_px) Laplacian operator.
        If `normalized = True`, return L_{n}.
        If `normalized = False`, return L.
    rho : float
        Scale parameter \rho corresponding to the average distance of a point
        on the graph to its nearest neighbor.
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
    W = W.tocsr()

    # Form Graph Laplacian
    D = W.sum(axis=0)
    D_hinv = sp.diags((1 / np.sqrt(D)).tolist()[0], 0, format='csr')
    I_sp = sp.identity(W.shape[0], dtype=np.float, format='csr')
    L = I_sp - D_hinv.dot(W.dot(D_hinv))

    if normalized:
        D_max = splinalg.eigsh(L, k=1, return_eigenvectors=False)
        Ln = (2 / D_max[0]) * L - I_sp
        return Ln, rho
    else:
        return L, rho


class ConvolutionalFilter:
    """
    Helper class for fast evaluation of Chebychev polynomial filters.
    """

    def __init__(self, Ln, K):
        """
        Parameters
        ----------
        Ln : :py:class:`~scipy.sparse.csr_matrix`
            (N_px, N_px) normalized graph Laplacian.
        K : int
            Order of polynomial filter.
        """
        if not isinstance(Ln, sp.csr_matrix):
            raise ValueError('Parameter[Ln] must be (N_px, N_px) CSR.')
        self._Ln = Ln
        self._N_px = Ln.shape[0]

        if not (isinstance(K, int) and (K >= 0)):
            raise ValueError('Parameter[K] must be non-negative.')
        self._K = K

    def _next_cheby(self, x_km1, x_km2):
        r"""
        Recursive computation of k-th Chebychev filter based on lower-order
        filters.

        T_{k}(\tilde{L}) x = (2 * \tilde{L} T_{k-1}(\tilde{L}) x -
                                            T_{k-2}(\tilde{L}) x)

        Parameters
        ----------
        x_km1 : :py:class:`~numpy.ndarray`
            ([N_sample], N_px) result of T_{k-1}(\tilde{L}) x
        x_km2 : :py:class:`~numpy.ndarray`
            ([N_sample], N_px) result of T_{k-2}(\tilde{L}) x

        Returns
        -------
        x_k : :py:class:`~numpy.ndarray`
            ([N_sample], N_px) result of T_{k}(\tilde{L}) x
        """
        was_1d = (x_km1.ndim == 1) and (x_km2.ndim == 1)
        if was_1d:
            x_km1 = x_km1.reshape(1, self._N_px)
            x_km2 = x_km2.reshape(1, self._N_px)

        x_k = 2 * self._Ln.dot(x_km1.T) - x_km2.T
        if was_1d:
            x_k = x_k.squeeze()
        return x_k.T

    def _sub_filter(self, x):
        r"""
        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            ([N_sample], N_px) signal.

        Returns
        -------
        y : :py:class:`~numpy.ndarray`
            ([N_sample], K + 1, N_px) filtered sub-signals.

            y[i, j] = T_{j}(\tilde{L}) x[i]
        """
        was_1d = (x.ndim == 1)
        if was_1d:
            x = x.reshape(1, self._N_px)
        N_sample = x.shape[0]

        x_km2, x_km1 = x, self._Ln.dot(x.T).T
        y = np.zeros((N_sample, self._K + 1, self._N_px), dtype=x.dtype)
        y[:, 0], y[:, 1] = x_km2, x_km1
        for k in range(2, self._K + 1):
            x_k = self._next_cheby(x_km1, x_km2)
            y[:, k] = x_k
            x_km1, x_km2 = x_k, x_km1

        if was_1d:
            y = np.squeeze(y, axis=0)
        return y

    def filter(self, a, x):
        r"""
        Apply polynomial filter `a` to signal `x`.

        Parameters
        ----------
        a : :py:class:`~numpy.ndarray`
            (K + 1,) filter coefficients.
        x : :py:class:`~numpy.ndarray`
            ([N_sample], N_px) signal.

        Returns
        -------
        y : :py:class:`~numpy.ndarray`
            ([N_sample], N_px) filtered signal.

            y[i] = \sum_{k = 0}^{K} a[k] T_{k}(\tilde{L}) x[i]
        """
        was_1d = (x.ndim == 1)
        if was_1d:
            x = x.reshape(1, self._N_px)
        N_sample = x.shape[0]

        x_km2, x_km1 = x, self._Ln.dot(x.T).T
        y = a[1] * x_km1 + a[0] * x_km2
        for k in range(2, self._K + 1):
            x_k = self._next_cheby(x_km1, x_km2)
            y += a[k] * x_k
            x_km1, x_km2 = x_k, x_km1

        if was_1d:
            y = np.squeeze(y, axis=0)
        return y

    def trace(self, a, b):
        r"""
        Parameters
        ----------
        a : :py:class:`~numpy.ndarray`
            ([N_sample], N_px)
        b : :py:class:`~numpy.ndarray`
            ([N_sample], N_px)

        Returns
        -------
        c : :py:class:`~numpy.ndarray`
            ([N_sample], K + 1)

            c[i, j] = a[i].T T_{j}(\tilde{L}) b[i]
        """
        was_1d = (a.ndim == 1) and (b.ndim == 1)
        if was_1d:
            a = a.reshape(1, self._N_px)
            b = b.reshape(1, self._N_px)
        N_sample = a.shape[0]

        c = np.zeros((N_sample, self._K + 1), dtype=a.dtype)
        x_km2, x_km1 = b, self._Ln.dot(b.T).T
        c[:, 0] = np.sum(a * x_km2, axis=1)
        c[:, 1] = np.sum(a * x_km1, axis=1)
        for k in range(2, self._K + 1):
            x_k = self._next_cheby(x_km1, x_km2)
            c[:, k] = np.sum(a * x_k, axis=1)
            x_km1, x_km2 = x_k, x_km1

        if was_1d:
            c = np.squeeze(c, axis=0)
        return c

    @classmethod
    def estimate_order(cls, XYZ, rho, wl, eps):
        r"""
        Compute order of polynomial filter to approximate asymptotic
        point-spread function on \cS^{2}.

        Parameters
        ----------
        XYZ : :py:class:`~numpy.ndarray`
            (3, N_antenna) Cartesian instrument coordinates.
        rho : float
            Scale parameter \rho corresponding to the average distance of a point
            on the graph to its nearest neighbor.
            Output of :py:func:`~deepwave.tools.math.graph.laplacian_exp`.
        wl : float
            Wavelength of observations [m].
        eps : float
            Ratio in (0, 1).
            Ensures all PSF magnitudes lower than `max(PSF)*eps` past the main
            lobe are clipped at 0.

        Returns
        -------
        K : int
            Order of polynomial filter.
        """
        XYZ = XYZ / wl
        XYZ_centroid = np.mean(XYZ, axis=1, keepdims=True)
        XYZ_radius = np.mean(linalg.norm(XYZ - XYZ_centroid, axis=0))

        theta = np.linspace(0, np.pi, 1000)
        f = 20 * np.log10(np.abs(np.sinc(theta / np.pi)))
        eps_dB = 10 * np.log10(eps)
        theta_max = np.max(theta[f >= eps_dB])

        beam_width = theta_max / (2 * np.pi * XYZ_radius)
        K = np.sqrt(2 - 2 * np.cos(beam_width)) / rho
        K = int(np.ceil(K))
        return K

    def fit(self, x, y):
        r"""
        Find least-squares solution `a` to

        [T_{0}(\tilde{L}) x, \ldots, T_{K}(\tilde{L}) x] a = y

        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            (N_px,) signal
        y : :py:class:`~numpy.ndarray`
            (N_px,) signal

        Returns
        -------
        a : :py:class:`~numpy.ndarray`
            (K + 1,) filter coefficients.
        """
        T = self._sub_filter(x)
        G = T @ T.transpose()
        Ty = T @ y

        a, _ = splinalg.cg(G, Ty)
        return a

    def operator(self, a):
        """
        Parameters
        ----------
        a : :py:class:`~numpy.ndarray`
            (K + 1,) filter coefficients.

        Returns
        -------
        h : :py:class:`~scipy.sparse.csr_matrix`
            (N_px, N_px) sparse operator

            h = \sum_{k = 0}^{K} a[k] T_{k}(\tilde{L})
        """
        N_px = self._Ln.shape[0]
        h_dense = self.filter(a, np.eye(N_px))
        h = sp.csr_matrix(h_dense)
        return h
