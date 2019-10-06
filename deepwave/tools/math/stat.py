# ############################################################################
# stat.py
# =======
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Statistical functions not available in `SciPy <https://www.scipy.org/>`_.
"""

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats


class Wishart:
    """
    `Wishart <https://en.wikipedia.org/wiki/Wishart_distribution>`_ distribution.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from acoustic_camera.util.math.stat import Wishart
       import scipy.linalg as linalg

       np.random.seed(0)

       def hermitian_array(N: int) -> np.ndarray:
           '''
           Construct a (N, N) Hermitian matrix.
           '''
           D = np.arange(1, N + 1)
           Rmtx = np.random.randn(N,N) + 1j * np.random.randn(N, N)
           Q, _ = linalg.qr(Rmtx)

           A = (Q * D) @ Q.conj().T
           return A

    .. doctest::

       >>> A = hermitian_array(N=4)  # random (N, N) PSD array.
       >>> W = Wishart(A, n=7)

       >>> samples = W(N_sample=2)  # 2 samples of the distribution.
       >>> np.around(samples, 2)
       array([[[ 19.92+0.j  ,  -8.21-3.72j,  -1.44-0.15j, -15.16+1.41j],
               [ -8.21+3.72j,  12.62+0.j  ,   5.91+2.06j,  13.14-2.17j],
               [ -1.44+0.15j,   5.91-2.06j,   8.47+0.j  ,  11.35-1.76j],
               [-15.16-1.41j,  13.14+2.17j,  11.35+1.76j,  31.42+0.j  ]],
       <BLANKLINE>
              [[ 32.27+0.j  ,   8.45-6.03j,  -4.68+5.54j,   2.63+6.9j ],
               [  8.45+6.03j,   7.96+0.j  ,  -3.77+1.8j ,  -1.11+3.37j],
               [ -4.68-5.54j,  -3.77-1.8j ,   7.08+0.j  ,   4.8 -2.09j],
               [  2.63-6.9j ,  -1.11-3.37j,   4.8 +2.09j,   8.79+0.j  ]]])
    """

    def __init__(self, V, n):
        """
        Parameters
        ----------
        V : :py:class:`~numpy.ndarray`
            (p, p) positive-semidefinite scale matrix.
        n : int
            degrees of freedom.
        """
        p = len(V)

        if not (V.shape == (p, p) and np.allclose(V, V.conj().T)):
            raise ValueError('Parameter[V] must be hermitian symmetric.')
        if not (n > p):
            raise ValueError(f'Parameter[n] must be greater than {p}.')

        self._V = V
        self._p = p
        self._n = n

        Vq = linalg.sqrtm(V)
        _, R = linalg.qr(Vq)
        self._L = R.conj().T

    def __call__(self, N_sample=1):
        """
        Generate random samples.

        Parameters
        ----------
        N_sample : int
            Number of samples to generate.

        Returns
        -------
        x : :py:class:`~numpy.ndarray`
            (N_sample, p, p) samples.

        Notes
        -----
        The Wishart estimate is obtained using the `Bartlett Decomposition`_.

        .. _Bartlett Decomposition: https://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition
        """
        if N_sample < 1:
            raise ValueError('Parameter[N_sample] must be positive.')

        A = np.zeros((N_sample, self._p, self._p))

        diag_idx = np.diag_indices(self._p)
        df = (self._n * np.ones((N_sample, 1)) - np.arange(self._p))
        A[:, diag_idx[0], diag_idx[1]] = np.sqrt(stats.chi2.rvs(df=df))

        tril_idx = np.tril_indices(self._p, k=-1)
        size = (N_sample, self._p * (self._p - 1) // 2)
        A[:, tril_idx[0], tril_idx[1]] = stats.norm.rvs(size=size)

        W = self._L @ A
        X = W @ W.conj().transpose(0, 2, 1)
        return X
