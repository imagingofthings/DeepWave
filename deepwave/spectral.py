# ############################################################################
# spectral.py
# ===========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Spectral algorithms to estimate intensity field.
"""

import numpy as np
import scipy.linalg as linalg

import imot_tools.phased_array as phased_array


def DAS(XYZ, S, wl, R, A_H=None):
    """
    Delay-and-Sum (DAS) algorithm.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian instrument geometry.
    S : :py:class:`~numpy.ndarray`
        (N_antenna, N_antenna) Hermitian visibility matrix.
    wl : float
        Wave-length >= 0 [m].
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points to scan.
    A_H : :py:class:`~numpy.ndarray`
        (N_antenna, N_px) pre-computed hermitian transposed steering matrix.

    Returns
    -------
    SP : :py:class:`~numpy.ndarray`
        (N_px,) DAS spectrum.
    """
    N_antenna = XYZ.shape[1]
    if not (XYZ.shape == (3, N_antenna)):
        raise ValueError('Parameter[XYZ] must be (3, N_antenna) real-valued.')
    if not ((S.shape == (N_antenna, N_antenna)) and
            np.allclose(S, S.conj().T)):
        raise ValueError('Parameter[S] must be (N_antenna, N_antenna) hermitian.')

    N_px = R.shape[1]
    if not (R.shape == (3, N_px)):
        raise ValueError('Parameter[R] must be (3, N_px) real-valued.')

    if not (wl > 0):
        raise ValueError('Parameter[wl] must be positive.')

    D, V = linalg.eigh(S)
    idx = D > 0  # To avoid np.sqrt() issues.
    D, V = D[idx], V[:, idx]

    if A_H is None:
        A_H = phased_array.steering_operator(XYZ, R, wl).conj().T
    SP = linalg.norm(A_H @ (V * np.sqrt(D)), axis=1) ** 2
    return SP
