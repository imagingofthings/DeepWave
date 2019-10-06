# #############################################################################
# linalg.py
# =========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# #############################################################################

"""
Linear algebra routines.
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg

import imot_tools.phased_array as phased_array


def eighMax(A):
    r"""
    Evaluate :math:`\mu_{\max}(\bbB)` with

    :math:

    B = (\overline{\bbA} \circ \bbA)^{H} (\overline{\bbA} \circ \bbA)

    Uses a matrix-free formulation of the Lanczos algorithm.

    Parameters
    ----------
    A : :py:class:`~numpy.ndarray`
        (M, N) array.

    Returns
    -------
    D_max : float
        Leading eigenvalue of `B`.
    """
    if A.ndim != 2:
        raise ValueError('Parameter[A] has wrong dimensions.')

    def matvec(v):
        r"""
        Parameters
        ----------
        v : :py:class:`~numpy.ndarray`
            (N,) or (N, 1) array

        Returns
        -------
        w : :py:class:`~numpy.ndarray`
            (N,) array containing :math:`\bbB \bbv`
        """
        v = v.reshape(-1)

        C = (A * v) @ A.conj().T
        D = C @ A
        w = np.sum(A.conj() * D, axis=0).real
        return w

    M, N = A.shape
    B = splinalg.LinearOperator(shape=(N, N),
                                matvec=matvec,
                                dtype=np.float64)
    D_max = splinalg.eigsh(B, k=1, which='LM', return_eigenvectors=False)
    return D_max[0]


def psf_exp(XYZ, R, wl, center):
    """
    True complex plane-wave point-spread function.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian instrument coordinates.
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points.
    wl : float
        Wavelength of observations [m].
    center : :py:class:`~numpy.ndarray`
        (3,) Cartesian position of PSF focal point.

    Returns
    -------
    psf_mag2 : :py:class:`~numpy.ndarray`
        (N_px,) PSF squared magnitude.
    """
    N_antenna = XYZ.shape[1]
    if not (XYZ.shape == (3, N_antenna)):
        raise ValueError('Parameter[XYZ] must be (3, N_antenna) real-valued.')

    N_px = R.shape[1]
    if not (R.shape == (3, N_px)):
        raise ValueError('Parameter[R] must be (3, N_px) real-valued.')

    if not (wl > 0):
        raise ValueError('Parameter[wl] must be positive.')

    if not (center.shape == (3,)):
        raise ValueError('Parameter[center] must be (3,) real-valued.')

    A = phased_array.steering_operator(XYZ, R, wl)
    d = phased_array.steering_operator(XYZ, center.reshape(3, 1), wl)

    psf = np.reshape(d.T.conj() @ A, (N_px,))
    psf_mag2 = np.abs(psf) ** 2
    return psf_mag2


def psf_sinc(XYZ, R, wl, center):
    """
    Asymptotic point-spread function for uniform spherical arrays as antenna
    density converges to 1.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian instrument coordinates.
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points.
    wl : float
        Wavelength of observations [m].
    center : :py:class:`~numpy.ndarray`
        (3,) Cartesian position of PSF focal point.

    Returns
    -------
    psf_mag2 : :py:class:`~numpy.ndarray`
        (N_px,) PSF squared magnitude.
    """
    N_antenna = XYZ.shape[1]
    if not (XYZ.shape == (3, N_antenna)):
        raise ValueError('Parameter[XYZ] must be (3, N_antenna) real-valued.')

    N_px = R.shape[1]
    if not (R.shape == (3, N_px)):
        raise ValueError('Parameter[R] must be (3, N_px) real-valued.')

    if not (wl > 0):
        raise ValueError('Parameter[wl] must be positive.')

    if not (center.shape == (3,)):
        raise ValueError('Parameter[center] must be (3,) real-valued.')

    XYZ_centroid = np.mean(XYZ, axis=1, keepdims=True)
    XYZ_radius = np.mean(linalg.norm(XYZ - XYZ_centroid, axis=0))
    center = center / linalg.norm(center)

    psf = np.sinc((2 * XYZ_radius / wl) *
                  linalg.norm(R - center.reshape(3, 1), axis=0))
    psf_mag2 = psf ** 2
    return psf_mag2
