# #############################################################################
# linalg.py
# =========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# #############################################################################

"""
Linear algebra routines.
"""

import acoustic_camera.tools.instrument as instrument
import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg


def rot(axis, angle):
    """
    3D rotation matrix.

    Parameters
    ----------
    axis : :py:class:`~numpy.ndarray`
        (3,) rotation axis.
    angle : float
        Signed rotation angle [rad].

    Returns
    -------
    R : :py:class:`~numpy.ndarray`
        (3, 3) rotation matrix.
    """
    a, b, c = axis / linalg.norm(axis)
    ct, st = np.cos(angle), np.sin(angle)

    p00 = a ** 2 + (b ** 2 + c ** 2) * ct
    p11 = b ** 2 + (a ** 2 + c ** 2) * ct
    p22 = c ** 2 + (a ** 2 + b ** 2) * ct
    p01 = a * b * (1 - ct) - c * st
    p10 = a * b * (1 - ct) + c * st
    p12 = b * c * (1 - ct) - a * st
    p21 = b * c * (1 - ct) + a * st
    p20 = a * c * (1 - ct) - b * st
    p02 = a * c * (1 - ct) + b * st

    R = np.array([[p00, p01, p02], [p10, p11, p12], [p20, p21, p22]])
    return R


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

    A = instrument.steering_operator(XYZ, R, wl)
    d = instrument.steering_operator(XYZ, center.reshape(3, 1), wl)

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
