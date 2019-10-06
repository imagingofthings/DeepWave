# #############################################################################
# source.py
# =========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# #############################################################################

"""
Source model.
"""

import acoustic_camera.tools.math.linalg as pylinalg
import acoustic_camera.tools.math.sphere as sph
import numpy as np
import scipy.linalg as linalg


class SkyModel:
    """
    Container to store source positions and amplitudes.
    """

    def __init__(self, XYZ, I):
        """
        Parameters
        ----------
        XYZ : :py:class:`~numpy.ndarray`
            (3, N_source) source directions.
        I : :py:class:`~numpy.ndarray`
            (N_source,) source intensities >= 0.
        """
        if XYZ.shape[1] != len(I):
            raise ValueError('Dimension mismatch between XYZ and I.')

        if np.any(I < 0):
            raise ValueError('Intensities must be non-negative.')

        self._xyz = XYZ / linalg.norm(XYZ, axis=0)
        self._intensity = I

    @property
    def xyz(self):
        return self._xyz

    @property
    def intensity(self):
        return self._intensity

    def encode(self):
        """
        Encode object into buffer.

        Returns
        -------
        enc : :py:class:`~numpy.ndarray`
            (N_data,) vectorized encoding.
        """
        Q = self._intensity.shape[0]

        N_data = 1 + Q + 3 * Q
        enc = np.zeros((N_data,))

        enc[0] = Q
        enc[1:(1 + Q)] = self._intensity
        enc[(1 + Q):] = self._xyz.reshape(-1)

        return enc

    @classmethod
    def decode(cls, enc):
        """
        Decode object.

        Parameters
        ----------
        enc : :py:class:`~numpy.ndarray`
            (N_data,) vectorized encoding.

        Returns
        -------
        sky : :py:class:`~acoustic_camera.tools.data_gen.source.SkyModel`
        """
        Q = int(enc[0])
        N_data = len(enc)
        N_data_required = 1 + Q + (3 * Q)
        if N_data != N_data_required:
            raise ValueError('Parameter[enc] is ill-formed.')

        I = enc[1:(1 + Q)]
        XYZ = enc[(1 + Q):].reshape(3, Q)

        sky = cls(XYZ, I)
        return sky


def from_circular_distribution(direction, FoV, N_src):
    """
    Distribute `N_src` sources on a circle centered at `direction`.

    Parameters
    ----------
    direction : :py:class:`~numpy.ndarray`
        (3,) direction in the sky.
    FoV : float
        Spherical angle [rad] of the sky centered at `direction` from which sources are extracted.
    N_src : int
        Number of dominant sources to extract.

    Returns
    -------
    sky_model : :py:class:`~tools.data_gen.source.SkyModel`
        Sky model.
    """
    if not (0 < FoV < 2 * np.pi):
        raise ValueError('Parameter[FoV] must lie in (0, 360) degrees.')

    colat = FoV / 4
    lon = np.linspace(0, 2 * np.pi, N_src, endpoint=False)
    XYZ = np.stack(sph.pol2cart(1, colat, lon), axis=0)

    # Center grid at 'direction'
    _, dir_colat, _ = sph.cart2pol(*direction)
    R_axis = np.cross([0, 0, 1], direction)
    if np.allclose(R_axis, 0):
        # R_axis is in span(E_z), so we must manually set R
        R = np.eye(3)
        if direction[2] < 0:
            R[2, 2] = -1
    else:
        R = pylinalg.rot(axis=R_axis, angle=dir_colat)

    XYZ = np.tensordot(R, XYZ, axes=1)
    I = np.ones((N_src,))
    sky_model = SkyModel(XYZ, I)
    return sky_model
