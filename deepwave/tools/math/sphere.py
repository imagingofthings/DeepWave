# ############################################################################
# sphere.py
# =========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Spherical Geometry Tools.
"""

import acoustic_camera.tools.math.func as func
import astropy.coordinates as coord
import astropy.units as u
import numpy as np


def pol2eq(colat):
    """
    Polar coordinates to Equatorial coordinates.

    Parameters
    ----------
    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].

    Returns
    -------
    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].
    """
    lat = (np.pi / 2) - colat
    return lat


def eq2pol(lat):
    """
    Equatorial coordinates to Polar coordinates.

    Parameters
    ----------
    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].

    Returns
    -------
    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].
    """
    colat = (np.pi / 2) - lat
    return colat


def eq2cart(r, lat, lon):
    """
    Equatorial coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : :py:class:`~numpy.ndarray`
        Radius [m].
    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].
    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, ...) Cartesian XYZ coordinates.
    """
    if np.any(r < 0):
        raise ValueError("Parameter[r] must be non-negative.")

    XYZ = (coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
           .to_cartesian().xyz.to_value(u.dimensionless_unscaled))

    return XYZ


def pol2cart(r, colat, lon):
    """
    Polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : :py:class:`~numpy.ndarray`
        Radius [m].
    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].
    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, ...) Cartesian XYZ coordinates.
    """
    lat = (np.pi / 2) - colat
    return eq2cart(r, lat, lon)


def cart2pol(x, y, z):
    """
    Cartesian coordinates to Polar coordinates.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        X coordinate [m].
    y : :py:class:`~numpy.ndarray`
        Y coordinate [m].
    z : :py:class:`~numpy.ndarray`
        Z coordinate [m].

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius [m].

    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].

    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].
    """
    cart = coord.CartesianRepresentation(x, y, z)
    sph = coord.SphericalRepresentation.from_cartesian(cart)

    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to_value(u.rad)
    lon = u.Quantity(sph.lon).to_value(u.rad)

    return r, colat, lon


def cart2eq(x, y, z):
    """
    Cartesian coordinates to Equatorial coordinates.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        X coordinate [m].
    y : :py:class:`~numpy.ndarray`
        Y coordinate [m].
    z : :py:class:`~numpy.ndarray`
        Z coordinate [m].

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius [m].

    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].

    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].
    """
    r, colat, lon = cart2pol(x, y, z)
    lat = (np.pi / 2) - colat
    return r, lat, lon


class Interpolator:
    r"""
    Interpolate order-limited zonal function from spatial samples.

    Computes :math:`f(r) = \sum_{q} \alpha_{q} f(r_{q}) K_{N}(\langle r, r_{q} \rangle)`,
    where :math:`r_{q} \in \mathbb{S}^{2}` are points from a spatial sampling
    scheme, :math:`K_{N}(\cdot)` is the spherical Dirichlet kernel of order
    :math:`N`, and the :math:`\alpha_{q}` are scaling factors tailored to the
    sampling scheme.
    """

    def __init__(self, N, approximate_kernel=False):
        r"""
        Parameters
        ----------
        N : int
            Order of the reconstructed zonal function.
        approximate_kernel : bool
            If :py:obj:`True`, pass the `approx` option to :py:class:`~pypeline.util.math.func.SphericalDirichlet`.
        """
        super().__init__()

        if not (N > 0):
            raise ValueError('Parameter[N] must be positive.')
        self._N = N
        self._kernel_func = func.SphericalDirichlet(N, approximate_kernel)

    def __call__(self, weight, support, f, r, sparsity_mask=None):
        """
        Interpolate function samples at order `N`.

        Parameters
        ----------
        weight : :py:class:`~numpy.ndarray`
            (N_s,) weights to apply per support point.
        support : :py:class:`~numpy.ndarray`
            (3, N_s) critical support points.
        f : :py:class:`~numpy.ndarray`
            (L, N_s) zonal function values at support points. (float or complex)
        r : :py:class:`~numpy.ndarray`
            (3, N_px) evaluation points.
        sparsity_mask : :py:class:`~scipy.sparse.spmatrix`
            (N_s, N_px) sparsity mask to perform localized kernel evaluation.

        Returns
        -------
        f_interp : :py:class:`~numpy.ndarray`
            (L, N_px) function values at specified coordinates.
        """
        if not (weight.shape == (weight.size,)):
            raise ValueError('Parameter[weight] must have shape (N_s,).')
        N_s = weight.size

        if not (support.shape == (3, N_s)):
            raise ValueError('Parameter[support] must have shape (3, N_s).')

        L = len(f)
        if not (f.shape == (L, N_s)):
            raise ValueError('Parameter[f] must have shape (L, N_s).')

        if not ((r.ndim == 2) and (r.shape[0] == 3)):
            raise ValueError('Parameter[r] must have shape (3, N_px).')
        N_px = r.shape[1]

        if sparsity_mask is not None:
            if not (sparsity_mask.shape == (N_s, N_px)):
                raise ValueError('Parameter[sparsity_mask] must have shape (N_s, N_px).')

        if sparsity_mask is None:  # Dense evaluation
            similarity = np.clip(support.T @ r, -1, 1)
            kernel = self._kernel_func(similarity)
            beta = f * weight
            f_interp = beta @ kernel
        else:  # Sparse evaluation
            raise NotImplementedError
            # kernel = sp.csc_matrix(sparsity_mask.tocsc(), dtype=np.float)  # (N_px, N_s) CSC
            # for i in range(N_s):
            #     kernel[:, i] = self._kernel_func()  # compute sparse kernel
            # beta = (f * weight).T
            # f_interp = kernel.dot(beta).T

        return f_interp
