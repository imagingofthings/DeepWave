# #############################################################################
# grid.py
# =======
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# #############################################################################

"""
Pixel-grid generation for spherical surfaces.
"""

import acoustic_camera.tools.math.sphere as sph
import numpy as np


def fibonacci_grid(order, direction=None, FoV=None):
    """
    Fibonacci pixel grid.

    Parameters
    ----------
    order : int
        Max SH order of representable plane waves.
    direction : :py:class:`~numpy.ndarray`
        (3,) vector around which the grid is centered.
    FoV : float
        Span of the grid centered at `direction` [rad].

    Returns
    -------
    :py:class:`~numpy.ndarray`
        if `direction == FoV == None`:
            (3, 4*(L+1)**2) pixel grid.
        else:
            (3, N_px) pixel grid limited to desired FoV.

    """
    if (FoV is None) and (direction is None):
        pass
    elif (FoV is not None) and (direction is not None):
        pass
    else:
        raise ValueError('Parameters[direction, FoV] must be simultaneously None or non-None.')

    if (FoV is not None) and (not (0 < FoV < 2 * np.pi)):
        raise ValueError('Parameter[FoV] must lie in (0, 360) degrees.')
    if order < 0:
        raise ValueError('Parameter[order] must be non-negative.')

    N_px = 4 * (order + 1) ** 2

    n = np.arange(N_px)
    colat = np.arccos(1 - (2 * n + 1) / N_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5))
    XYZ = np.stack(sph.pol2cart(1, colat, lon), axis=0)

    # Filtering to FoV
    if (direction is not None) and (FoV is not None):
        min_similarity = np.cos(FoV / 2)
        mask = (direction @ XYZ) >= min_similarity
        XYZ = XYZ[:, mask]
    return XYZ


def thin_grid(R, min_dist):
    """
    Return a subset of pixels that lie at least `min_dist` apart.

    Parameters
    ----------
    R : :py:class:`~numpy.ndarray`
        (3, N_px)
    min_dist : float
        Minimum angular distance [rad] between any pixel and its neighbors.

    Returns
    -------
    R_thin : :py:class:`~numpy.ndarray`
        (3, N_px2)
    """
    if not ((R.ndim == 2) and (R.shape[0] == 3)):
        raise ValueError('Parameter[R] must be (3, N_px) real-valued.')
    N_px = R.shape[1]

    px_idx = [np.argmax(np.mean(R, axis=1) @ R)]
    while True:
        idx_available = np.setdiff1d(np.arange(N_px), px_idx)
        mask_available = np.all(R[:, px_idx].T @ R[:, idx_available] <
                                np.cos(min_dist), axis=0)
        if np.any(mask_available):
            r = R[:, idx_available][:, mask_available][:, 0]
            px_idx.append(np.argmax(r @ R))
        else:
            break

    R_thin = R[:, px_idx]
    return R_thin
