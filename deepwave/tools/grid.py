# #############################################################################
# grid.py
# =======
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# #############################################################################

"""
Pixel-grid generation for spherical surfaces.
"""

import numpy as np


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
