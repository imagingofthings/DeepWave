# ############################################################################
# instrument.py
# =============
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Array Geometries.
"""

import numpy as np
import scipy.linalg as linalg

import imot_tools.math.sphere.transform as transform


def pyramic_geometry():
    """
    `Pyramic <https://github.com/LCAV/Pyramic>`_ 3D microphone array geometry.

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian coordinates.
    """
    x = 0.27 + 2 * 0.015  # length of one side
    c1, c2, c3, c4 = 1 / np.sqrt(3), np.sqrt(2 / 3), np.sqrt(3) / 6, 0.5
    corners = np.array([[0, x * c1, -x * c3, -x * c3], [0, 0, x * c4, -x * c4],
                        [0, x * c2, x * c2, x * c2]])

    # Relative placement of microphones on one PCB.
    pcb = np.r_[-0.100, -0.060, -0.020, -0.004, 0.004, 0.020, 0.060, 0.100]

    def line(p1, p2, dist):
        center = (p1 + p2) / 2.
        unit_vec = (p2 - p1) / linalg.norm(p2 - p1)

        pts = [center + d * unit_vec for d in dist]
        return pts

    coordinates = np.array(
        line(corners[:, 0], corners[:, 3], pcb) +
        line(corners[:, 3], corners[:, 2], pcb) +
        line(corners[:, 0], corners[:, 1], pcb) +
        line(corners[:, 1], corners[:, 3], pcb) +
        line(corners[:, 0], corners[:, 2], pcb) +
        line(corners[:, 2], corners[:, 1], pcb))

    # Reference point is 1cm below zero-th microphone
    coordinates[:, 2] += 0.01 - coordinates[0, 2]
    return coordinates.T


def spherical_geometry():
    """
    Spherical 3D microphone array geometry.

    Radius: 0.20[m]

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian coordinates.
    """
    N_antenna = 64

    n = np.arange(N_antenna)
    colat = np.arccos(1 - (2 * n + 1) / N_antenna)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5))
    r = 0.2

    XYZ = np.stack(transform.pol2cart(r, colat, lon), axis=0)
    return XYZ
