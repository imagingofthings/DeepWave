# ############################################################################
# func.py
# =======
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Various mathematical functions.
"""

import warnings

import numpy as np


def relu(x):
    """
    Rectified Linear Unit.
    """
    return np.fmax(0, x)


def d_relu(x):
    """
    Rectified Linear Unit derivative.
    """
    y = np.zeros_like(x)
    y[x > 0] = 1
    return y


def retanh(alpha, x):
    r"""
    Rectified Hyperbolic Tangent.

    :math: f(x) = (\alpha / \tanh(1)) \tanh(x / \alpha) \bbOne{x > 0}
    """
    beta = alpha / np.tanh(1)
    return np.fmax(0, beta * np.tanh(x / alpha))


def d_retanh(alpha, x):
    r"""
    Rectified Hyperbolic Tangent derivative.

    :math: f'(x) = (1 / \tanh(1)) \bigCurly{1 - \tanh^{2}(x / \alpha)} \bbOne{x > 0}
    """
    y = np.zeros_like(x)
    mask = x > 0
    beta = alpha / np.tanh(1)
    y[mask] = (beta / alpha) * (1 - np.tanh(x[mask] / alpha) ** 2)
    return y
