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
import scipy.interpolate as interpolate
import scipy.special as sp


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


class Tukey:
    r"""
    Parameterized Tukey function.

    Notes
    -----
    The Tukey function is defined as:

    .. math::

       \text{Tukey}(T, \beta, \alpha)(\varphi): \mathbb{R} & \to [0, 1] \\
       \varphi & \to
       \begin{cases}
           % LINE 1
           \sin^{2} \left( \frac{\pi}{T \alpha}
                    \left[ \frac{T}{2} - \beta + \varphi \right] \right) &
           0 \le \frac{T}{2} - \beta + \varphi < \frac{T \alpha}{2} \\
           % LINE 2
           1 &
           \frac{T \alpha}{2} \le \frac{T}{2} - \beta +
           \varphi \le T - \frac{T \alpha}{2} \\
           % LINE 3
           \sin^{2} \left( \frac{\pi}{T \alpha}
                    \left[ \frac{T}{2} + \beta - \varphi \right] \right) &
           T - \frac{T \alpha}{2} < \frac{T}{2} - \beta + \varphi \le T \\
           % LINE 4
           0 &
           \text{otherwise.}
       \end{cases}
    """

    def __init__(self, T, beta, alpha):
        """
        Parameters
        ----------
        T : float
            Function support.
        beta : float
            Function mid-point.
        alpha : float
           Normalized decay-rate.
        """
        self._beta = beta

        if not (T > 0):
            raise ValueError('Parameter[T] must be positive.')
        self._T = T

        if not (0 <= alpha <= 1):
            raise ValueError('Parameter[alpha] must be in [0, 1].')
        self._alpha = alpha

    def __call__(self, x):
        """
        Sample the Tukey(T, beta, alpha) function.

        Parameters
        ----------
        x : float or array-like(float)
            Sample points.

        Returns
        -------
        Tukey(T, beta, alpha)(x) : :py:class:`~numpy.ndarray`
        """
        x = np.array(x, copy=False)

        y = x - self._beta + self._T / 2
        left_lim = float(self._T * self._alpha / 2)
        right_lim = float(self._T - (self._T * self._alpha / 2))

        ramp_up = (0 <= y) & (y < left_lim)
        body = (left_lim <= y) & (y <= right_lim)
        ramp_down = (right_lim < y) & (y <= self._T)

        amplitude = np.zeros_like(x)
        amplitude[body] = 1
        if not np.isclose(self._alpha, 0):
            amplitude[ramp_up] = np.sin(np.pi / (self._T * self._alpha) * y[ramp_up]) ** 2
            amplitude[ramp_down] = np.sin(np.pi / (self._T * self._alpha) * (self._T - y[ramp_down])) ** 2
        return amplitude


class SphericalDirichlet:
    r"""
    Parameterized spherical Dirichlet kernel.

    Notes
    -----
    The spherical Dirichlet function :math:`K_{N}(t): [-1, 1] \to \mathbb{R}` is defined as:

    .. math:: K_{N}(t) = \frac{P_{N+1}(t) - P_{N}(t)}{(N + 1)(t - 1)},

    where :math:`P_{N}(t)` is the `Legendre polynomial <https://en.wikipedia.org/wiki/Legendre_polynomials>`_ of order :math:`N`.
    """

    def __init__(self, N, approx=False):
        """
        Parameters
        ----------
        N : int
            Kernel order.
        approx : bool
            Approximate kernel using cubic-splines.

            This method provides extremely reliable estimates of :math:`K_{N}(t)` in the
            vicinity of 1 where the function's main sidelobes are found.
            Values outside the vicinity smoothly converge to 0.

            Only works for `N` greater than 10.
        """
        if N < 0:
            raise ValueError("Parameter[N] must be non-negative.")
        self._N = N

        if (approx is True) and (N <= 10):
            raise ValueError('Cannot use approximation method if Parameter[N] <= 10.')
        self._approx = approx

        if approx is True:  # Fit cubic-spline interpolator.
            N_samples = 10 ** 3

            # Find interval LHS after which samples will be evaluated exactly.
            theta_max = np.pi
            while True:
                x = np.linspace(0, theta_max, N_samples)
                cx = np.cos(x)
                cy = self._exact_kernel(cx)
                zero_cross = np.diff(np.sign(cy))
                N_cross = np.abs(np.sign(zero_cross)).sum()

                if N_cross > 10:
                    theta_max /= 2
                else:
                    break

            window = Tukey(T=2 - 2 * np.cos(2 * theta_max), beta=1, alpha=0.5)

            x = np.r_[np.linspace(np.cos(theta_max * 2), np.cos(theta_max), N_samples, endpoint=False),
                      np.linspace(np.cos(theta_max), 1, N_samples)]
            y = self._exact_kernel(x) * window(x)
            self.__cs_interp = interpolate.interp1d(x, y, kind='cubic', bounds_error=False, fill_value=0)

            # Store zero_threshold to simplify optimizations in :py:class:`~pypeline.util.math.sphere.Interpolator`
            self._zero_threshold = x[0]

    def __call__(self, x):
        r"""
        Sample the order-N spherical Dirichlet kernel.

        Parameters
        ----------
        x : float or array-like(float)
            Values at which to compute :math:`K_{N}(x)`.

        Returns
        -------
        K_N(x) : :py:class:`~numpy.ndarray`
        """
        x = np.array(x, copy=False, dtype=float)
        if x.ndim == 0:  # scalar input
            x = x.reshape(1)

        if not np.all((-1 <= x) & (x <= 1)):
            raise ValueError('Parameter[x] must lie in [-1, 1].')

        if self._approx is True:
            f = self._approx_kernel
        else:
            f = self._exact_kernel

        amplitude = f(x) / (self._N + 1)
        return amplitude

    def _exact_kernel(self, x):
        amplitude = sp.eval_legendre(self._N + 1, x) - sp.eval_legendre(self._N, x)
        with warnings.catch_warnings():
            # The kernel is so condensed near 1 at high N that np.isclose()
            # does a terrible job at letting us manually treat values close to
            # the upper limit.
            # The best way to implement K_N(t) is to let the floating point
            # division fail and then replace NaNs.
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            amplitude /= x - 1
        amplitude[np.isnan(amplitude)] = self._N + 1

        return amplitude

    def _approx_kernel(self, x):
        amplitude = self.__cs_interp(x)
        return amplitude
