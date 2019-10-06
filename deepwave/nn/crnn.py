# ############################################################################
# crnn.py
# =======
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
RNN architecture based on graph convolutions.
"""

import acoustic_camera.nn as nn
import acoustic_camera.tools.instrument as instrument
import acoustic_camera.tools.math.func as func
import acoustic_camera.tools.math.graph as graph
import acoustic_camera.tools.math.optim as optim
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sp


class Parameter(optim.Parameter):
    """
    Serializer to encode/decode RNN parameters.
    """

    def __init__(self, N_antenna, N_px, K):
        """
        Parameters
        ----------
        N_antenna : int
        N_px : int
        K : int
        """
        super().__init__()

        self._N_antenna = N_antenna
        self._N_px = N_px
        self._K = K
        self._N_cell = (self._K + 1) + self._N_px * (2 * self._N_antenna + 1)

    def encode(self, buffer=None, mu=None, D=None, tau=None):
        """
        Encode parameter information in buffer.

        Parameters
        ----------
        buffer : :py:class:`~numpy.ndarray`
            (N_cell,) buffer in which to write the data.
            If `None`, a new buffer will be allocated.
        mu : :py:class:`~numpy.ndarray`
            (K + 1,)
            If `None`, the buffer is not modified at the intended location.
        D : :py:class:`~numpy.ndarray`
            (N_antenna, N_px)
            If `None`, the buffer is not modified at the intended location.
        tau : :py:class:`~numpy.ndarray`
            (N_px,)
            If `None`, the buffer is not modified at the intended location.

        Returns
        -------
        enc : :py:class:`~numpy.ndarray`
            (N_cell,) vectorized encoding.
        """
        if buffer is None:
            enc = np.zeros((self._N_cell,))
        else:
            if not (buffer.shape == (self._N_cell,)):
                raise ValueError('Parameter[buffer] must be (N_cell,) real-valued.')
            enc = buffer
        N_start = 0

        N_fill = self._K + 1
        if mu is not None:
            if not (mu.shape == (self._K + 1,)):
                raise ValueError('Parameter[mu] must be (K + 1,) real-valued.')
            enc[N_start:(N_start + N_fill)] = mu
        N_start += N_fill

        N_fill = 2 * self._N_antenna * self._N_px
        if D is not None:
            if not (D.shape == (self._N_antenna, self._N_px)):
                raise ValueError('Parameter[D] must be (N_antenna, N_px) complex-valued.')
            enc[N_start:(N_start + N_fill)] = (np.ascontiguousarray(D, dtype=np.complex128)
                                               .reshape(-1)
                                               .view(dtype=np.float64))
        N_start += N_fill

        N_fill = self._N_px
        if tau is not None:
            if not (tau.shape == (self._N_px,)):
                raise ValueError('Parameter[tau] must be (N_px,) real-valued.')
            enc[N_start:(N_start + N_fill)] = tau
        N_start += N_fill

        return enc

    def decode(self, enc, keepdims=False):
        """
        Decode parameter information from buffer.

        Parameters
        ----------
        enc : :py:class:`~numpy.ndarray`
            ([N_sample], N_cell,) vectorized encoding, output of
            :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.
        keepdims : bool
            If `True` and `enc.ndim == 1', then the `1`-sized leading dimension
            of the outputs is dropped.

        Returns
        -------
        mu : :py:class:`~numpy.ndarray`
            ([N_sample], K + 1)
        D : :py:class:`~numpy.ndarray`
            ([N_sample], N_antenna, N_px)
        tau : :py:class:`~numpy.ndarray`
            ([N_sample], N_px)
        """
        was_1d = (enc.ndim == 1)
        if was_1d:
            enc = enc.reshape(1, -1)
        N_sample = len(enc)

        if not (enc.shape == (N_sample, self._N_cell)):
            raise ValueError('Parameter[enc] must be ([N_sample], N_cell) real-valued.')
        N_start = 0

        N_fill = self._K + 1
        mu = enc[:, N_start:(N_start + N_fill)]
        N_start += N_fill

        N_fill = 2 * self._N_antenna * self._N_px
        D = (np.ascontiguousarray(enc[:, N_start:(N_start + N_fill)])
             .view(np.complex128)
             .reshape(N_sample, self._N_antenna, self._N_px))
        N_start += N_fill

        N_fill = self._N_px
        tau = enc[:, N_start:(N_start + N_fill)]
        N_start += N_fill

        if (not keepdims) and was_1d:
            mu = mu.squeeze(axis=0)
            D = D.squeeze(axis=0)
            tau = tau.squeeze(axis=0)
        return mu, D, tau


class SampleLossFunction(optim.ScalarFunction):
    r"""
    Proxy object to evaluate loss function

    f(p, x) = \cL^{t}(p, x) = 0.5 * \frac{\norm{\hat{I}_{APGD} - \hat{I}_{CRNN}}{2}^{2}}{\delta}
    OR
    f(p, x) = \cL^{t}(p, x) = (\delta + \hat{I}_{APGD})^{T}
                            * \log\bigParen{\frac{\delta + \hat{I}_{APGD}}{\delta + \hat{I}_{CRNN}}}
                            - 1^{T}\bigParen{\hat{I}_{APGD} - \hat{I}_{CRNN}},
    where \delta = \norm{\hat{I}_{APGD}}{2}^{2}
    """

    def __init__(self, N_layer, p, s, Ln, loss='relative-l2', afunc=(func.relu, func.d_relu),
                 trainable_parameter=(('mu', True), ('D', True), ('tau', True))):
        r"""
        Parameters
        ----------
        N_layer : int
            Number of iterations `L` in RNN.
        p : :py:class:`~acoustic_camera.nn.crnn.Parameter`
            Serializer to encode/decode parameters.
        s : :py:class:`~acoustic_camera.nn.Sampler`
            Serializer to encode/decode samples.
        Ln : :py:class:`~scipy.sparse.csr_matrix`
            (N_px, N_px) normalized graph Laplacian.
        loss : str
            If 'relative-l2', use the relative squared error
                \eps = (1/2) * \norm{\hat{x} - x^{L}}{2}^{2} / \norm{\hat{x}}{2}^{2}
            If 'shifted-kl', use the generalized shifted Kullback-Leibler divergence
                \eps = (1 + \hat{x})^{T} \log\bigParen{\frac{1 + \hat{x}}{1 + x^{L}}}
                     - 1^{T}\bigParen{\hat{x} - x^{L}}
        afunc : tuple(function)
            (activation function, activation function derivative)
        trainable_parameter : tuple(tuple(str, bool))
            Tuple of (str, bool) pairs that state whether the corresponding parameters should have a gradient or not.
        """
        super().__init__()

        if N_layer < 1:
            raise ValueError('Parameter[N_layer] must be positive.')
        self._N_layer = N_layer

        if not isinstance(p, Parameter):
            raise ValueError('Parameter[p]: expected acoustic_camera.nn.crnn.Parameter')
        self._p = p

        if not isinstance(s, nn.Sampler):
            raise ValueError('Parameter[s]: expected acoustic_camera.nn.Sampler')
        self._s = s

        N_px = self._s._N_px
        if not (isinstance(Ln, sp.csr_matrix) and
                (Ln.shape == (N_px, N_px))):
            raise ValueError('Parameter[Ln] must be (N_px, N_px) CSR.')
        self._h = graph.ConvolutionalFilter(Ln, self._p._K)

        if loss == 'relative-l2':
            self._use_l2 = True
        elif loss == 'shifted-kl':
            self._use_l2 = False
        else:
            raise ValueError('Parameter[loss] must be one of {"relative-l2", "shifted-kl"}.')

        if not (isinstance(afunc, tuple) and (len(afunc) == 2)):
            raise ValueError('Parameter[afunc]: expected (function, function)')
        self._afunc = afunc[0]
        self._afunc_d = afunc[1]

        param_msg = ('Parameter[trainable_parameter] must take form '
                     "(('mu', T/F), ('D', T/F), ('tau', T/F)).")
        if not (isinstance(trainable_parameter, tuple) and
                (len(trainable_parameter) == 3) and
                all([len(p) == 2 for p in trainable_parameter])):
            raise ValueError(param_msg)
        self._param = dict(trainable_parameter)
        if not ((set(self._param.keys()) == {'mu', 'D', 'tau'}) and
                (set(self._param.values()) <= {True, False})):
            raise ValueError(param_msg)

        # Buffer of intermediate values for grad().
        # Will always have shape (N_sample, N_layer, N_px)
        self._tape_buffer = None

        # Buffer (p, x) variables used in eval(). If the same ones are used in
        # grad() afterwards, we can skip re-computation of eval() during
        # grad().
        self._tape_p = None
        self._tape_x = None

    def eval(self, p, x):
        r"""
        Evaluate f(p, x).

        Parameters
        ----------
        p : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter encoding, output of
            :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.
        x : :py:class:`~numpy.ndarray`
            ([N_sample,], N_cell_2) vectorized sample encoding, output of
            :py:meth:`~acoustic_camera.nn.Sampler.encode`.

            Several samples can be provided if stacked along axis 0.

        Returns
        -------
        z : float
            z = \frac{1}{N_sample} \sum_{i = 1}^{N_sample} f(p, x[i])
        """
        mu, D, tau = self._p.decode(p)
        S, I, I_prev = self._s.decode(x, keepdims=True)
        N_sample, N_px = S.shape[0], tau.shape[0]

        y = np.zeros((N_sample, N_px))
        for i in range(N_sample):  # Broadcasting solution is slower
            Ds, Vs = linalg.eigh(S[i])
            idx = Ds > 0  # To avoid np.sqrt() issues.
            Ds, Vs = Ds[idx], Vs[:, idx]
            y[i] = linalg.norm(D.conj().T @ (Vs * np.sqrt(Ds)), axis=1) ** 2
        y -= tau

        s = np.zeros((N_sample, self._N_layer, N_px))
        xx = I_prev
        for l in range(self._N_layer):
            s[:, l] = self._h.filter(mu, xx) + y
            xx = self._afunc(s[:, l])

        self._tape_buffer = s
        self._tape_p = p
        self._tape_x = x

        if self._use_l2:
            z = 0.5 * np.sum((I - xx) ** 2, axis=1) / np.sum(I ** 2, axis=1)
        else:  # shifted-kl
            delta = linalg.norm(I, axis=1, keepdims=True) ** 2
            z = (((np.reshape(delta + I, (N_sample, 1, N_px)) @
                   np.reshape(np.log((delta + I) / (delta + xx)), (N_sample, N_px, 1)))
                  .squeeze(axis=(1, 2))) -
                 np.sum(I - xx, axis=1))
        z = np.sum(z) / N_sample
        return z

    def grad(self, p, x):
        r"""
        Evaluate \grad_{p} f(p, x).

        Parameters
        ----------
        p : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter encoding, output of
            :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.
        x : :py:class:`~numpy.ndarray`
            ([N_sample,], N_cell_2) vectorized sample encoding, output of
            :py:meth:`~acoustic_camera.nn.Sampler.encode`.

            Several samples can be provided if stacked along axis 0.

        Returns
        -------
        z : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter gradient, output of
            :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.

            z = \frac{1}{N_sample} \sum_{i=1}^{N_sample} \grad_{p}{f(p, x[i])}
        """
        if not ((p is self._tape_p) and
                (x is self._tape_x)):
            self.eval(p, x)  # To save intermediate values
        s = self._tape_buffer  # (N_sample, N_layer, N_px)

        mu, D, tau = self._p.decode(p)
        S, I, I_prev = self._s.decode(x, keepdims=True)
        N_antenna, N_px = D.shape
        N_sample = S.shape[0]
        K = mu.shape[0]

        if self._use_l2:
            dx = ((self._afunc(s[:, self._N_layer - 1]) - I) /
                  np.sum(I ** 2, axis=1, keepdims=True))
        else:  # shifted-kl
            delta = linalg.norm(I, axis=1, keepdims=True) ** 2
            dx = (np.r_[1] -
                  ((delta + I) / (delta + self._afunc(s[:, self._N_layer - 1]))))
        dmu = np.zeros((N_sample, K))
        dtau = np.zeros((N_sample, N_px))

        for l in np.arange(self._N_layer)[::-1]:
            ds = self._afunc_d(s[:, l]) * dx
            dx = self._h.filter(mu, ds)
            dtau -= ds

            if l > 0:
                a = self._afunc(s[:, l - 1])
            else:
                a = I_prev
            dmu += self._h.trace(a, ds)

        dD = np.zeros((N_antenna, N_px), dtype=np.complex)
        for i in range(N_sample):  # Broadcasting solution is slower.
            dD += (S[i] @ D) * (-2 * dtau[i])
        dmu = np.sum(dmu, axis=0)
        dtau = np.sum(dtau, axis=0)

        if not self._param['mu']:
            dmu[:] = 0
        if not self._param['D']:
            dD[:] = 0
        if not self._param['tau']:
            dtau[:] = 0

        z = self._p.encode(mu=dmu, D=dD, tau=dtau)
        z /= N_sample
        return z


class D_RidgeLossFunction(optim.ScalarFunction):
    r"""
    Proxy object to evaluate loss function.

    f(p) = \frac{\lambda_{1}}{2} \frac{\norm{\bbD}{F}^{2}}{N_{antenna} N_{px}}
    """

    def __init__(self, lambda_, p):
        r"""
        Parameters
        ----------
        lambda_ : float
            Regularization parameter \ge 0
        p : :py:class:`~acoustic_camera.nn.crnn.Parameter`
            Serializer to encode/decode parameters.
        """
        super().__init__()

        if not isinstance(p, Parameter):
            raise ValueError('Parameter[p]: expected acoustic_camera.nn.crnn.Parameter.')
        self._p = p

        if not (lambda_ >= 0):
            raise ValueError('Parameter[lambda_] must be non-negative.')
        self._lambda_ = lambda_

    def eval(self, p, x):
        r"""
        Evaluate f(p, x).

        Parameters
        ----------
        p : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter encoding, output of
            :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.
        x : :py:class:`~numpy.ndarray`
            ([N_sample,], N_cell_2) vectorized sample encoding, output of
            :py:meth:`~acoustic_camera.nn.Sampler.encode`.

            Several samples can be provided if stacked along axis 0.

            This variable is not used in the implementation, but is provided
            for interface consistency.

        Returns
        -------
        z : float
            z = f(p)
        """
        mu, D, tau = self._p.decode(p)
        N_antenna, N_px = D.shape

        z = ((0.5 * self._lambda_) *
             ((np.sum(D.real ** 2) + np.sum(D.imag ** 2)) / (N_antenna * N_px)))
        return z

    def grad(self, p, x):
        r"""
        Evaluate \grad_{p} f(p, x).

        Parameters
        ----------
        p : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter encoding, output of
            :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.
        x : :py:class:`~numpy.ndarray`
            ([N_sample,], N_cell_2) vectorized sample encoding, output of
            :py:meth:`~acoustic_camera.nn.Sampler.encode`.

            Several samples can be provided if stacked along axis 0.

            This variable is not used in the implementation, but is provided
            for interface consistency.

        Returns
        -------
        z : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter gradient, output of
            :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.

            z = \grad_{p}{f(p)}
        """
        mu, D, tau = self._p.decode(p)
        N_antenna, N_px = D.shape

        dD = D * (self._lambda_ / (N_antenna * N_px))

        z = self._p.encode(D=dD)
        return z


class LaplacianLossFunction(optim.ScalarFunction):
    r"""
    Proxy object to evaluate loss function.

    f(p) = \frac{\lambda_{2}}{2 N_{px}}\norm{\bbB^{1/2} \bbtau(p)}{2}^{2}
    """

    def __init__(self, B, lambda_, p):
        """
        Parameters
        ----------
        B : :py:class:`~scipy.sparse.csr_matrix`
            (N_px, N_px) Graph Laplacian.
        lambda_ : float
            Regularization parameter >= 0.
        p : :py:class:`~acoustic_camera.nn.crnn.Parameter`
            Serializer to encode/decode parameters.
        """
        super().__init__()

        if lambda_ < 0:
            raise ValueError('Parameter[lambda_] must be non-negative.')
        self._lambda_ = lambda_

        if not isinstance(B, sp.csr_matrix):
            raise ValueError('Parameter[B] must be (N_px, N_px) CSR.')
        self._B = B

        if not isinstance(p, Parameter):
            raise ValueError('Parameter[p]: expected acoustic_camera.nn.crnn.Parameter')
        self._p = p

    def eval(self, p, x):
        r"""
        Evaluate f(p, x).

        Parameters
        ----------
        p : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter encoding, output of
            :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.
        x : :py:class:`~numpy.ndarray`
            ([N_sample,], N_cell_2) vectorized sample encoding, output of
            :py:meth:`~acoustic_camera.nn.Sampler.encode`.

            Several samples can be provided if stacked along axis 0.

            This variable is not used in the implementation, but is provided
            for interface consistency.

        Returns
        -------
        z : float
            z = f(p)
        """
        _, _, tau = self._p.decode(p)
        N_px = tau.size

        z = ((0.5 * self._lambda_ / N_px) *
             (tau @ self._B.dot(tau)))
        return z

    def grad(self, p, x):
        r"""
        Evaluate \grad_{p} f(p, x).

        Parameters
        ----------
        p : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter encoding, output of
            :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.
        x : :py:class:`~numpy.ndarray`
            ([N_sample,], N_cell_2) vectorized sample encoding, output of
            :py:meth:`~acoustic_camera.nn.Sampler.encode`.

            Several samples can be provided if stacked along axis 0.

            This variable is not used in the implementation, but is provided
            for interface consistency.

        Returns
        -------
        z : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter gradient, output of
            :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.

            z = \grad_{p}{f(p)}
        """
        _, _, tau = self._p.decode(p)
        N_px = tau.size

        dtau = ((self._lambda_ / N_px) *
                (self._B.T).dot(tau))

        z = self._p.encode(tau=dtau)
        return z


def APGD_Parameter(XYZ, R, wl, lambda_, gamma, L, eps):
    r"""
    Theoretical values of mu, D, tau in APGD, used as initializer point for SGD.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian array geometry.
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points.
    wl : float
        Wavelength \ge 0 [m]
    lambda_ : float
        Regularization parameter.
    gamma : float
        Linear trade-off between lasso and ridge regularizers.
    L : float
        Lipschitz constant from Remark 3.3
    eps : float
        PSF truncation coefficient for
        :py:method:`~acoustic_camera.tools.math.graph.ConvolutionalFilter.estimate_order`

    Returns
    -------
    p : :py:class:`~numpy.ndarray`
        (N_cell,) vectorized parameter value, output of
        :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.
    K : int
        Order of polynomial filter.
    """

    def e(i: int, N: int):
        v = np.zeros((N,))
        v[i] = 1
        return v

    A = instrument.steering_operator(XYZ, R, wl)
    N_antenna, N_px = A.shape
    alpha = 1 / L
    beta = 2 * lambda_ * alpha * (1 - gamma) + 1

    Ln, rho = graph.laplacian_exp(R, normalized=True)
    K = graph.ConvolutionalFilter.estimate_order(XYZ, rho, wl, eps)
    K *= 2  # Why?: just to be on the safe side.
    h = graph.ConvolutionalFilter(Ln, K)

    # Solve LSQ problem \sum_{k = 0}^{K} \mu_{k} T_{k}(\tilde{L}) =
    #                   \frac{I_{N} - 2 \alpha \abs{A^{H} A}^{2}}{beta}
    R_focus = np.mean(R, axis=1)
    R_focus /= linalg.norm(R_focus)
    idx = np.argmax(R_focus @ R)
    psf_mag2 = np.abs(A.conj().T @ A[:, idx]) ** 2
    c = (e(idx, N_px) - 2 * alpha * psf_mag2) / beta

    mu = h.fit(e(idx, N_px), c)
    D = A * np.sqrt(2 * alpha / beta)
    tau = np.ones((N_px,)) * (lambda_ * alpha * gamma / beta)

    parameter = Parameter(N_antenna, N_px, K)
    p = parameter.encode(None, mu, D, tau)
    return p, K


class Evaluator:
    """
    Fastest evaluation possible of CRNN forward pass.
    Use this class once training is done to get good runtime performance during inference.
    """

    def __init__(self, N_layer, p, p_opt, Ln, afunc=func.relu):
        """
        Parameters
        ----------
        N_layer : int
            Number of iterations `L` in RNN.
        p : :py:class:`~acoustic_camera.nn.crnn.Parameter`
            Serializer to encode/decode parameters.
        p_opt : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter encoding, output of
            :py:meth:`~acoustic_camera.nn.crnn.Parameter.encode`.
        Ln : :py:class:`~scipy.sparse.csr_matrix`
            (N_px, N_px) normalized graph Laplacian.
        afunc : function
            activation function
        """
        if N_layer < 1:
            raise ValueError('Parameter[N_layer] must be positive.')
        self._N_layer = N_layer

        if not isinstance(p, Parameter):
            raise ValueError('Parameter[p]: expected acoustic_camera.nn.crnn.Parameter')

        self._afunc = afunc

        mu, D, tau = p.decode(p_opt)
        self._mu = mu.copy()
        self._tau = tau.copy()
        self._h = graph.ConvolutionalFilter(Ln, p._K)
        self._D_conj_T = np.ascontiguousarray(D.conj().T)

    def __call__(self, S, I_prev):
        """
        Get RNN output.

        Parameters
        ----------
        S : :py:class:`~numpy.ndarray`
            (N_antenna, N_antenna) visibility matrix.
        I_prev : :py:class:`~numpy.ndarray`
            (N_px,) initialization point.

        Returns
        -------
        xx : :py:class:`~numpy.ndarray`
            (N_px,) intensity estimate
        """
        D, V = linalg.eigh(S)
        idx = D > 0  # To avoid np.sqrt() issues.
        D, V = D[idx], V[:, idx]

        alpha = linalg.norm(self._D_conj_T @ (V * np.sqrt(D)), axis=1) ** 2
        alpha -= self._tau

        xx = I_prev.copy()
        for l in range(self._N_layer):
            xx = self._afunc(self._h.filter(self._mu, xx) + alpha)

        return xx
