# ############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Neural Network Tools.
"""

import acoustic_camera.tools.data_gen.source as source
import acoustic_camera.tools.math.optim as optim
import numpy as np


class DataSet(optim.DataSet):
    """
    Dataset of samples for NN.
    """

    def __init__(self, data, XYZ, R, wl, ground_truth, lambda_, gamma, N_iter, tts):
        r"""
        Parameters
        ----------
        data : :py:class:`~numpy.ndarray`
            (N_sample, N_data) encoded samples. (real-valued)
        XYZ : :py:class:`~numpy.ndarray`
            (3, N_antenna) Cartesian array geometry.
        R : :py:class:`~numpy.ndarray`
            (3, N_px) Cartesian grid points.
        wl : float
            Wavelength >= 0 [m].
        ground_truth : list(:py:class:`~acoustic_camera.tools.data_gen.source.SkyModel`)
            (N_sample,) ground truth sky models. (Restricted to the region of interest.)
        lambda_ : :py:class:`~numpy.ndarray`
            (N_sample,) APGD regularization parameters.
        gamma : float
            Linear trade-off between lasso and ridge regularizers.
        N_iter : :py:class:`~numpy.ndarray`
            (N_sample,) APGD iterations required to reach convergence.
        tts : :py:class:`~numpy.ndarray`
            (N_sample,) time [s] required for APGD solution to converge.
        """
        super().__init__(data=data)

        if not ((XYZ.ndim == 2) and (XYZ.shape[0] == 3)):
            raise ValueError('Parameter[XYZ] must be (3, N_antenna) real-valued.')
        N_antenna = XYZ.shape[1]
        self._XYZ = XYZ

        if not ((R.ndim == 2) and (R.shape[0] == 3)):
            raise ValueError('Parameter[R] must be (3, N_px) real-valued.')
        N_px = R.shape[1]
        self._R = R

        if wl < 0:
            raise ValueError('Parameter[wl] must be positive.')
        self._wl = wl

        N_sample = len(self)
        if not (len(ground_truth) == N_sample):
            raise ValueError('Parameter[ground_truth] must be (N_sample,) SkyModels.')
        self._ground_truth = ground_truth

        if not ((lambda_.shape == (N_sample,)) and np.all(lambda_ >= 0)):
            raise ValueError('Parameter[lambda_] must be (N_sample,) non-negative.')
        self._lambda_ = lambda_

        if not (isinstance(gamma, float) and (0 <= gamma <= 1)):
            raise ValueError('Parameter[gamma] must be in [0, 1].')
        self._gamma = gamma

        if not ((N_iter.shape == (N_sample,)) and np.all(N_iter > 0)):
            raise ValueError('Parameter[N_iter] must be (N_sample,) positive.')
        self._N_iter = N_iter

        if not ((tts.shape == (N_sample,)) and np.all(tts > 0)):
            raise ValueError('Parameter[tts] must be (N_sample,) positive.')
        self._tts = tts

    @property
    def XYZ(self):
        """
        Returns
        -------
        XYZ : :py:class:`~numpy.ndarray`
            (3, N_antenna) Cartesian array geometry.
        """
        return self._XYZ

    @property
    def R(self):
        """
        Returns
        -------
        R : :py:class:`~numpy.ndarray`
            (3, N_px) Cartesian grid points.
        """
        return self._R

    @property
    def wl(self):
        """
        Returns
        -------
        wl : float
            Wavelength [m].
        """
        return self._wl

    @property
    def ground_truth(self):
        """
        Returns
        -------
        model : list(:py:class:`~acoustic_camera.tools.data_gen.source.SkyModel`)
            (N_sample,) ground truth sky models.
        """
        model = self._ground_truth
        return model

    @property
    def lambda_(self):
        """
        Returns
        -------
        lambda_ : :py:class:`~numpy.ndarray`
            (N_sample,) APGD regularization parameters.
        """
        lambda_ = self._lambda_
        return lambda_

    @property
    def gamma(self):
        """
        Returns
        -------
        gamma : float
            Linear trade-off between lasso and ridge regularizers.
        """
        gamma = self._gamma
        return gamma

    @property
    def N_iter(self):
        """
        Returns
        -------
        N_iter : :py:class:`~numpy.ndarray`
            (N_sample,) APGD iterations required to reach convergence.
        """
        N_iter = self._N_iter
        return N_iter

    @property
    def tts(self):
        """
        Returns
        -------
        tts : :py:class:`~numpy.ndarray`
            (N_sample,) time [s] required for APGD solution to converge.
        """
        tts = self._tts
        return tts

    def sampler(self):
        """
        Returns
        -------
        s : :py:class:`~acoustic_camera.nn.Sampler`
            Serializer used to read the encoded samples provided by
            :py:meth:`~acoustic_camera.nn.DataSet.__getitem__`.
        """
        N_antenna = self._XYZ.shape[1]
        N_px = self._R.shape[1]
        s = Sampler(N_antenna, N_px)
        return s

    def _encode_data(self):
        """
        Serialize data into buffer.

        Returns
        -------
        enc : :py:class:`~numpy.ndarray`
            (N_cell,) vectorized encoding.
        """
        N_antenna = self._XYZ.shape[1]
        N_px = self._R.shape[1]
        N_sample, N_data = self._data.shape

        N_cell = 6 + 3 * (N_antenna + N_px) + N_sample * N_data + 3 * N_sample
        enc = np.zeros((N_cell,))

        enc[0] = self._wl
        enc[1] = N_antenna
        enc[2] = N_px
        enc[3] = N_sample
        enc[4] = N_data
        enc[5] = self._gamma
        N_start = 6

        N_fill = 3 * N_antenna
        enc[N_start:(N_start + N_fill)] = self._XYZ.reshape(-1)
        N_start += N_fill

        N_fill = 3 * N_px
        enc[N_start:(N_start + N_fill)] = self._R.reshape(-1)
        N_start += N_fill

        N_fill = N_sample * N_data
        enc[N_start:(N_start + N_fill)] = self._data.reshape(-1)
        N_start += N_fill

        N_fill = N_sample
        enc[N_start:(N_start + N_fill)] = self._lambda_
        N_start += N_fill

        N_fill = N_sample
        enc[N_start:(N_start + N_fill)] = self._N_iter
        N_start += N_fill

        N_fill = N_sample
        enc[N_start:(N_start + N_fill)] = self._tts
        N_start += N_fill

        return enc

    def _encode_sky(self):
        """
        Serialize ground-truth into buffer.

        Returns
        -------
        enc : :py:class:`~numpy.ndarray`
            (N_cell,) vectorized encoding.
        """
        N_sample = self._data.shape[0]
        enc_len = [None] * N_sample
        enc_data = [None] * N_sample

        for i in range(N_sample):
            sky = self._ground_truth[i].encode()
            enc_data[i] = sky
            enc_len[i] = len(sky)

        N_cell = 1 + N_sample + sum(enc_len)
        enc = np.zeros((N_cell,))

        enc[0] = N_sample
        N_start = 1

        for i in range(N_sample):
            N_fill = 1
            enc[N_start:(N_start + N_fill)] = enc_len[i]
            N_start += N_fill

            N_fill = enc_len[i]
            enc[N_start:(N_start + N_fill)] = enc_data[i]
            N_start += N_fill

        return enc

    def encode(self):
        """
        Serialize data into buffer.

        Returns
        -------
        enc : :py:class:`~numpy.ndarray`
            (N_cell,) vectorized encoding.
        """
        enc_data = self._encode_data()
        N_cell_data = len(enc_data)

        enc_sky = self._encode_sky()
        N_cell_sky = len(enc_sky)

        N_cell = 2 + N_cell_data + N_cell_sky
        enc = np.zeros((N_cell,))

        enc[0] = N_cell_data
        enc[1] = N_cell_sky
        N_start = 2

        N_fill = N_cell_data
        enc[N_start:(N_start + N_fill)] = enc_data
        N_start += N_fill

        N_fill = N_cell_sky
        enc[N_start:(N_start + N_fill)] = enc_sky
        N_start += N_fill

        return enc

    @classmethod
    def _decode_data(cls, enc):
        """
        Decode data from buffer.

        Parameters
        ----------
        enc : :py:class:`~numpy.ndarray`
            (N_cell,) vectorized encoding, output of
            :py:meth:`~acoustic_camera.nn.DataSet._encode_data`.

        Returns
        -------
        data : :py:class:`~numpy.ndarray`
            (N_sample, N_data) encoded samples. (real-valued)
        XYZ : :py:class:`~numpy.ndarray`
            (3, N_antenna) Cartesian array geometry.
        R : :py:class:`~numpy.ndarray`
            (3, N_px) Cartesian grid points.
        wl : float
            Wavelength >= 0 [m].
        lambda_ : :py:class:`~numpy.ndarray`
            (N_sample,) APGD regularization parameters.
        gamma : float
            Linear trade-off between lasso and ridge regularizers.
        N_iter : :py:class:`~numpy.ndarray`
            (N_sample,) APGD iterations required to reach convergence.
        tts : :py:class:`~numpy.ndarray`
            (N_sample,) time [s] required for APGD solution to converge.
        """
        wl = enc[0]
        N_antenna = int(enc[1])
        N_px = int(enc[2])
        N_sample, N_data = enc[3:5].astype(int)
        gamma = enc[5]
        N_start = 6

        N_fill = 3 * N_antenna
        XYZ = enc[N_start:(N_start + N_fill)].reshape(3, N_antenna)
        N_start += N_fill

        N_fill = 3 * N_px
        R = enc[N_start:(N_start + N_fill)].reshape(3, N_px)
        N_start += N_fill

        N_fill = N_sample * N_data
        data = enc[N_start:(N_start + N_fill)].reshape(N_sample, N_data)
        N_start += N_fill

        N_fill = N_sample
        lambda_ = enc[N_start:(N_start + N_fill)]
        N_start += N_fill

        N_fill = N_sample
        N_iter = enc[N_start:(N_start + N_fill)].astype(int)
        N_start += N_fill

        N_fill = N_sample
        tts = enc[N_start:(N_start + N_fill)]
        N_start += N_fill

        return data, XYZ, R, wl, lambda_, gamma, N_iter, tts

    @classmethod
    def _decode_sky(cls, enc):
        """
        Decode ground-truth from buffer.

        Parameters
        ----------
        enc : :py:class:`~numpy.ndarray`
            (N_cell,) vectorized encoding, output of
            :py:meth:`~acoustic_camera.nn.DataSet._encode_sky`.

        Returns
        -------
        ground_truth : list(:py:class:`~acoustic_camera.tools.data_gen.source.SkyModel`)
            (N_sample,) ground truth sky models.
        """
        N_sample = int(enc[0])
        N_start = 1

        ground_truth = [None] * N_sample
        for i in range(N_sample):
            N_fill = 1
            len_data = int(enc[N_start:(N_start + N_fill)])
            N_start += N_fill

            N_fill = len_data
            data = enc[N_start:(N_start + N_fill)]
            N_start += N_fill

            sky_model = source.SkyModel.decode(data)
            ground_truth[i] = sky_model

        return ground_truth

    @classmethod
    def decode(cls, enc):
        """
        Decode dataset from buffer.

        Parameters
        ----------
        enc : :py:class:`~numpy.ndarray`
            (N_cell,) vectorized encoding, output of
            :py:meth:`~acoustic_camera.nn.DataSet.encode`.

        Returns
        -------
        D : :py:class:`~acoustic_camera.nn.DataSet`
        """
        N_cell_data = int(enc[0])
        N_cell_sky = int(enc[1])
        N_start = 2

        N_fill = N_cell_data
        data, XYZ, R, wl, lambda_, gamma, N_iter, tts = cls._decode_data(enc[N_start:(N_start + N_fill)])
        N_start += N_cell_data

        N_fill = N_cell_sky
        ground_truth = cls._decode_sky(enc[N_start:(N_start + N_fill)])
        N_start += N_fill

        D = cls(data, XYZ, R, wl, ground_truth, lambda_, gamma, N_iter, tts)
        return D


class Sampler(optim.Sampler):
    """
    Serializer to encode/decode samples of :py:class:`~acoustic_camera.nn.DataSet`.
    """

    def __init__(self, N_antenna, N_px):
        """
        Parameters
        ----------
        N_antenna : int
        N_px : int
        """
        super().__init__()

        self._N_antenna = N_antenna
        self._N_px = N_px
        self._N_cell = 2 * ((self._N_antenna ** 2) + self._N_px)

    def encode(self, buffer=None, S=None, I=None, I_prev=None):
        """
        Encode sample information in buffer.

        Parameters
        ----------
        buffer : :py:class:`~numpy.ndarray`
            (N_cell,) buffer in which to write the data.
            If `None`, a new buffer will be allocated.
        S : :py:class:`~numpy.ndarray`
            (N_antenna, N_antenna) visibility matrix (complex-valued).
            If `None`, the buffer is not modified at the intended location.
        I : :py:class:`~numpy.ndarray`
            (N_px,) APGD intensity estimate.
            If `None`, the buffer is not modified at the intended location.
        I_prev : :py:class:`~numpy.ndarray`
            (N_px,) initial-point used during APGD optimization.
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
                raise ValueError('Parameter[buffer] ill-formed.')
            enc = buffer
        N_start = 0

        N_fill = 2 * (self._N_antenna ** 2)
        if S is not None:
            if not (S.shape == (self._N_antenna, self._N_antenna)):
                raise ValueError('Parameter[S] must be (N_antenna, N_antenna) hermitian.')
            enc[N_start:(N_start + N_fill)] = (np.ascontiguousarray(S, dtype=np.complex128)
                                               .reshape(-1)
                                               .view(dtype=np.float64))
        N_start += N_fill

        N_fill = self._N_px
        if I is not None:
            if not (I.shape == (self._N_px,)):
                raise ValueError('Parameter[I] is ill-formed.')
            enc[N_start:(N_start + N_fill)] = I
        N_start += N_fill

        N_fill = self._N_px
        if I_prev is not None:
            if not (I_prev.shape == (self._N_px,)):
                raise ValueError('Parameter[I_prev] is ill-formed.')
            enc[N_start:(N_start + N_fill)] = I_prev
        N_start += N_fill

        return enc

    def decode(self, enc, keepdims=False):
        """
        Decode sample information from buffer.

        Parameters
        ----------
        enc : :py:class:`~numpy.ndarray`
            ([N_sample], N_cell) vectorized encoding, output of
            :py:meth:`~acoustic_camera.nn.Sampler.encode`.
        keepdims : bool
            If `True` and `enc.ndim == 1', then the `1`-sized leading dimension
            of the outputs is dropped.

        Returns
        -------
        S : :py:class:`~numpy.ndarray`
            ([N_sample], N_antenna, N_antenna) visibility matrix (complex-valued).
        I : :py:class:`~numpy.ndarray`
            ([N_sample], N_px) APGD intensity estimate.
        I_prev : :py:class:`~numpy.ndarray`
            ([N_sample], N_px) initial-point used during APGD optimization.
        """
        was_1d = (enc.ndim == 1)
        if was_1d:
            enc = enc.reshape(1, -1)
        N_sample = len(enc)

        if not (enc.shape == (N_sample, self._N_cell)):
            raise ValueError('Parameter[enc] is ill-formed.')
        N_start = 0

        N_fill = 2 * (self._N_antenna ** 2)
        S = (np.ascontiguousarray(enc[:, N_start:(N_start + N_fill)], dtype=np.float64)
             .view(np.complex128)
             .reshape(N_sample, self._N_antenna, self._N_antenna))
        N_start += N_fill

        N_fill = self._N_px
        I = enc[:, N_start:(N_start + N_fill)]
        N_start += N_fill

        N_fill = self._N_px
        I_prev = enc[:, N_start:(N_start + N_fill)]
        N_start += N_fill

        if (not keepdims) and was_1d:
            S = S.squeeze(axis=0)
            I = I.squeeze(axis=0)
            I_prev = I_prev.squeeze(axis=0)
        return S, I, I_prev
