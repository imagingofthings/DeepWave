# #############################################################################
# statistics.py
# =============
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# #############################################################################

"""
Visibility generation utilities.
"""

import acoustic_camera.tools.instrument as inst
import acoustic_camera.tools.math.stat as stat
import numpy as np
import scipy.signal.windows as windows
import skimage.util as skutil


class VisibilityGenerator:
    """
    Generate synthetic visibility matrices using the Wishart distribution.
    """

    def __init__(self, T, fs, SNR):
        """
        Parameters
        ----------
        T : float
            Integration time [s].
        fs : float
            Sampling rate [Hz].
        SNR : float
            Signal-to-Noise-Ratio (dB).
        """
        if T <= 0:
            raise ValueError('Parameter[T] must be positive.')
        if fs <= 0:
            raise ValueError('Parameter[fs] must be positive.')

        self._N_sample = int(T * fs) + 1
        self._SNR = 10 ** (SNR / 10)

    def __call__(self, XYZ, wl, sky_model):
        """
        Compute visibility matrix.

        Parameters
        ----------
        XYZ : :py:class:`~numpy.ndarray`
            (3, N_antenna) Cartesian instrument geometry.
        wl : float
            Wave-length >= 0 [m].
        sky_model : :py:class:`~tools.data.source.SkyModel`
            Source model.

        Returns
        -------
        S : :py:class:`~numpy.ndarray`
            (N_antenna, N_antenna) visibility matrix.
        """
        if wl <= 0:
            raise ValueError('Parameter[wl] must be positive.')

        N_antenna = XYZ.shape[1]

        s_xyz = sky_model.xyz
        s_I = sky_model.intensity

        A = inst.steering_operator(XYZ, s_xyz, wl)
        S_sky = (A * s_I) @ A.conj().T

        noise_var = np.sum(s_I) / (2 * self._SNR)
        S_noise = noise_var * np.eye(N_antenna)

        wishart = stat.Wishart(V=S_sky + S_noise, n=self._N_sample)
        S = wishart()[0] / self._N_sample
        return S


class TimeSeriesGenerator:
    """
    Generate synthetic baseband-equivalent time-series.
    """

    def __init__(self, fs, SNR):
        """
        Parameters
        ----------
        fs : float
            Sampling rate [Hz].
        SNR : float
            Signal-to-Noise-Ratio (dB).
        """
        if fs <= 0:
            raise ValueError('Parameter[fs] must be positive.')
        self._fs = fs

        self._SNR = 10 ** (SNR / 10)

    def __call__(self, XYZ, wl, sky_model, T):
        """
        Compute time series.

        Parameters
        ----------
        XYZ : :py:class:`~numpy.ndarray`
            (3, N_antenna) Cartesian instrument geometry.
        wl : float
            Wave-length >= 0 [m].
        sky_model : :py:class:`~tools.data.source.SkyModel`
            Source model.
        T : float
            Signal duration [s].

        Returns
        -------
        ts : :py:class:`~numpy.ndarray`
            (N_sample, N_channel) baseband-equivalent time series.
        """
        if wl <= 0:
            raise ValueError('Parameter[wl] must be positive.')
        if not (isinstance(T, float) and (T > 0)):
            raise ValueError('Parameter[T] must be positive.')
        N_sample, N_channel = int(T * self._fs) + 1, XYZ.shape[1]

        s_xyz = sky_model.xyz
        s_I = sky_model.intensity
        N_src = s_I.shape[0]

        A = inst.steering_operator(XYZ, s_xyz, wl)
        s_eps = (np.sqrt(s_I / 2).reshape(N_src, 1) *
                 (np.random.randn(N_src, N_sample) +
                  1j * np.random.randn(N_src, N_sample)))
        s_ts = A @ s_eps

        noise_var = np.sum(s_I) / (2 * self._SNR)
        n_ts = np.sqrt(noise_var) * (np.random.randn(N_channel, N_sample) +
                                     1j * np.random.randn(N_channel, N_sample))

        ts = s_ts.T + n_ts.T
        return ts


class TimeSeries:
    """
    Time-Series representation.
    """

    def __init__(self, data, rate):
        """
        Parameters
        ----------
        data : :py:class:`~numpy.ndarray`
            (N_sample, N_channel) time series (real-valued)
        rate : int
            Sample Rate [Hz]
        """
        if not (isinstance(data, np.ndarray) and
                (data.ndim == 2) and
                np.issubdtype(data.dtype, np.floating)):
            raise ValueError('Parameter[data] must be a (N_sample, N_channel) real-valued.')
        self._data = data

        if not (isinstance(rate, int) and (rate > 0)):
            raise ValueError('Parameter[rate] must be positive.')
        self._rate = rate

    @property
    def data(self):
        """
        Returns
        -------
        data : :py:class:`~numpy.ndarray`
            (N_sample, N_channel) time series (real-valued)
        """
        return self._data

    @property
    def rate(self):
        """
        Returns
        -------
        rate : int
            Sample Rate [Hz]
        """
        return self._rate

    def extract_visibilities(self, T, fc, bw, alpha):
        """
        Transform time-series to visibility matrices.

        Parameters
        ----------
        T : float
            Integration time [s].
        fc : float
            Center frequency [Hz] around which visibility matrices are formed.
        bw : float
            Double-wide bandwidth [Hz] of the visibility matrix.
        alpha : float
            Shape parameter of the Tukey window, representing the fraction of
            the window inside the cosine tapered region. If zero, the Tukey
            window is equivalent to a rectangular window. If one, the Tukey
            window is equivalent to a Hann window.

        Returns
        -------
        S : :py:class:`~numpy.ndarray`
            (N_slot, N_channel, N_channel) visibility matrices (complex-valued).
        """
        if not (isinstance(T, float) and (T > 0)):
            raise ValueError('Parameter[T] must be positive.')

        if not (isinstance(fc, float) and (fc > 0)):
            raise ValueError('Parameter[fc] must be positive.')

        if not (isinstance(bw, float) and (bw > 0)):
            raise ValueError('Parameter[bw] must be positive.')

        if not (isinstance(alpha, float) and (0 <= alpha <= 1)):
            raise ValueError('Parameter[alpha] must be in [0, 1].')

        if not (bw < 2 * fc < self._rate - bw):
            raise ValueError('Interested frequency range too broad given sampling rate.')

        N_stft_sample = int(self._rate * T)
        if N_stft_sample == 0:
            raise ValueError('Not enough samples per time frame.')
        print(f'Samples per STFT: {N_stft_sample}')

        N_sample = (self._data.shape[0] // N_stft_sample) * N_stft_sample
        N_channel = self._data.shape[1]
        stf_data = (skutil.view_as_blocks(self._data[:N_sample], (N_stft_sample, N_channel))
                    .squeeze(axis=1))  # (N_stf, N_stft_sample, N_channel)

        window = windows.tukey(M=N_stft_sample, alpha=alpha, sym=True).reshape(1, -1, 1)
        stf_win_data = stf_data * window  # (N_stf, N_stft_sample, N_channel)
        N_stf = stf_win_data.shape[0]

        stft_data = np.fft.fft(stf_win_data, axis=1)  # (N_stf, N_stft_sample, N_channel)
        # Find frequency channels to average together.
        idx_start = int((fc - 0.5 * bw) * N_stft_sample / self._rate)
        idx_end = int((fc + 0.5 * bw) * N_stft_sample / self._rate)
        print(f'Spectrum start index: {idx_start}/{N_stft_sample}')
        print(f'Spectrum end index: {idx_end}/{N_stft_sample}')
        collapsed_spectrum = np.sum(stft_data[:, idx_start:idx_end + 1, :], axis=1)

        # Don't understand yet why conj() on first term?
        S = (collapsed_spectrum.reshape(N_stf, -1, 1).conj() *
             collapsed_spectrum.reshape(N_stf, 1, -1))
        return S
