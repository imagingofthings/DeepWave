# ############################################################################
# extract_dataset.py
# ==================
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

r"""
Form per-frequency APGD datasets from pyramic-dataset .wav files.

Datasets stored under ./dataset and can be merged using merge_dataset.py
"""

import argparse
import itertools
import logging
import pathlib
import re
import sys

import numpy as np
import scipy.constants as constants
import scipy.io.wavfile as wavfile
import scipy.linalg as linalg
import skimage.util as skutil

import deepwave.apgd as apgd
import deepwave.nn as nn
import deepwave.tools.data_gen.source as source
import deepwave.tools.data_gen.statistics as statistics
import imot_tools.math.sphere.grid as grid
import imot_tools.phased_array as phased_array

import helpers


def parse_args():
    parser = argparse.ArgumentParser(description='Extract multi-spectral datasets from Pyramic time-series',
                                     epilog="""
    Example
    -------
    python3 extract_dataset.py --N_sample=3 --N_src=1 --seed=0
                """,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--N_src',
                        help='Number of sources per measurement.',
                        required=True,
                        type=int)
    parser.add_argument('--N_sample',
                        help=('Number of measurements to simulate. Setting this to -1 means generate '
                              'all possible options given available data.'),
                        required=True,
                        type=int)
    parser.add_argument('--seed',
                        help='Random number generator seed.',
                        required=True,
                        type=int)
    parser.add_argument('--angles',
                        help=('List of angles in range(0, 360, 2) that are to be used. '
                              'If left unspecified, use all angles.'),
                        required=False,
                        default=np.arange(0, 358, 2)[::18])

    args = parser.parse_args()
    if not (1 <= args.N_src <= 100):  # Arbitrary upper-bound
        raise ValueError('Parameter[N_src] is out of bounds.')
    if not ((1 <= args.N_sample <= 50000) or (args.N_sample == -1)):  # Arbitrary upper-bound
        raise ValueError('Parameter[N_sample] is out of bounds.')

    angles = np.unique(eval(args.angles))
    all_in_range = np.isclose(angles.reshape(-1, 1),
                              np.arange(0, 360, 2).reshape(1, -1)).any(axis=1).all()
    if not all_in_range:
        raise ValueError('Parameter[angles] contains out-of-range indices.')
    args.angles = angles

    return vars(args)


def get_data(N_src, angles, permute):
    """
    Generator of `N_src`-emission data streams.

    Parameters
    ----------
    N_src : int
        Number of source emissions per data-stream.
    angles : :py:class:`~numpy.ndarray`
        (N_angle,) valid angles to choose from.
    permute : bool
        If :py:obj:`True`, then output data-streams in random order.
        If :py:obj:`False`, then output data-streams in deterministic order.

    Returns
    -------
    iterable

        Generator object returning (rate, data, sky_model) triplets with:

        * rate (int): Sample rate [Hz]
        * data (:py:class:`~numpy.ndarray`): (N_sample, N_channel) samples (float64)
        * sky_model (:py:class:`deepwave.tools.data_gen.source.SkyModel`): (N_src,) ground truth for simulated emissions.
    """
    range_human = np.arange(5)
    range_colat = np.arange(3)
    range_lon = angles
    if permute:
        rng = np.random.default_rng()
        range_human = rng.permutation(range_human)
        range_colat = rng.permutation(range_colat)
        range_lon = rng.permutation(range_lon)

    f_gen = lambda h, c, l: pathlib.Path(f'./pyramic-dataset/segmented/fq_sample{h}/fq_sample{h}_spkr{c}_angle{l}.wav')
    f_name = [f_gen(idx_human, idx_colat, idx_lon) for idx_human in range_human
                                                   for idx_colat in range_colat
                                                   for idx_lon in range_lon]


    def different_triplets(f):
        def extract_triplets(f):
            pattern = r'fq_sample(\d)_spkr(\d)_angle(\d+).wav'
            match = re.search(pattern, str(f))
            h, c, l = map(int, match.group(1, 2, 3))
            return h, c, l

        triplets = [extract_triplets(_) for _ in f]
        if len(set(triplets)) == len(triplets):
            return True
        else:
            return False


    for f in itertools.combinations(f_name, r=N_src):
        # if the same angle came (idx_colat, idx_lon) pair came out twice due to different humans
        # speaking, discard the sample.
        if different_triplets(f):
            rate, data = zip(*[wavfile.read(_) for _ in f])
            if not np.allclose(rate, rate[0]):
                raise ValueError('Data-files have different rates.')
            rate = rate[0]

            # Truncate data streams to match minimum length one.
            # (Because we guarantee N_src different emissions in the field.)
            min_length = np.amin([len(_) for _ in data])
            max_int = np.iinfo(data[0].dtype).max
            data = np.sum([_[:min_length].astype(np.float) for _ in data], axis=0)
            data = data / max_int

            sky_model = helpers.merge_sky([helpers.sky(_) for _ in f])
            yield rate, data, sky_model


def form_visibility(data, rate, fc, bw, T_sti, T_stationarity):
    """
    Parameter
    ---------
    data : :py:class:`~numpy.ndarray`
        (N_sample, N_channel) antenna samples. (float)
    rate : int
        Sample rate [Hz]
    fc : float
        Center frequency [Hz] around which visibility matrices are formed.
    bw : float
        Double-wide bandwidth [Hz] of the visibility matrix.
    T_sti : float
        Integration time [s]. (time-series)
    T_stationarity : float
        Integration time [s]. (visibility)

    Returns
    -------
    S : :py:class:`~numpy.ndarray`
        (N_slot, N_channel, N_channel) visibility matrices.

    Note
    ----
    Visibilities computed directly in the frequency domain.
    For some reason visibilities are computed correctly using
    `x.reshape(-1, 1).conj() @ x.reshape(1, -1)` and not the converse.
    Don't know why at the moment.
    """
    S_sti = (statistics.TimeSeries(data, rate)
             .extract_visibilities(T_sti, fc, bw, alpha=1.0))

    N_sample, N_channel = data.shape
    N_sti_per_stationary_block = int(T_stationarity / T_sti) + 1
    S = (skutil.view_as_windows(S_sti,
                                (N_sti_per_stationary_block, N_channel, N_channel),
                                (N_sti_per_stationary_block, N_channel, N_channel))
         .squeeze(axis=(1, 2))
         .sum(axis=1))
    return S


def process_track(rate, data, sky_model):
    """
    Parameters
    ----------
    rate : int
        Sample rate [Hz].
    data : :py:class:`~numpy.ndarray`
        (N_sample, N_channel) samples. (float64)
    sky_model : :py:class:`~deepwave.tools.data_gen.source.SkyModel`
        (N_src,) ground truth for synthesized data stream.

    Returns
    -------
    D : dict(freq_idx -> :py:class:`~deepwave.nn.DataSet`)
    """
    dev_xyz = helpers.mic_xyz()
    N_antenna = dev_xyz.shape[1]

    freq, bw = (skutil  # Center frequencies to form images
                .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
                .mean(axis=-1)), 50.0  # [Hz]
    T_sti = 12.5e-3
    T_stationarity = 8 * T_sti  # Choose to have frame_rate = 10.
    N_freq = len(freq)

    wl_min = helpers.speed_of_sound() / (freq.max() + 500)
    sh_order = phased_array.nyquist_rate(dev_xyz, wl_min)
    R = grid.fibonacci(sh_order)
    R_mask = np.abs(R[2, :]) < np.sin(np.deg2rad(30))
    R = R[:, R_mask]  # Shrink visible view to avoid border effects.
    N_px = R.shape[1]

    D = dict()
    sampler = nn.Sampler(N_antenna, N_px)
    for idx_freq in range(N_freq):
        wl = helpers.speed_of_sound() / freq[idx_freq]
        A = phased_array.steering_operator(dev_xyz, R, wl)

        S = form_visibility(data, rate, freq[idx_freq], bw, T_sti, T_stationarity)
        N_sample = S.shape[0]

        apgd_gamma = 0.5
        apgd_lambda_ = np.zeros((N_sample,))
        apgd_N_iter = np.zeros((N_sample,), dtype=int)
        apgd_tts = np.zeros((N_sample,))
        apgd_data = np.zeros((N_sample, sampler._N_cell))
        I_prev = np.zeros((N_px,))
        for idx_s in range(N_sample):
            logging.info(f'Processing freq_idx={idx_freq + 1:02d}/{N_freq:02d}, '
                         f'sample_idx={idx_s + 1:03d}/{N_sample:03d}')

            # Normalize visibilities
            S_D, S_V = linalg.eigh(S[idx_s])
            if S_D.max() <= 0:
                S_D[:] = 0
            else:
                S_D = np.clip(S_D / S_D.max(), 0, None)
            S_norm = (S_V * S_D) @ S_V.conj().T

            I_apgd = apgd.solve(S_norm, A, gamma=apgd_gamma, x0=I_prev.copy(), verbosity='NONE')
            apgd_data[idx_s] = sampler.encode(S=S_norm,
                                              I=I_apgd['backtrace'][-1],
                                              I_prev=I_apgd['backtrace'][0])
            apgd_lambda_[idx_s] = I_apgd['lambda_']
            apgd_N_iter[idx_s] = I_apgd['niter']
            apgd_tts[idx_s] = I_apgd['time']

        D_ = nn.DataSet(data=apgd_data,
                        XYZ=dev_xyz,
                        R=R,
                        wl=wl,
                        ground_truth=[sky_model] * N_sample,
                        lambda_=apgd_lambda_,
                        gamma=apgd_gamma,
                        N_iter=apgd_N_iter,
                        tts=apgd_tts)

        D[idx_freq] = D_
    return D


def process(N_src, N_sample, angles):
    stream = get_data(N_src, angles, permute=False if (N_sample == -1) else True)
    for idx_s, (rate, data, sky_model) in enumerate(stream, start=1):
        logging.info(f'Processing synthesized data stream {idx_s:05d}/{N_sample:05d}.')

        D_ = process_track(rate, data, sky_model)
        for idx_freq, D in D_.items():
            f_name = pathlib.Path(f'./dataset/D_freq{idx_freq}_NSRC_{N_src:03d}_NSAMPLE_{N_sample:05d}_sample_{idx_s:05d}.npz')
            D.to_file(str(f_name))
            logging.info(f'Wrote {f_name.name} to disk')

        if (N_sample != -1) and (idx_s >= N_sample):
            break


if __name__ == '__main__':
    args = parse_args()

    log_fname = (pathlib.Path(f'./dataset/D_NSRC_{args["N_src"]:03d}_NSAMPLE_{args["N_sample"]:05d}.log')
                 .expanduser().absolute())
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s | %(message)s',
                        filename=log_fname,
                        filemode='w')
    # Setup logging to stdout.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s | %(message)s')
    console.setFormatter(console_formatter)
    logging.getLogger(__name__).addHandler(console)
    logging.info(str(argparse.Namespace(**args)))
    logging.info('START')

    np.random.seed(args['seed'])
    process(args['N_src'], args['N_sample'], args['angles'])
    logging.info('DONE')
