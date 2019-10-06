# ############################################################################
# simulate.py
# ===========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Generate Nyquist-resolution APGD datasets on chosen FoV.
"""

import argparse
import logging
import pathlib
import sys

import numpy as np
import scipy.constants as constants
import scipy.linalg as linalg
import scipy.stats as stats

import deepwave.apgd as apgd
import deepwave.nn as nn
import deepwave.tools.data_gen.source as source
import deepwave.tools.data_gen.statistics as statistics
import deepwave.tools.instrument as instrument
import imot_tools.math.sphere.grid as grid
import imot_tools.math.sphere.transform as transform
import imot_tools.phased_array as phased_array


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate APGD images for setup described in README.txt',
        epilog="""
    Example
    -------
    python3 simulate.py --dataset='~/D.npz' \
                        --N_sample=50       \
                        --N_src=3           \
                        --rate=1.0
                """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--dataset',
                        help='.npz file to which dataset is stored.',
                        required=True,
                        type=str)
    parser.add_argument('--N_sample',
                        help='Number of images to simulate.',
                        required=True,
                        type=int)
    parser.add_argument('--N_src',
                        help='Number of sources present per image',
                        required=True,
                        type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--intensity',
                       help='Generate equi-amplitude sources',
                       action='store_const',
                       const=1.0)
    group.add_argument('--rate',
                       help='Generate rayleigh-distributed sources',
                       type=float)

    args = parser.parse_args()
    if not (args.N_sample > 0):
        raise ValueError('Parameter[N_sample] must be positive.')
    if not (args.N_src > 0):
        raise ValueError('Parameter[N_src] must be positive.')
    args.dataset = pathlib.Path(args.dataset).expanduser().absolute()
    return args


def simulate_dataset(N_sample, N_src, XYZ, R, wl, src_mask, intensity=None, rate=None):
    """
    Generate APGD dataset.

    Parameters
    ----------
    N_sample : int
        Number of images to generate.
    N_src : int
        Number of sources present per image.
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) microphone positions.
    R : :py:class:`~numpy.ndarray`
        (3, N_px) pixel grid.
    wl : float
        Wavelength [m] of plane wave.
    src_mask : :py:class:`~numpy.ndarray`
        (N_px,) boolean mask saying near which pixels it is possible to place sources.
    intensity : float
        If present, generate equi-amplitude sources.
    rate : float
        If present, generate rayleigh-amplitude sources.

    Returns
    -------
    D : :py:class:`~acoustic_camera.nn.DataSet`
        (N_sample,) dataset

    Note
    ----
    Either `intensity` or `rate` must be specified, not both.
    """
    if not (N_sample > 0):
        raise ValueError('Paremeter[N_sample] must be positive.')

    if not (N_src > 0):
        raise ValueError('Paremeter[N_src] must be positive.')

    if not ((XYZ.ndim == 2) and (XYZ.shape[0] == 3)):
        raise ValueError('Parameter[XYZ]: expected (3, N_antenna) array.')
    N_antenna = XYZ.shape[1]

    if not ((R.ndim == 2) and (R.shape[0] == 3)):
        raise ValueError('Parameter[R]: expected (3, N_px) array.')
    N_px = R.shape[1]

    if wl < 0:
        raise ValueError('Parameter[wl] is out of bounds.')

    if not ((src_mask.ndim == 1) and (src_mask.size == N_px)):
        raise ValueError('Parameter[src_mask]: expected (N_px,) boolean array.')

    if (((intensity is None) and (rate is None)) or
            ((intensity is not None) and (rate is not None))):
        raise ValueError('One of Parameters[intensity, rate] must be specified.')

    if (intensity is not None) and (intensity <= 0):
        raise ValueError('Parameter[intensity] must be positive.')

    if (rate is not None) and (rate <= 0):
        raise ValueError('Parameter[rate] must be positive.')

    vis_gen = statistics.VisibilityGenerator(T=50e-3, fs=48000, SNR=10)
    A = phased_array.steering_operator(XYZ, R, wl)

    sampler = nn.Sampler(N_antenna, N_px)
    N_data = sampler._N_cell
    data = np.zeros((N_sample, N_data))
    ground_truth = [None] * N_sample
    apgd_gamma = 0.5
    apgd_lambda_ = np.zeros(N_sample)
    apgd_N_iter = np.zeros(N_sample)
    apgd_tts = np.zeros(N_sample)

    for i in range(N_sample):
        logging.info(f'Generate APGD image {i + 1}/{N_sample}.')

        ### Create synthetic sky
        if (intensity is not None):
            sky_I = intensity * np.ones(N_src)
        elif (rate is not None):
            sky_I = stats.rayleigh.rvs(scale=rate, size=N_src)
        sky_XYZ = R[:, src_mask][:, np.random.randint(0, np.sum(src_mask), size=N_src)]
        ## Randomize positions slightly to not fall straight onto grid.
        _, sky_colat, sky_lon = transform.cart2pol(*sky_XYZ)
        px_pitch = np.arccos(np.clip(R[:, 0] @ R[:, 1:], -1, 1)).min()
        colat_noise, lon_noise = 0.1 * px_pitch * np.random.randn(2, N_src)
        sky_XYZ = np.stack(transform.pol2cart(1, sky_colat + colat_noise, sky_lon + lon_noise), axis=0)
        sky_model = source.SkyModel(sky_XYZ, sky_I)

        S = vis_gen(XYZ, wl, sky_model)
        # Normalize `S` spectrum for scale invariance.
        S_D, S_V = linalg.eigh(S)
        if S_D.max() <= 0:
            S_D[:] = 0
        else:
            S_D = np.clip(S_D / S_D.max(), 0, None)
        S = (S_V * S_D) @ S_V.conj().T

        I_apgd = apgd.solve(S, A,
                            lambda_=None,
                            gamma=apgd_gamma,
                            L=None,
                            d=50,
                            x0=None,
                            eps=1e-3,
                            N_iter_max=200,
                            verbosity='NONE',  # 'LOW',
                            momentum=True)

        data[i] = sampler.encode(S=S, I=I_apgd['sol'])
        ground_truth[i] = sky_model
        apgd_lambda_[i] = I_apgd['lambda_']
        apgd_N_iter[i] = I_apgd['niter']
        apgd_tts[i] = I_apgd['time']

    D = nn.DataSet(data, XYZ, R, wl, ground_truth,
                   apgd_lambda_, apgd_gamma, apgd_N_iter, apgd_tts)
    return D


if __name__ == '__main__':
    args = parse_args()

    fname_log = (pathlib.Path(args.dataset.parent) /
                 (args.dataset.stem + ".log"))
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s | %(message)s',
                        filename=fname_log,
                        filemode='w')
    # Setup logging to stdout.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s | %(message)s')
    console.setFormatter(console_formatter)
    logging.getLogger(__name__).addHandler(console)
    logging.info(str(args))

    FoV, focus = np.deg2rad(120), np.r_[1.0, 0, 0]
    freq, freq_max = np.r_[2000, 5500]
    wl, wl_min = constants.speed_of_sound / np.r_[freq, freq_max]

    XYZ = instrument.spherical_geometry()
    grid_order = phased_array.nyquist_rate(XYZ, wl_min)
    R = grid.fibonacci(grid_order, direction=focus, FoV=FoV)

    # Limit sources to center of FoV
    angle_border_gap = np.deg2rad(10)
    px_mask = focus @ R > np.cos(0.5 * FoV - angle_border_gap)

    kwargs = dict(intensity=args.intensity if (args.intensity is not None) else None,
                  rate=args.rate if (args.rate is not None) else None, )
    D = simulate_dataset(args.N_sample, args.N_src, XYZ, R, wl, px_mask, **kwargs)

    logging.info('Write DataSet to disk.')
    D.to_file(str(args.dataset))

    logging.info('END')
