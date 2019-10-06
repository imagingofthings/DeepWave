# ############################################################################
# export_rgb.py
# =============
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Compute RGB image contrast.
"""

import argparse
import collections.abc as abc
import os
import pathlib

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pkg_resources as pkg
import tqdm

import deepwave.nn as nn
import deepwave.nn.crnn as crnn
import deepwave.spectral as spectral
import deepwave.tools.math.func as func
import deepwave.tools.math.graph as graph
import deepwave.tools.math.linalg as pylinalg
import imot_tools.math.sphere.transform as transform
import imot_tools.phased_array as phased_array


def parse_args():
    parser = argparse.ArgumentParser(description='Export RGB images.',
                                     epilog=r"""
    Example
    -------
    python3 export_rgb.py --datasets D_freq[0-8].npz         \
                          --parameters D_freq[0-8]_train.npz
                """,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--datasets',
                        help=('Multi-frequency datasets. It is assumed files once sorted go from '
                              'smallest to largest frequency.'),
                        nargs='+',
                        type=str,
                        required=True,)
    parser.add_argument('--parameters',
                        help=('Trained network parameter files. It is assumed files once sorted '
                              'go from smallest to largest frequency.'),
                        nargs='+',
                        type=str,
                        required=True,)

    parser.add_argument('--img_type',
                        help='Type of image to produce.',
                        required=True,
                        type=str,
                        choices=['APGD', 'RNN', 'DAS'])
    parser.add_argument('--img_idx',
                        help=('Image indices to export. Interpreted as Python code. '
                              'If left unspecified, export all images.'),
                        default=None)

    parser.add_argument('--out',
                        required=True,
                        help='.npz file to save contrast to.')

    parser.add_argument('--lon_ticks',
                        help='in degrees',
                        default=np.linspace(-180, 180, 5))

    args = parser.parse_args()

    datasets = []
    for f in args.datasets:
        ff = pathlib.Path(f).expanduser().absolute()
        if not ff.exists():
            raise ValueError(f'File "{str(ff)}" does not exist.')
        datasets.append(ff)
    args.datasets = datasets

    parameters = []
    for f in args.parameters:
        ff = pathlib.Path(f).expanduser().absolute()
        if not ff.exists():
            raise ValueError(f'File "{str(ff)}" does not exist.')
        parameters.append(ff)
    args.parameters = parameters

    # Make sure (dataset, parameter) are conformant (light check)
    if not (len(args.datasets) == len(args.parameters)):
        raise ValueError('Parameters[datasets, parameters] have different number of arguments.')
    args.datasets = sorted(args.datasets)
    args.parameters = sorted(args.parameters)

    if args.img_idx is None:
        D = nn.DataSet.from_file(str(args.datasets[0]))
        N_sample = len(D)
        img_idx = np.arange(N_sample)
    else:
        img_idx = np.unique(eval(args.img_idx))
    for f in args.datasets:
        D = nn.DataSet.from_file(str(f))
        N_sample = len(D)
        if not np.all((img_idx >= 0) & (img_idx < N_sample)):
            raise ValueError('Some image indices are out of range.')
    args.img_idx = img_idx

    f_out = pathlib.Path(args.out).expanduser().absolute()
    args.out = f_out

    if isinstance(args.lon_ticks, str):  # user-specified values
        lon_ticks = eval(args.lon_ticks)
    else:
        lon_ticks = np.unique(args.lon_ticks)
    if not ((-180 <= lon_ticks.min()) &
            (lon_ticks.max() <= 180)):
        raise ValueError('Parameter[lon_ticks] is out of range.')
    args.lon_ticks = lon_ticks

    return vars(args)


def get_field(D, P, idx_img, img_type):
    """
    Parameters
    ----------
    D : list(:py:class:`~deepwave.nn.DataSet`)
        (9,) multi-frequency datasets.
    P : list(:py:class:`~deepwave.nn.crnn.Parameter`)
        (9,) multi-frequency trained parameters.
    idx_img : int
        Image index
    img_type : str
        One of ['APGD', 'RNN', 'DAS']

    Returns
    -------
    I : :py:class:`~numpy.ndarray`
        (9, N_px) frequency intensities of specified image.
    """
    I = []
    for idx_freq in range(9):
        Df, Pf = D[idx_freq], P[idx_freq]

        N_antenna = Df.XYZ.shape[1]
        N_px = Df.R.shape[1]
        K = int(Pf['K'])
        parameter = crnn.Parameter(N_antenna, N_px, K)
        sampler = Df.sampler()

        A = phased_array.steering_operator(Df.XYZ, Df.R, Df.wl)
        if img_type == 'APGD':
            _, I_apgd, _ = sampler.decode(Df[idx_img])
            I.append(I_apgd)
        elif img_type == 'RNN':
            Ln, _ = graph.laplacian_exp(Df.R, normalized=True)
            afunc = lambda _: func.retanh(Pf['tanh_lin_limit'], _)
            p_opt = Pf['p_opt'][np.argmin(Pf['v_loss'])]
            S, _, I_prev = sampler.decode(Df[idx_img])
            N_layer = Pf['N_layer']
            rnn_eval = crnn.Evaluator(N_layer, parameter, p_opt, Ln, afunc)
            I_rnn = rnn_eval(S, I_prev)
            I.append(I_rnn)
        elif img_type == 'DAS':
            S, _, _ = sampler.decode(Df[idx_img])
            alpha = 1 / (2 * pylinalg.eighMax(A))
            beta = 2 * Df.lambda_[idx_img] * alpha * (1 - Df.gamma) + 1

            I_das = spectral.DAS(Df.XYZ, S, Df.wl, Df.R) * 2 * alpha / beta
            I.append(I_das)
        else:
            raise ValueError(f'Parameter[img_type] invalid.')

    I = np.stack(I, axis=0)
    return I


def to_RGB(I):
    """
    Parameters
    ----------
    I : :py:class:`~numpy.ndarray`
        (9, N_px) real-valued intensity (per frequency)

    Returns
    -------
    I_rgb: :py:class:`~numpy.ndarray`
        (3, N_px) real-valued intensity (per color-band)
    """
    N_px = I.shape[1]
    I_rgb = I.reshape((3, 3, N_px)).sum(axis=1)
    return I_rgb


def wrapped_rad2deg(lat_r, lon_r):
    """
    Equatorial coordinate [rad] -> [deg] unit conversion.
    Output longitude guaranteed to lie in [-180, 180) [deg].

    Parameters
    ----------
    lat_r : :py:class:`~numpy.ndarray`
    lon_r : :py:class:`~numpy.ndarray`

    Returns
    -------
    lat_d : :py:class:`~numpy.ndarray`
    lon_d : :py:class:`~numpy.ndarray`
    """
    lat_d = coord.Angle(lat_r * u.rad).to_value(u.deg)
    lon_d = coord.Angle(lon_r * u.rad).wrap_at(180 * u.deg).to_value(u.deg)
    return lat_d, lon_d


if __name__ == '__main__':
    args = parse_args()

    D = [nn.DataSet.from_file(str(_)) for _ in args['datasets']]
    P = [np.load(_) for _ in args['parameters']]

    R = D[0].R
    N_px = R.shape[1]
    N_sample = len(args['img_idx'])

    I_all = np.zeros((N_sample, 3, N_px))
    for i, idx_img in tqdm.tqdm(list(enumerate(args['img_idx']))):
        I = get_field(D, P, idx_img, img_type=args['img_type'])

        I_rgb = to_RGB(I)
        # I_rgb /= I_rgb.max()

        # Filter field to lie in specified interval
        _, R_lat, R_lon = transform.cart2eq(*R)
        _, R_lon_d = wrapped_rad2deg(R_lat, R_lon)
        min_lon, max_lon = args['lon_ticks'].min(), args['lon_ticks'].max()
        mask_lon = (min_lon <= R_lon_d) & (R_lon_d <= max_lon)

        R_field = transform.eq2cart(1, R_lat[mask_lon], R_lon[mask_lon])
        I_rgb = I_rgb[:, mask_lon]

        I_all[i] = I_rgb

    np.savez(args['out'], I_rgb=I_all)
