# ############################################################################
# color_plot.py
# =============
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Plot multi-frequency datasets in RGB colors.
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
    parser = argparse.ArgumentParser(description='Produce DAS/RNN true-color plots.',
                                     epilog=r"""
    Example
    -------
    python3 color_plot.py --datasets D_freq[0-8].npz         \
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

    parser.add_argument('--mode',
                        help='Visualization mode.',
                        choices=['disk', 'display'],
                        default='display')
    parser.add_argument('--out',
                        help=('Folder to save images. (if "--mode=disk"). '
                              'Will be created if non-existent.'))

    parser.add_argument('--show_catalog',
                        action='store_true')
    parser.add_argument('--lon_ticks',
                        help='in degrees',
                        default=np.linspace(-180, 180, 5))
    parser.add_argument('--hide_labels',
                        help='If provided, "Azimuth/Elevation" will not be written on the side of the plots.',
                        action='store_true')
    parser.add_argument('--hide_axis',
                        help='If provided, tick marks will not be written on the side of the plots.',
                        action='store_true')

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

    if args.mode == 'disk':
        if args.out is None:
            raise ValueError('Parameter[out] must be specified if saving to disk.')
        f_out = pathlib.Path(args.out).expanduser().absolute()
        os.makedirs(f_out, exist_ok=True)
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


def cmap_from_list(name, colors, N=256, gamma=1.0):
    """
    Parameters
    ----------
    name : str
    colors :
        * a list of (value, color) tuples; or
        * list of color strings
    N : int
        Number of RGB quantization levels.
    gamma : float
        Something?

    Returns
    -------
    cmap : :py:class:`matplotlib.colors.LinearSegmentedColormap`
    """
    from collections import Sized
    import matplotlib.colors

    if not isinstance(colors, abc.Iterable):
        raise ValueError('colors must be iterable')

    if (isinstance(colors[0], Sized) and
        (len(colors[0]) == 2) and
        (not isinstance(colors[0], str))):  # List of value, color pairs
        vals, colors = zip(*colors)
    else:
        vals = np.linspace(0, 1, len(colors))

    cdict = dict(red=[], green=[], blue=[], alpha=[])
    for val, color in zip(vals, colors):
        r, g, b, a = matplotlib.colors.to_rgba(color)
        cdict['red'].append((val, r, r))
        cdict['green'].append((val, g, g))
        cdict['blue'].append((val, b, b))
        cdict['alpha'].append((val, a, a))

    return matplotlib.colors.LinearSegmentedColormap(name, cdict, N, gamma)


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


def draw_map(I, R, lon_ticks, catalog=None, show_labels=False, show_axis=False):
    """
    Parameters
    ==========
    I : :py:class:`~numpy.ndarray`
        (3, N_px)
    R : :py:class:`~numpy.ndarray`
        (3, N_px)
    """
    import mpl_toolkits.basemap as basemap
    import matplotlib.tri as tri

    _, R_el, R_az = transform.cart2eq(*R)
    R_el, R_az = wrapped_rad2deg(R_el, R_az)
    R_el_min, R_el_max = np.around([np.min(R_el), np.max(R_el)])
    R_az_min, R_az_max = np.around([np.min(R_az), np.max(R_az)])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bm = basemap.Basemap(projection='mill',
                         llcrnrlat=R_el_min, urcrnrlat=R_el_max,
                         llcrnrlon=R_az_min, urcrnrlon=R_az_max,
                         resolution='c',
                         ax=ax)

    if show_axis:
        bm_labels = [1, 0, 0, 1]
    else:
        bm_labels = [0, 0, 0, 0]
    bm.drawparallels(np.linspace(R_el_min, R_el_max, 5),
                    color='w', dashes=[1, 0], labels=bm_labels, labelstyle='+/-',
                    textcolor='#565656', zorder=0, linewidth=2)
    bm.drawmeridians(lon_ticks,
                    color='w', dashes=[1, 0], labels=bm_labels, labelstyle='+/-',
                    textcolor='#565656', zorder=0, linewidth=2)

    if show_labels:
        ax.set_xlabel('Azimuth (degrees)', labelpad=20)
        ax.set_ylabel('Elevation (degrees)', labelpad=40)

    R_x, R_y = bm(R_az, R_el)
    triangulation = tri.Triangulation(R_x, R_y)

    N_px = I.shape[1]
    mycmap = cmap_from_list('mycmap', I_rgb.T, N=N_px)
    colors_cmap = np.arange(N_px)
    ax.tripcolor(triangulation, colors_cmap, cmap=mycmap,
                 shading='gouraud', alpha=0.9, edgecolors='w', linewidth=0.1)

    if catalog is not None:
        _, sky_el, sky_az = transform.cart2eq(*catalog.xyz)
        sky_el, sky_az = wrapped_rad2deg(sky_el, sky_az)
        sky_x, sky_y = bm(sky_az, sky_el)
        ax.scatter(sky_x, sky_y, c='w', s=5)

    return fig, ax


if __name__ == '__main__':
    args = parse_args()

    D = [nn.DataSet.from_file(str(_)) for _ in args['datasets']]
    P = [np.load(_) for _ in args['parameters']]

    import matplotlib
    if args['mode'] == 'disk':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    style_path = pathlib.Path('data', 'io', 'imot_tools.mplstyle')
    style_path = pkg.resource_filename('imot_tools', str(style_path))
    matplotlib.style.use(style_path)

    R = D[0].R
    for idx_img in tqdm.tqdm(args['img_idx']):
        I = get_field(D, P, idx_img, img_type=args['img_type'])

        I_rgb = to_RGB(I)
        I_rgb /= I_rgb.max()

        # Filter field to lie in specified interval
        _, R_lat, R_lon = transform.cart2eq(*R)
        _, R_lon_d = wrapped_rad2deg(R_lat, R_lon)
        min_lon, max_lon = args['lon_ticks'].min(), args['lon_ticks'].max()
        mask_lon = (min_lon <= R_lon_d) & (R_lon_d <= max_lon)

        R_field = transform.eq2cart(1, R_lat[mask_lon], R_lon[mask_lon])
        I_rgb = I_rgb[:, mask_lon]

        sky_model = D[0].ground_truth[idx_img]
        fig, ax = draw_map(I_rgb, R_field,
                           lon_ticks=args['lon_ticks'],
                           catalog=sky_model if args['show_catalog'] else None,
                           show_labels=not args['hide_labels'],
                           show_axis=not args['hide_axis'])

        if args['mode'] == 'disk':
            img_type = args['img_type'].lower()
            f_out = args['out'] / f'I_{img_type}_idx_{idx_img:05d}.png'
            fig.savefig(f_out, transparent=True, dpi=300)
        else:
            fig.suptitle(f'idx\_img = {idx_img}')
            fig.show()
