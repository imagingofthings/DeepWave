# ############################################################################
# inspect_dataset.py
# ==================
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Plot an APGD image from a dataset.

Also add DAS image for comparison.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pkg_resources as pkg

import deepwave.nn as nn
import deepwave.spectral as spectral
import deepwave.tools.math.linalg as pylinalg
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.interpolate as interpolate
import imot_tools.phased_array as phased_array


style_path = pathlib.Path('data', 'io', 'imot_tools.mplstyle')
style_path = pkg.resource_filename('imot_tools', str(style_path))
matplotlib.style.use(style_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot APGD image for a given dataset.',
        epilog="""
    Example
    -------
    python3 inspect_dataset.py --dataset=D.npz  \
                               --img_idx=3
                """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--dataset',
                        help='Dataset .npz file',
                        required=True,
                        type=str)

    parser.add_argument('--img_idx',
                        help='Which image to plot.',
                        required=True,
                        type=int)

    parser.add_argument('--projection',
                        help='Spherical projection to use. (Default="AEQD")',
                        required=False,
                        default='AEQD',
                        type=str)

    parser.add_argument('--interpolation_order',
                        help=('Interpolate zonal functions with Dirichlet kernel of order N. '
                              'It is assumed that grids in datasets are fibonacci grids.'),
                        default=None,
                        type=int)

    args = parser.parse_args()
    return args


def process(args):
    file = pathlib.Path(args.dataset).expanduser().absolute()
    if not (file.exists() and (file.suffix == '.npz')):
        raise ValueError('Dataset is non-conformant.')
    D = nn.DataSet.from_file(str(file))
    N_sample = len(D)

    if not (0 <= args.img_idx < N_sample):
        raise ValueError('Parameter[img_idx] is out-of-bounds.')

    S, I_apgd, I_prev = D.sampler().decode(D[args.img_idx])
    I_das = spectral.DAS(D.XYZ, S, D.wl, D.R)

    # Rescale DAS to lie on same range as APGD
    A = phased_array.steering_operator(D.XYZ, D.R, D.wl)
    alpha = 1 / (2 * pylinalg.eighMax(A))
    beta = 2 * D.lambda_[args.img_idx] * alpha * (1 - D.gamma) + 1
    I_das *= (2 * alpha) / beta

    if args.interpolation_order is not None:
        N = args.interpolation_order
        approximate_kernel = True if (N > 15) else False
        interp = interpolate.Interpolator(N, approximate_kernel)
        N_s = N_px = D.R.shape[1]

        I_prev = interp.__call__(weight=np.ones((N_s,)),
                                 support=D.R,
                                 f=I_prev.reshape((1, N_px)),
                                 r=D.R)
        I_prev = np.clip(I_prev, a_min=0, a_max=None)

        I_apgd = interp.__call__(weight=np.ones((N_s,)),
                                 support=D.R,
                                 f=I_apgd.reshape((1, N_px)),
                                 r=D.R)
        I_apgd = np.clip(I_apgd, a_min=0, a_max=None)

        I_das = interp.__call__(weight=np.ones((N_s,)),
                                support=D.R,
                                f=I_das.reshape((1, N_px)),
                                r=D.R)
        I_das = np.clip(I_das, a_min=0, a_max=None)

    fig = plt.figure()
    ax_prev = fig.add_subplot(131)
    ax_apgd = fig.add_subplot(132)
    ax_das = fig.add_subplot(133)

    s2image.Image(I_prev, D.R).draw(catalog=D.ground_truth[args.img_idx].xyz,
                                    projection=args.projection,
                                    catalog_kwargs=dict(edgecolor='g', ),
                                    ax=ax_prev)
    ax_prev.set_title(r'$APGD_{init}$')

    s2image.Image(I_apgd, D.R).draw(catalog=D.ground_truth[args.img_idx].xyz,
                                    projection=args.projection,
                                    catalog_kwargs=dict(edgecolor='g', ),
                                    ax=ax_apgd)
    ax_apgd.set_title('APGD')

    s2image.Image(I_das, D.R).draw(catalog=D.ground_truth[args.img_idx].xyz,
                                   projection=args.projection,
                                   catalog_kwargs=dict(edgecolor='g', ),
                                   ax=ax_das)
    ax_das.set_title('DAS')

    fig.show()


if __name__ == '__main__':
    args = parse_args()
    process(args)
