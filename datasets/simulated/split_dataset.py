# ############################################################################
# split_dataset.py
# ================
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Generate a .npz file of train/test indices.
"""

import argparse
import pathlib

import numpy as np

import deepwave.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training preprocessor.',
        epilog=r"""
    Example
    -------
    python3 split_dataset.py --dataset=D.npz          \
                             --out=D_train_idx.npz    \
                             --tv_ratio=0.2
                """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--dataset',
                        help='.npz DataSet.',
                        required=True,
                        type=str)
    parser.add_argument('--out',
                        help='.npz file to which train/test indices are stored.',
                        required=True,
                        type=str)
    parser.add_argument('--tv_ratio',
                        help='training set/validation set split ratio. (0, 1)',
                        required=True,
                        type=float)
    parser.add_argument('--seed',
                        help=('Seed for random number generator. '
                              'If unspecified use machine state to generate random outcomes.'),
                        default=None,
                        type=int)

    args = parser.parse_args()
    args.dataset = pathlib.Path(args.dataset).expanduser().absolute()
    args.out = pathlib.Path(args.out).expanduser().absolute()

    return args


def preprocess(args):
    """
    Parameters
    ----------
    args : :py:class:`~argparse.Namespace`

    Returns
    -------
    opt : dict
        idx_train : :py:class:`~numpy.ndarray`
            (N_train,) training indices
        idx_test : :py:class:`~numpy.ndarray`
            (N_test,) test indices
    """
    if args.seed is not None:
        np.random.seed(args.seed)

    D = nn.DataSet.from_file(str(args.dataset))
    sampler = D.sampler()

    _, I, _ = sampler.decode(D[:])
    sample_mask = ~np.isclose(I.sum(axis=1), 0)
    idx_valid = np.flatnonzero(sample_mask)
    idx_sample = np.random.permutation(idx_valid)

    N_sample = len(idx_valid)
    idx_ts = idx_sample[int(N_sample * args.tv_ratio):]
    idx_vs = idx_sample[:int(N_sample * args.tv_ratio)]

    out = dict(idx_train=idx_ts,
               idx_test=idx_vs,)
    return out


if __name__ == '__main__':
    args = parse_args()
    idx_tv = preprocess(args)
    np.savez(args.out, **idx_tv)
