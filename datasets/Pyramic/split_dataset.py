# ############################################################################
# split_dataset.py
# ================
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Generate a .npz file of train/test indices where the train set explicitly misses some azimuths.
"""

import argparse
import ast
import logging
import pathlib
import sys

import numpy as np

import deepwave.nn as nn
import deepwave.nn.crnn as crnn
import deepwave.tools.math.func as func
import deepwave.tools.math.graph as graph
import deepwave.tools.math.linalg as pylinalg
import deepwave.tools.math.optim as optim
import imot_tools.util.argcheck as argcheck
import imot_tools.phased_array as phased_array

import helpers


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training preprocessor.',
        epilog=r"""
    Example
    -------
    python3 split_dataset.py --dataset=D.npz          \
                             --out=D_train_idx.npz    \
                             --tv_ratio=0.2           \
                             --test_only_angles=list(range(0, 358, 2))[::18]
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
    parser.add_argument('--test_only_angles',
                        help='List of angles in range(0, 360, 2) that are to only remain in test set.',
                        required=True,
                        type=str)
    parser.add_argument('--seed',
                        help=('Seed for random number generator. '
                              'If unspecified use machine state to generate random outcomes.'),
                        default=None,
                        type=int)

    args = parser.parse_args()
    args.dataset = pathlib.Path(args.dataset).expanduser().absolute()
    args.out = pathlib.Path(args.out).expanduser().absolute()

    angles = np.unique(eval(args.test_only_angles))
    all_in_range = np.isclose(angles.reshape(-1, 1),
                              np.arange(0, 360, 2).reshape(1, -1)).any(axis=1).all()
    if not all_in_range:
        raise ValueError('Parameter[test_only_angles] contains out-of-range indices.')
    args.test_only_angles = angles

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

    # Filter idx_ts to only keep angles not specified
    test_only_sky = helpers.merged_sky(args.test_only_angles)
    idx_to_keep = []
    for idx in idx_ts:
        sky = D.ground_truth[idx]
        if not np.isclose(test_only_sky.xyz.T @ sky.xyz, 1).any():
            # Keep if directions in sample do not match any of test sky
            idx_to_keep.append(idx)
    idx_ts = np.array(idx_to_keep)

    out = dict(idx_train=idx_ts,
               idx_test=idx_vs,)
    return out


if __name__ == '__main__':
    args = parse_args()
    idx_tv = preprocess(args)
    np.savez(args.out, **idx_tv)
