# ############################################################################
# merge_dataset.py
# ================
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Combine multiple datasets to form large dataset.
"""

import argparse
import pathlib

import numpy as np

import deepwave.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Form aggregate dataset by combining many small datasets.',
        epilog="""
    Example
    -------
    python3 merge_dataset.py --out='~/D_all.npz' \
                             D_1.npz D_2.npz D_3.npz
                """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--out',
                        help='Name of file to which merged dataset is stored.',
                        required=True,
                        type=str)
    parser.add_argument('dataset', nargs='+')

    args = parser.parse_args()
    args.out = pathlib.Path(args.out).expanduser().absolute()
    args.dataset = [pathlib.Path(_).expanduser().absolute()
                    for _ in args.dataset]
    return args


def merge_dataset(D, f_out):
    XYZ, R, wl, gamma = D[0].XYZ, D[0].R, D[0].wl, D[0].gamma
    if not all([np.allclose(XYZ, d.XYZ) for d in D]):
        raise ValueError('Cannot merge datasets over different configurations.')
    if not all([np.allclose(R, d.R) for d in D]):
        raise ValueError('Cannot merge datasets over different configurations.')
    if not all([np.allclose(wl, d.wl) for d in D]):
        raise ValueError('Cannot merge datasets over different configurations.')
    if not all([np.isclose(gamma, d.gamma) for d in D]):
        raise ValueError('Cannot merge datasets over different configurations.')

    data_merge = np.concatenate([d._data for d in D], axis=0)
    gt_merge = [gt for d in D for gt in d.ground_truth]
    lambda_merge = np.concatenate([d.lambda_ for d in D], axis=0)
    N_iter_merge = np.concatenate([d.N_iter for d in D], axis=0)
    tts_merge = np.concatenate([d.tts for d in D], axis=0)

    D_merge = nn.DataSet(data_merge, XYZ, R, wl, gt_merge,
                         lambda_merge, gamma, N_iter_merge, tts_merge)
    D_merge.to_file(str(f_out))
    return D_merge


if __name__ == '__main__':
    args = parse_args()
    D = [nn.DataSet.from_file(str(_)) for _ in args.dataset]
    D_merged = merge_dataset(D, args.out)
