# ############################################################################
# export_dataset.py
# =================
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Extract part of a dataset.
"""

import argparse
import pathlib

import numpy as np

import deepwave.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export a partial dataset.',
        epilog="""
    Example
    -------
    python3 export_dataset.py --dataset=D.npz  \
                              --img_idx='[1, 3, 5]'
                """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--dataset',
                        help='Dataset .npz file',
                        required=True,
                        type=str)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--parameter',
                       help='.npz file holding trained CRNN parameters. Output of `train_crnn.py`.',
                       type=str)
    group.add_argument('--img_idx',
                       help='List of image indices to export.',
                       type=str)
    parser.add_argument('--out',
                        help='.npz file to which trimmed dataset is exported.',
                        required=True,
                        type=str)

    args = parser.parse_args()
    args.dataset = pathlib.Path(args.dataset).expanduser().absolute()
    args.out = pathlib.Path(args.out).expanduser().absolute()
    if args.parameter is not None:
        args.parameter = pathlib.Path(args.parameter).expanduser().absolute()
    else:
        args.img_idx = np.unique(eval(args.img_idx))

    return args


def process(D, img_idx, f_out):
    XYZ, R, wl, gamma = D.XYZ, D.R, D.wl, D.gamma

    data_sub = D[img_idx]
    gt_sub = [D.ground_truth[i] for i in img_idx]
    lambda_sub = D.lambda_[img_idx]
    N_iter_sub = D.N_iter[img_idx]
    tts_sub = D.tts[img_idx]

    D_sub = nn.DataSet(data_sub, XYZ, R, wl, gt_sub, lambda_sub, gamma, N_iter_sub, tts_sub)
    D_sub.to_file(str(f_out))
    return D_sub


if __name__ == '__main__':
    args = parse_args()
    D = nn.DataSet.from_file(str(args.dataset))
    if args.parameter is not None:
        P = np.load(args.parameter)
        idx_img = P['idx_v']
    else:
        idx_img = args.img_idx
    D_sub = process(D, idx_img, args.out)
