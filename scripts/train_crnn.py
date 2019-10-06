# ############################################################################
# train_crnn.py
# =============
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Stochastic Gradient Descent on dataset.
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='RNN training with SGD.',
        epilog=r"""
    Example
    -------
    python3 train_crnn.py --dataset=D.npz          \
                          --parameter=D_train.npz  \
                          --D_lambda=1.0           \
                          --tau_lambda=1.0         \
                          --N_layer=7              \
                          --psf_threshold=1e-6     \
                          --tanh_lin_limit=1.0     \
                          --loss=relative-l2       \
                          --tv_ratio=0.2           \
                          --lr=1e-3                \
                          --mu=0.9                 \
                          --N_epoch=20             \
                          --batch_size=256
                """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--dataset',
                        help='.npz DataSet.',
                        required=True,
                        type=str)
    parser.add_argument('--parameter',
                        help='.npz file to which trained parameters are stored.',
                        required=True,
                        type=str)

    parser.add_argument('--D_lambda',
                        help='lambda parameter for ridge regularizer on D term.',
                        required=True,
                        type=float)
    parser.add_argument('--tau_lambda',
                        help='lambda parameter for Laplacian regularizer on tau term.',
                        required=True,
                        type=float)

    parser.add_argument('--N_layer',
                        help='Number of layers in RNN.',
                        required=True,
                        type=int)
    parser.add_argument('--psf_threshold',
                        help='PSF main lobe decay truncation in (0, 1).',
                        required=True,
                        type=float)
    parser.add_argument('--tanh_lin_limit',
                        help='Value past which argument of non-linearity is no longer linear.',
                        required=True,
                        type=float)
    parser.add_argument('--loss',
                        help='Loss function to use.',
                        required=True,
                        type=str,
                        choices=['relative-l2', 'shifted-kl'])

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tv_ratio',
                       help='training set/validation set split ratio. (0, 1)',
                       type=float)
    group.add_argument('--tv_index',
                       help=('.npz file with explicit train set/validation set indices. '
                             'Keys must be named idx_train & idx_test'),
                       type=str)

    parser.add_argument('--lr',
                        help='SGD step size',
                        required=True,
                        type=float)
    parser.add_argument('--mu',
                        help='NAG max momentum coefficient',
                        required=True,
                        type=float)
    parser.add_argument('--N_epoch',
                        help='Total number of iterations over entire dataset.',
                        required=True,
                        type=int)
    parser.add_argument('--batch_size',
                        help='Batch size for training/validation.',
                        required=True,
                        type=int)

    parser.add_argument('--fix_mu',
                        help=r'Lock \mu parameter to its initial value.',
                        action='store_true')
    parser.add_argument('--fix_D',
                        help='Lock D parameter to its initial value.',
                        action='store_true')
    parser.add_argument('--fix_tau',
                        help=r'Lock \tau parameter to its initial value.',
                        action='store_true')
    parser.add_argument('--random_initializer',
                        help=('If provided, distribute SGD initial point for free parameters '
                              'as Gaussian (with suitable variance). '
                              'Frozen parameters are set to their APGD-optimal values.'),
                        action='store_true')
    parser.add_argument('--seed',
                        help=('Seed for random number generator. '
                              'If unspecified use machine state to generate random outcomes.'),
                        default=None,
                        type=int)

    args = parser.parse_args()
    args.dataset = pathlib.Path(args.dataset).expanduser().absolute()
    args.parameter = pathlib.Path(args.parameter).expanduser().absolute()
    if args.tv_index is not None:
        args.tv_index = pathlib.Path(args.tv_index).expanduser().absolute()

    # Overwrite user's choice of D_lambda, tau_lambda depending of --fix_[D, tau] choices.
    if args.fix_D:
        args.D_lambda = 0
    if args.fix_tau:
        args.tau_lambda = 0

    return args


def train_network(args):
    """
    Parameters
    ----------
    args : :py:class:`~argparse.Namespace`

    Returns
    -------
    opt : dict
        p_opt : :py:class:`~numpy.ndarray`
            (N_epoch + 1, N_cell) optimized parameter per epoch.
            `p_opt[0] = p_apgd`
        iter_loss : :py:class:`~numpy.ndarray`
            (N_epoch, N_batch) loss function value per (epoch, batch) on
            training set.
        t_loss : :py:class:`~numpy.ndarray`
            (N_epoch + 1,) loss function value per epoch on training set.
        v_loss : :py:class:`~numpy.ndarray`
            (N_epoch + 1,) loss function value per epoch on validation set.
        t : :py:class:`~numpy.ndarray`
            (N_epoch,) execution time [s] per epoch.
            Includes time to compute training/validation loss.
        idx_t : :py:class:`~numpy.ndarray`
            (N_k1,) sample indices used for training set.
        idx_v : :py:class:`~numpy.ndarray`
            (N_k2,) sample indices used for validation set.
        K : int
            Order of polynomial filter.
        D_lambda : float
        tau_lambda : float
        N_layer : int
        psf_threshold : float
        tanh_lin_limit : float
        lr : float
        mu : float
        batch_size : int
    """
    if args.seed is not None:
        np.random.seed(args.seed)

    D = nn.DataSet.from_file(str(args.dataset))
    A = phased_array.steering_operator(D.XYZ, D.R, D.wl)
    N_antenna, N_px = A.shape
    sampler = nn.Sampler(N_antenna, N_px)

    # Set optimization initial point.
    p_apgd, K = crnn.APGD_Parameter(D.XYZ, D.R, D.wl,
                                    lambda_=np.median(D.lambda_),
                                    gamma=D.gamma,
                                    L=2 * pylinalg.eighMax(A),
                                    eps=args.psf_threshold)
    parameter = crnn.Parameter(N_antenna, N_px, K)
    p0 = p_apgd.copy()
    if args.random_initializer:
        p_mu, p_D, p_tau = parameter.decode(p0)
        if not args.fix_mu:
            mu_step = np.abs(p_mu[~np.isclose(p_mu, 0)]).min()
            p_mu[:] = mu_step * np.random.randn(K + 1)
        if not args.fix_tau:
            tau_step = np.abs(p_tau[~np.isclose(p_tau, 0)]).min()
            p_tau[:] = tau_step * np.random.randn(N_px)
        if not args.fix_D:
            D_step = np.abs(p_D[~np.isclose(p_D, 0)]).min() / 2  # because complex-valued.
            p_D[:] = D_step * (     np.random.randn(N_antenna, N_px) +
                               1j * np.random.randn(N_antenna, N_px))

    R_laplacian, _ = graph.laplacian_exp(D.R, normalized=True)

    afunc = (lambda x: func.retanh(args.tanh_lin_limit, x),
             lambda x: func.d_retanh(args.tanh_lin_limit, x))
    trainable_parameter = (('mu', not args.fix_mu),
                           ('D', not args.fix_D),
                           ('tau', not args.fix_tau))
    sample_loss = crnn.SampleLossFunction(args.N_layer, parameter, sampler, R_laplacian,
                                          args.loss, afunc, trainable_parameter)
    ridge_loss = crnn.D_RidgeLossFunction(args.D_lambda, parameter)
    laplacian_loss = crnn.LaplacianLossFunction(R_laplacian, args.tau_lambda, parameter)
    sgd_solver = optim.StochasticGradientDescent(func=[sample_loss, ridge_loss, laplacian_loss],
                                                 batch_size=args.batch_size,
                                                 N_epoch=args.N_epoch,
                                                 alpha=args.lr,
                                                 mu=args.mu,
                                                 verbosity='HIGH')

    log_fname = (pathlib.Path(args.parameter.parent) /
                 (args.parameter.stem + ".log"))
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
    logging.info(str(args))

    ### Dataset Preprocessing: drop all-0 samples + permutation
    _, I, _ = sampler.decode(D[:])
    sample_mask = ~np.isclose(I.sum(axis=1), 0)
    if args.tv_index is None:  # Random split
        idx_valid = np.flatnonzero(sample_mask)
        idx_sample = np.random.permutation(idx_valid)

        N_sample = len(idx_valid)
        idx_ts = idx_sample[int(N_sample * args.tv_ratio):]
        idx_vs = idx_sample[:int(N_sample * args.tv_ratio)]
    else:  # Deterministic split
        idx_tv = np.load(args.tv_index)
        if not (('idx_train' in idx_tv) and ('idx_test' in idx_tv)):
            raise ValueError('Parameter[tv_index] does not have keys "idx_train" and "idx_test".')
        idx_ts = idx_tv['idx_train']
        if not (argcheck.has_integers(idx_ts) and
                np.all((0 <= idx_ts) & (idx_ts < len(D)))):
            raise ValueError('Specified "idx_ts" values must be integer and in {0, ..., len(D) - 1}.')
        idx_vs = idx_tv['idx_test']
        if not (argcheck.has_integers(idx_vs) and
                np.all((0 <= idx_vs) & (idx_vs < len(D)))):
            raise ValueError('Specified "idx_vs" values must be integer and in {0, ..., len(D) - 1}.')

        idx_invalid = np.flatnonzero(~sample_mask)
        idx_ts = np.setdiff1d(idx_ts, idx_invalid)
        idx_vs = np.setdiff1d(idx_vs, idx_invalid)

    D_ts = nn.DataSet(D[idx_ts], D.XYZ, D.R, D.wl,
                      ground_truth=[D.ground_truth[idx] for idx in idx_ts],
                      lambda_=np.array([np.median(D.lambda_[idx_ts])] * len(idx_ts)),
                      gamma=D.gamma,
                      N_iter=D.N_iter[idx_ts],
                      tts=D.tts[idx_ts])
    D_vs = nn.DataSet(D[idx_vs], D.XYZ, D.R, D.wl,
                      ground_truth=[D.ground_truth[idx] for idx in idx_vs],
                      lambda_=np.array([np.median(D.lambda_[idx_vs])] * len(idx_vs)),
                      gamma=D.gamma,
                      N_iter=D.N_iter[idx_vs],
                      tts=D.tts[idx_vs])
    out = sgd_solver.fit(D_ts, D_vs, p0, file_name=args.parameter)

    # Augment output with extra information.
    out = dict(**out,
               D_lambda=args.D_lambda,
               tau_lambda=args.tau_lambda,
               N_layer=args.N_layer,
               psf_threshold=args.psf_threshold,
               tanh_lin_limit=args.tanh_lin_limit,
               lr=args.lr,
               mu=args.mu,
               batch_size=args.batch_size,
               K=K,
               idx_t=idx_ts,
               idx_v=idx_vs,
               loss=args.loss)
    return out


if __name__ == '__main__':
    args = parse_args()
    opt = train_network(args)
    np.savez(args.parameter, **opt)
