# ############################################################################
# inspect_crnn.py
# ===============
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Inspect/Analyse trained CRNN.
"""

import argparse
import pathlib
import time

import acoustic_camera.apgd as apgd
import acoustic_camera.nn as nn
import acoustic_camera.nn.crnn as crnn
import acoustic_camera.spectral as spectral
import acoustic_camera.tools.instrument as instrument
import acoustic_camera.tools.io.image as image
import acoustic_camera.tools.io.plot as plot
import acoustic_camera.tools.math.func as func
import acoustic_camera.tools.math.graph as graph
import acoustic_camera.tools.math.linalg as pylinalg
import acoustic_camera.tools.math.sphere as sphere
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pkg_resources as pkg
import scipy.linalg as linalg

style_path = pathlib.Path('data', 'tools', 'io', 'plot', 'siml_style.mplstyle')
style_path = pkg.resource_filename('acoustic_camera', str(style_path))
matplotlib.style.use(style_path)


def e(i: int, N: int):
    v = np.zeros((N,))
    v[i] = 1
    return v


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Compare APGD/RNN outputs after training.',
                                     epilog=r"""
    Example
    -------
    python3 inspect_crnn.py --dataset=D.npz           \
                            --parameter=D_train.npz   \
                            --show_reconstruction=1
                                             """,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--dataset',
                        help='.npz file holding APGD ground-truth.',
                        required=True,
                        type=str)
    parser.add_argument('--parameter',
                        help='.npz file holding trained CRNN parameters. Output of `train_crnn.py`.',
                        required=True,
                        type=str)
    parser.add_argument('--projection',
                        help='Spherical projection to use. (Default="AEQD")',
                        required=False,
                        default='AEQD',
                        type=str)
    parser.add_argument('--show_loss',
                        help='Plot training/validation loss.',
                        action='store_true')
    parser.add_argument('--show_parameters',
                        help='Compare APGD/RNN parameters.',
                        action='store_true')
    parser.add_argument('--show_reconstruction',
                        help='Plot APGD/RNN reconstructions for specified image index.',
                        type=int,
                        default=None)
    parser.add_argument('--interpolation_order',
                        help=('Interpolate zonal functions with Dirichlet kernel of order N. '
                              'It is assumed that grids in datasets are fibonacci grids.'),
                        default=None,
                        type=int)

    args = parser.parse_args()
    args.dataset = pathlib.Path(args.dataset).expanduser().absolute()
    args.parameter = pathlib.Path(args.parameter).expanduser().absolute()

    info = vars(args)
    return info


def plot_loss(info: dict):
    T = np.load(info['parameter'])  # Trained

    fig = plt.figure()
    fig.suptitle('Training Progression')
    ax_epoch = fig.add_subplot(121)
    ax_batch = fig.add_subplot(122)
    ax_epoch.plot(T['t_loss'], label='training loss')
    ax_epoch.plot(T['v_loss'], label='validation loss')
    ax_epoch.set_xlabel('epoch')
    ax_epoch.set_ylabel('loss')
    ax_epoch.legend()

    ax_batch.plot(T['iter_loss'].reshape(-1))
    ax_batch.set_xlabel('batch/epoch')
    ax_batch.set_ylabel('batch loss')

    return fig, dict(EPOCH=ax_epoch, BATCH=ax_batch)


def plot_parameters(info: dict):
    def draw_apgd_psf(D, ax):
        R_focus = np.mean(D.R, axis=1)
        R_focus /= linalg.norm(R_focus)

        A = instrument.steering_operator(D.XYZ, D.R, D.wl)
        alpha = 1 / (2 * pylinalg.eighMax(A))
        beta = 2 * np.median(D.lambda_) * alpha * (1 - D.gamma) + 1
        psf = (pylinalg.psf_exp(D.XYZ, D.R, D.wl, center=R_focus) *
               (2 * alpha / beta))

        if info['interpolation_order'] is not None:
            N = info['interpolation_order']
            approximate_kernel = True if (N > 15) else False
            interp = sphere.Interpolator(N, approximate_kernel)
            N_s = N_px = D.R.shape[1]
            psf = interp.__call__(weight=np.ones((N_s,)),
                                  support=D.R,
                                  f=psf.reshape((1, N_px)),
                                  r=D.R)
            psf = np.clip(psf, a_min=0, a_max=None)

        psf_plot = image.SphericalImage(data=psf, grid=D.R)
        psf_plot.draw(projection=info['projection'],
                      use_contours=False,
                      data_kwargs=dict(cmap=plot.magma_cmap(), ),
                      catalog_kwargs=dict(edgecolor='g', ),
                      ax=ax)
        ax.set_title(r'$\Psi_{APGD}(r, r_{0})$')

    def draw_apgd_filter(D, ax):
        R_focus = np.mean(D.R, axis=1)
        R_focus /= linalg.norm(R_focus)
        idx_focus = np.argmax(R_focus @ D.R)

        A = instrument.steering_operator(D.XYZ, D.R, D.wl)
        N_px = D.R.shape[1]
        alpha = 1 / (2 * pylinalg.eighMax(A))
        beta = 2 * np.median(D.lambda_) * alpha * (1 - D.gamma) + 1
        psf = pylinalg.psf_exp(D.XYZ, D.R, D.wl, center=R_focus)
        filter = (e(idx_focus, N_px) - 2 * alpha * psf) / beta

        if info['interpolation_order'] is not None:
            N = info['interpolation_order']
            approximate_kernel = True if (N > 15) else False
            interp = sphere.Interpolator(N, approximate_kernel)
            N_s = N_px = D.R.shape[1]
            filter = interp.__call__(weight=np.ones((N_s,)),
                                     support=D.R,
                                     f=filter.reshape((1, N_px)),
                                     r=D.R)
            filter = np.clip(filter, a_min=0, a_max=None)

        filter_plot = image.SphericalImage(data=filter, grid=D.R)
        filter_plot.draw(projection=info['projection'],
                         use_contours=False,
                         data_kwargs=dict(cmap=plot.magma_cmap(), ),
                         catalog_kwargs=dict(edgecolor='g', ),
                         ax=ax)
        ax.set_title(r'$p_{\theta}^{APGD}\left(\widetilde{L}\right)$')

    def draw_rnn_filter(D, P, ax):
        N_antenna, N_px, K = D.XYZ.shape[1], D.R.shape[1], int(P['K'])
        parameter = crnn.Parameter(N_antenna, N_px, K)

        R_focus = np.mean(D.R, axis=1)
        R_focus /= linalg.norm(R_focus)
        idx_focus = np.argmax(R_focus @ D.R)

        p_vec = P['p_opt'][np.argmin(P['v_loss'])]
        p = dict(zip(['mu', 'D', 'tau'], parameter.decode(p_vec)))

        Ln, _ = graph.laplacian_exp(D.R, normalized=True)
        fltr = graph.ConvolutionalFilter(Ln, K)
        filter = fltr.filter(p['mu'], e(idx_focus, N_px))

        if info['interpolation_order'] is not None:
            N = info['interpolation_order']
            approximate_kernel = True if (N > 15) else False
            interp = sphere.Interpolator(N, approximate_kernel)
            N_s = N_px = D.R.shape[1]
            filter = interp.__call__(weight=np.ones((N_s,)),
                                     support=D.R,
                                     f=filter.reshape((1, N_px)),
                                     r=D.R)
            filter = np.clip(filter, a_min=0, a_max=None)

        filter_plot = image.SphericalImage(data=filter, grid=D.R)
        filter_plot.draw(projection=info['projection'],
                         use_contours=False,
                         data_kwargs=dict(cmap=plot.magma_cmap(), ),
                         catalog_kwargs=dict(edgecolor='g', ),
                         ax=ax)
        ax.set_title(r'$p_{\theta}^{RNN}\left(\widetilde{L}\right)$')

    def draw_rnn_psf(D, P, ax):
        N_antenna, N_px, K = D.XYZ.shape[1], D.R.shape[1], int(P['K'])
        parameter = crnn.Parameter(N_antenna, N_px, K)

        R_focus = np.mean(D.R, axis=1)
        R_focus /= linalg.norm(R_focus)
        idx_focus = np.argmax(R_focus @ D.R)

        p_vec = P['p_opt'][np.argmin(P['v_loss'])]
        p = dict(zip(['mu', 'D', 'tau'], parameter.decode(p_vec)))

        Ln, _ = graph.laplacian_exp(D.R, normalized=True)
        fltr = graph.ConvolutionalFilter(Ln, K)
        filter = fltr.filter(p['mu'], e(idx_focus, N_px))
        psf = np.abs(filter)
        psf[idx_focus] = 0

        if info['interpolation_order'] is not None:
            N = info['interpolation_order']
            approximate_kernel = True if (N > 15) else False
            interp = sphere.Interpolator(N, approximate_kernel)
            N_s = N_px = D.R.shape[1]
            psf = interp.__call__(weight=np.ones((N_s,)),
                                  support=D.R,
                                  f=psf.reshape((1, N_px)),
                                  r=D.R)
            psf = np.clip(psf, a_min=0, a_max=None)

        psf_plot = image.SphericalImage(data=psf, grid=D.R)
        psf_plot.draw(projection=info['projection'],
                      use_contours=False,
                      data_kwargs=dict(cmap=plot.magma_cmap(), ),
                      catalog_kwargs=dict(edgecolor='g', ),
                      ax=ax)
        ax.set_title(r'$\Psi_{RNN}(r, r_{0})$')

    def draw_tau(D, P, ax):
        N_antenna, N_px, K = D.XYZ.shape[1], D.R.shape[1], int(P['K'])
        parameter = crnn.Parameter(N_antenna, N_px, K)

        p_apgd_vec, p_rnn_vec = P['p_opt'][[0, np.argmin(P['v_loss'])]]
        p_apgd = dict(zip(['mu', 'D', 'tau'], parameter.decode(p_apgd_vec)))
        p_rnn = dict(zip(['mu', 'D', 'tau'], parameter.decode(p_rnn_vec)))
        tau_diff = (p_apgd['tau'] - p_rnn['tau']) / linalg.norm(p_apgd['tau'])

        if info['interpolation_order'] is not None:
            N = info['interpolation_order']
            approximate_kernel = True if (N > 15) else False
            interp = sphere.Interpolator(N, approximate_kernel)
            N_s = N_px = D.R.shape[1]
            tau_diff = interp.__call__(weight=np.ones((N_s,)),
                                       support=D.R,
                                       f=tau_diff.reshape((1, N_px)),
                                       r=D.R)

        tau_plot = image.SphericalImage(data=tau_diff, grid=D.R)
        tau_plot.draw(projection=info['projection'],
                      use_contours=False,
                      data_kwargs=dict(cmap=plot.magma_cmap(), ),
                      catalog_kwargs=dict(edgecolor='g', ),
                      ax=ax)
        ax.set_title(r'$\frac{\tau_{APGD} - \tau_{RNN}}{\left\| \tau_{APGD} \right\|_{2}}$')

    D = nn.DataSet.from_file(str(info['dataset']))
    P = np.load(info['parameter'])

    fig = plt.figure()
    ax_filter_apgd = fig.add_subplot(231)
    ax_filter_rnn = fig.add_subplot(232)
    ax_tau = fig.add_subplot(233)
    ax_psf_apgd = fig.add_subplot(234)
    ax_psf_rnn = fig.add_subplot(235)

    draw_apgd_psf(D, ax_psf_apgd)
    draw_apgd_filter(D, ax_filter_apgd)
    draw_rnn_filter(D, P, ax_filter_rnn)
    draw_rnn_psf(D, P, ax_psf_rnn)
    draw_tau(D, P, ax_tau)

    return fig, dict(TAU=ax_tau,
                     APGD_FILTER=ax_filter_apgd,
                     RNN_FILTER=ax_filter_rnn,
                     APGD_PSF=ax_psf_apgd,
                     RNN_PSF=ax_psf_rnn)


def plot_reconstruction(info: dict):
    def draw_apgd(D, ax):
        idx_img = info['show_reconstruction']

        sampler = D.sampler()
        _, I_apgd, _ = sampler.decode(D[idx_img])
        sky_model = D.ground_truth[idx_img]
        tts = D.tts[idx_img]
        N_iter = D.N_iter[idx_img]

        if info['interpolation_order'] is not None:
            N = info['interpolation_order']
            approximate_kernel = True if (N > 15) else False
            interp = sphere.Interpolator(N, approximate_kernel)
            N_s = N_px = D.R.shape[1]
            I_apgd = interp.__call__(weight=np.ones((N_s,)),
                                     support=D.R,
                                     f=I_apgd.reshape((1, N_px)),
                                     r=D.R)
            I_apgd = np.clip(I_apgd, a_min=0, a_max=None)

        apgd_plot = image.SphericalImage(data=I_apgd, grid=D.R)
        apgd_plot.draw(catalog=sky_model,
                       projection=info['projection'],
                       use_contours=False,
                       data_kwargs=dict(cmap=plot.magma_cmap(), ),
                       catalog_kwargs=dict(edgecolor='g', ),
                       ax=ax)
        ax.set_title(f'APGD {N_iter:02d} iter, {tts:.02f} [s]')

    def draw_trunc(D, P, ax):
        idx_img = info['show_reconstruction']

        sampler = D.sampler()
        S, _, I_prev = sampler.decode(D[idx_img])
        sky_model = D.ground_truth[idx_img]

        N_layer = int(P['N_layer'])
        A = instrument.steering_operator(D.XYZ, D.R, D.wl)
        lambda_ = D.lambda_[idx_img]
        I_trunc = apgd.solve(S, A, lambda_=lambda_, gamma=D.gamma, x0=I_prev.copy(), N_iter_max=N_layer)
        tts = I_trunc['time']
        N_iter = I_trunc['niter']
        I_trunc = I_trunc['sol']

        if info['interpolation_order'] is not None:
            N = info['interpolation_order']
            approximate_kernel = True if (N > 15) else False
            interp = sphere.Interpolator(N, approximate_kernel)
            N_s = N_px = D.R.shape[1]
            I_trunc = interp.__call__(weight=np.ones((N_s,)),
                                      support=D.R,
                                      f=I_trunc.reshape((1, N_px)),
                                      r=D.R)
            I_trunc = np.clip(I_trunc, a_min=0, a_max=None)

        trunc_plot = image.SphericalImage(data=I_trunc, grid=D.R)
        trunc_plot.draw(catalog=sky_model,
                        projection=info['projection'],
                        use_contours=False,
                        data_kwargs=dict(cmap=plot.magma_cmap(), ),
                        catalog_kwargs=dict(edgecolor='g', ),
                        ax=ax)
        ax.set_title(f'APGD {N_iter:02d} iter, {tts:.02f} [s]')

    def draw_rnn(D, P, ax):
        idx_img = info['show_reconstruction']

        sampler = D.sampler()
        S, _, I_prev = sampler.decode(D[idx_img])
        sky_model = D.ground_truth[idx_img]

        N_antenna, N_px, K, N_layer = D.XYZ.shape[1], D.R.shape[1], int(P['K']), int(P['N_layer'])
        parameter = crnn.Parameter(N_antenna, N_px, K)
        p_vec = P['p_opt'][np.argmin(P['v_loss'])]
        p = dict(zip(['mu', 'D', 'tau'], parameter.decode(p_vec)))

        Ln, _ = graph.laplacian_exp(D.R, normalized=True)
        rnn_eval = crnn.Evaluator(N_layer, parameter, p_vec, Ln,
                                  lambda _: func.retanh(P['tanh_lin_limit'], _))
        exec_time = time.time()
        I_rnn = rnn_eval(S, I_prev)
        exec_time = time.time() - exec_time

        if info['interpolation_order'] is not None:
            N = info['interpolation_order']
            approximate_kernel = True if (N > 15) else False
            interp = sphere.Interpolator(N, approximate_kernel)
            N_s = N_px = D.R.shape[1]
            I_rnn = interp.__call__(weight=np.ones((N_s,)),
                                    support=D.R,
                                    f=I_rnn.reshape((1, N_px)),
                                    r=D.R)
            I_rnn = np.clip(I_rnn, a_min=0, a_max=None)

        rnn_plot = image.SphericalImage(data=I_rnn, grid=D.R)
        rnn_plot.draw(catalog=sky_model,
                      projection=info['projection'],
                      use_contours=False,
                      data_kwargs=dict(cmap=plot.magma_cmap(), ),
                      catalog_kwargs=dict(edgecolor='g', ),
                      ax=ax)
        ax.set_title(f'RNN {N_layer:02d} iter, {exec_time:.02f} [s]')

    def draw_das(D, ax):
        idx_img = info['show_reconstruction']
        sampler = D.sampler()
        S, _, _ = sampler.decode(D[idx_img])
        sky_model = D.ground_truth[idx_img]

        A = instrument.steering_operator(D.XYZ, D.R, D.wl)
        alpha = 1 / (2 * pylinalg.eighMax(A))
        beta = 2 * D.lambda_[idx_img] * alpha * (1 - D.gamma) + 1

        exec_time = time.time()
        das = spectral.DAS(D.XYZ, S, D.wl, D.R) * 2 * alpha / beta
        exec_time = time.time() - exec_time

        if info['interpolation_order'] is not None:
            N = info['interpolation_order']
            approximate_kernel = True if (N > 15) else False
            interp = sphere.Interpolator(N, approximate_kernel)
            N_s = N_px = D.R.shape[1]
            das = interp.__call__(weight=np.ones((N_s,)),
                                  support=D.R,
                                  f=das.reshape((1, N_px)),
                                  r=D.R)
            das = np.clip(das, a_min=0, a_max=None)

        das_plot = image.SphericalImage(data=das, grid=D.R)
        das_plot.draw(catalog=sky_model,
                      projection=info['projection'],
                      use_contours=False,
                      data_kwargs=dict(cmap=plot.magma_cmap(), ),
                      catalog_kwargs=dict(edgecolor='g', ),
                      ax=ax)
        ax.set_title(f'DAS, {exec_time:.02f} [s]')

    def draw_learned_dirty_image(D, P, ax):
        idx_img = info['show_reconstruction']

        N_antenna, N_px, K = D.XYZ.shape[1], D.R.shape[1], int(P['K'])
        parameter = crnn.Parameter(N_antenna, N_px, K)
        sampler = D.sampler()

        p_rnn_vec = P['p_opt'][np.argmin(P['v_loss'])]
        p_rnn = dict(zip(['mu', 'D', 'tau'], parameter.decode(p_rnn_vec)))

        S, _, _ = sampler.decode(D[idx_img])
        sky_model = D.ground_truth[idx_img]

        Ds, Vs = linalg.eigh(S)
        idx = Ds > 0  # To avoid np.sqrt() issues.
        Ds, Vs = Ds[idx], Vs[:, idx]
        I_learned = linalg.norm(p_rnn['D'].conj().T @ (Vs * np.sqrt(Ds)), axis=1) ** 2

        if info['interpolation_order'] is not None:
            N = info['interpolation_order']
            approximate_kernel = True if (N > 15) else False
            interp = sphere.Interpolator(N, approximate_kernel)
            N_s = N_px = D.R.shape[1]
            I_learned = interp.__call__(weight=np.ones((N_s,)),
                                        support=D.R,
                                        f=I_learned.reshape((1, N_px)),
                                        r=D.R)
            I_learned = np.clip(I_learned, a_min=0, a_max=None)

        learned_plot = image.SphericalImage(data=I_learned, grid=D.R)
        learned_plot.draw(catalog=sky_model,
                          projection=info['projection'],
                          use_contours=False,
                          data_kwargs=dict(cmap=plot.magma_cmap(), ),
                          catalog_kwargs=dict(edgecolor='g', ),
                          ax=ax)
        ax.set_title(r'diag$\left(D^{H} \hat{\Sigma} D\right)$')

    D = nn.DataSet.from_file(str(info['dataset']))
    P = np.load(info['parameter'])
    idx_img = info['show_reconstruction']

    fig = plt.figure()
    fig.suptitle(f'Image {idx_img}/{len(D) - 1}')
    ax_apgd = fig.add_subplot(231)
    ax_trunc = fig.add_subplot(232)
    ax_rnn = fig.add_subplot(233)
    ax_das = fig.add_subplot(223)
    ax_learned = fig.add_subplot(224)

    draw_apgd(D, ax_apgd)
    draw_trunc(D, P, ax_trunc)
    draw_rnn(D, P, ax_rnn)
    draw_das(D, ax_das)
    draw_learned_dirty_image(D, P, ax_learned)

    return fig, dict(APGD=ax_apgd,
                     APGD_TRUNC=ax_trunc,
                     RNN=ax_rnn,
                     DAS=ax_das,
                     DIRTY=ax_learned)


if __name__ == '__main__':
    info = parse_args()

    if info['show_loss']:
        fig_loss, ax_loss = plot_loss(info)
        fig_loss.show()

    if info['show_parameters']:
        fig_param, ax_param = plot_parameters(info)
        fig_param.show()

    if info['show_reconstruction'] is not None:
        fig_recon, ax_recon = plot_reconstruction(info)
        fig_recon.show()

    # Load DataSet and Parameter for easy browsing from an IPython terminal
    D = nn.DataSet.from_file(str(info['dataset']))
    P = np.load(info['parameter'])
    N_antenna, N_px, K = D.XYZ.shape[1], D.R.shape[1], int(P['K'])
    sampler = D.sampler()
    parameter = crnn.Parameter(N_antenna, N_px, K)
