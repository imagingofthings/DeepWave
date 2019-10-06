# ############################################################################
# extract_simulated_dataset.py
# ============================
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Test reconstruction.
"""

import pathlib

import numpy as np
import scipy.linalg as linalg
import skimage.util as skutil

import deepwave.apgd as apgd
import deepwave.nn as nn
import deepwave.tools.data_gen.source as source
import deepwave.tools.data_gen.statistics as statistics
import imot_tools.math.sphere.transform as transform
import imot_tools.phased_array as phased_array

import helpers


def unseen_data_direction(D, area: str, ratio):
    m_speaker = {v: k for (k, v) in helpers.speaker_map().items()}
    src_xyz = np.concatenate([_.xyz for _ in D.ground_truth], axis=1)
    _, src_colat, src_lon = transform.cart2pol(*src_xyz)

    mask = np.concatenate([_.intensity == m_speaker[area] for _ in D.ground_truth], axis=0)
    colat = np.mean(src_colat[mask])  # unique colat
    lon = skutil.view_as_windows(np.unique(src_lon[mask]), (2,), (1,)).mean(axis=1)
    lon = lon[(ratio * len(lon)).astype(int)]  # 3 lons

    return colat, lon

def unseen_simulated_visibility(D, wl):
    high_colat, high_lon = unseen_data_direction(D, 'high', np.r_[0.25, 0.5, 0.72])
    mid_colat, mid_lon = unseen_data_direction(D, 'middle', np.r_[0.1, 0.33, 0.6])
    low_colat, low_lon = unseen_data_direction(D, 'low', np.r_[0.25, 0.45, 0.89])

    m_speaker = {v: k for (k, v) in helpers.speaker_map().items()}
    # vis_I = np.r_[[m_speaker['high']] * len(high_lon),
    #               [m_speaker['middle']] * len(mid_lon),
    #               [m_speaker['low']] * len(low_lon)].flatten()
    vis_I = np.ones((len(high_lon) + len(mid_lon) + len(low_lon),))
    vis_xyz = np.concatenate([transform.pol2cart(1, np.array([high_colat] * len(high_lon)), high_lon),
                              transform.pol2cart(1, np.array([mid_colat] * len(mid_lon)), mid_lon),
                              transform.pol2cart(1, np.array([low_colat] * len(low_lon)), low_lon)], axis=1)
    sky_model = source.SkyModel(vis_xyz, vis_I)

    vis_gen = statistics.VisibilityGenerator(T=50e-3, fs=48000, SNR=10)
    S = vis_gen(D.XYZ, wl, sky_model)
    # Normalize `S` spectrum for scale invariance.
    S_D, S_V = linalg.eigh(S)
    if S_D.max() <= 0:
        S_D[:] = 0
    else:
        S_D = np.clip(S_D / S_D.max(), 0, None)
    S = (S_V * S_D) @ S_V.conj().T

    return S, sky_model

def unseen_simulated_dataset(D, wl):
    S, sky_model = unseen_simulated_visibility(D, wl)

    # to fill: data, lambda_, N_iter, tts
    sampler = D.sampler()
    N_px = D.R.shape[1]
    I_prev = np.zeros((N_px,))
    A = phased_array.steering_operator(D.XYZ, D.R, wl)
    I_apgd = apgd.solve(S, A, gamma=D.gamma, x0=I_prev, verbosity='NONE')
    data = sampler.encode(S=S,
                          I=I_apgd['backtrace'][-1],
                          I_prev=I_apgd['backtrace'][0]).reshape(1, -1)
    lambda_ = np.r_[I_apgd['lambda_']]
    N_iter = np.r_[I_apgd['niter']]
    tts = np.r_[I_apgd['time']]

    D_out = nn.DataSet(data, D.XYZ, D.R, wl, [sky_model], lambda_, D.gamma, N_iter, tts)
    return D_out

if __name__ == '__main__':
    path_dataset = pathlib.Path('./dataset/D_freq4.npz').expanduser().absolute()

    D = nn.DataSet.from_file(str(path_dataset))
    freq = (skutil  # Center frequencies to form images
            .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
            .mean(axis=-1))
    wl = helpers.speed_of_sound() / freq

    # Generate a visibility(sky_model) at unknown source
    # Procedure: for each 3 colats, place 3 sources in between holes.
    for idx_freq in range(len(wl)):
        S_unseen, sky_model_unseen = unseen_simulated_visibility(D, wl[idx_freq])
        D_sim = unseen_simulated_dataset(D, wl[idx_freq])
        D_sim.to_file(f'./dataset/D_freq{idx_freq}_sim.npz')
