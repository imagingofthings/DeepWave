# ############################################################################
# export_ablation_results.py
# ==========================
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Export ablation study results to a CSV file.
"""

import pathlib
import re

import numpy as np
import pandas as pd

import deepwave.nn as nn
import deepwave.nn.crnn as crnn
import deepwave.tools.math.graph as graph


def process(folder_path):
    dataset_path = folder_path / 'D.npz'
    D = nn.DataSet.from_file(str(dataset_path))
    R_laplacian, _ = graph.laplacian_exp(D.R, normalized=True)
    N_antenna = D.XYZ.shape[1]
    N_px = D.R.shape[1]

    param_path = [_ for _ in folder_path.iterdir()
                    if re.search(r"D_train_[01][01][01].npz", str(_))]
    df = []
    for file in param_path:
        pattern = r"D_train_([01])([01])([01]).npz"
        m = re.search(pattern, str(file))
        fix_mu, fix_D, fix_tau = map(lambda _: bool(int(_)), m.group(1, 2, 3))

        P = np.load(file)
        idx_opt = np.argmin(P['v_loss'])

        parameter = crnn.Parameter(N_antenna, N_px, int(P['K']))
        ridge_loss = crnn.D_RidgeLossFunction(float(P['D_lambda']), parameter)
        laplacian_loss = crnn.LaplacianLossFunction(R_laplacian, float(P['tau_lambda']), parameter)

        p_opt = P['p_opt'][idx_opt]
        x0 = np.zeros((N_px,))
        v_loss = (P['v_loss'][idx_opt] -
                    ridge_loss.eval(p_opt, x0) -
                    laplacian_loss.eval(p_opt, x0))

        df.append(pd.DataFrame({'fix_mu': fix_mu,
                                'fix_D': fix_D,
                                'fix_tau': fix_tau,
                                'v_loss': v_loss},
                                index=pd.RangeIndex(1)))
    df = (pd.concat(df, ignore_index=True,)
          .set_index(['fix_mu', 'fix_D', 'fix_tau']))
    df_all = (df.assign(v_loss_rel=df['v_loss'].values /
                        df.at[(True, True, True), 'v_loss'])
              .sort_values(by='v_loss_rel'))
    return df_all

if __name__ == '__main__':
    folder_path = pathlib.Path(__file__).parent / "dataset"
    df = process(folder_path)

    result_path = folder_path / 'ablation_study.csv'
    df.to_csv(result_path)
