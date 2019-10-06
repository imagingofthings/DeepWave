import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd
import pkg_resources as pkg


style_path = pathlib.Path('data', 'io', 'imot_tools.mplstyle')
style_path = pkg.resource_filename('imot_tools', str(style_path))
matplotlib.style.use(style_path)


"""
Old script to generate N_layer vs. precision plots. Will not run after calling './run.sh' unless
other network depths are trained as well.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Form N_layer vs. precision plots.',
                                     epilog=r"""
    Example
    -------
    python3 inspect_crnn.py --folder=./dataset
                                             """,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--folder',
                        help='folder containing trained datasets.',
                        required=True,
                        type=str)

    args = parser.parse_args()
    folder_path = pathlib.Path(args.folder).expanduser().absolute()
    if not (folder_path.exists() and folder_path.is_dir()):
        raise ValueError('Parameter[folder] must be a valid folder.')

    return folder_path

def process(path):
    # Get all files
    files = [_ for _ in path.iterdir()
             if (('_train_' in _.stem) and
                 ('.npz' == _.suffix))]

    data = []
    for f in files:
        d = np.load(f)

        pattern = r'D_freq(?P<freq_idx>\d+)_(?P<initialization>\w+)_train_\d+.npz'
        m = re.search(pattern, str(f))
        if m is not None:
            freq_idx = m.group('freq_idx')
            initialization = m.group('initialization')

            loss_function = str(d['loss'])
            D_lambda = float(d['D_lambda'])
            tau_lambda = float(d['tau_lambda'])
            N_layer = int(d['N_layer'])
            precision = d['v_loss'][np.argmin(d['v_loss'])]

            data.append([loss_function, D_lambda, tau_lambda, N_layer, precision, freq_idx, initialization])
    data = (pd.DataFrame(data, columns=['loss_function', 'D_lambda', 'tau_lambda', 'N_layer', 'precision', 'freq_idx', 'initialization'])
            .set_index(['loss_function', 'D_lambda', 'tau_lambda', 'freq_idx', 'initialization']))

    for lf, df in data.groupby(level='loss_function'):
        for fi, ddf in df.groupby(level='freq_idx'):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f'freq\\_idx{fi} {lf} loss')

            for (dl, tl, init), dddf in ddf.groupby(level=['D_lambda', 'tau_lambda', 'initialization']):
                clean = dddf.sort_values(by='N_layer')
                ax.plot(clean['N_layer'], clean['precision'],
                        label=(r'$(\lambda_{D}, \lambda_{\tau}, init) = $'
                               f'({dl:.02f}, {tl:.02f}, {init})'))
            ax.legend()
            ax.set_xlabel('$N_{layer}$')
            ax.set_ylabel('precision')
            ax.set_xticks(np.unique(data['N_layer']))

            f_name = f'{lf}_freq{fi}.png'
            fig.savefig(f_name, dpi=300)
    return data

if __name__ == '__main__':
    folder_path = parse_args()
    data = process(folder_path)
