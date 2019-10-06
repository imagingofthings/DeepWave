# ############################################################################
# helpers.py
# ==========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Helper functions.
"""

import io
import json
import pathlib
import re

import imot_tools.math.sphere.transform as transform
import numpy as np

import deepwave.tools.data_gen.source as source


def data():
    """
    Read calibrated JSON data.

    Returns
    =======
    d : dict
        JSON data.
    """
    current_dir = pathlib.Path(__file__).parent
    file_name = current_dir / 'pyramic-dataset' / 'calibration' / 'calibrated_locations.json'

    with io.open(file_name, 'r') as f:
        d = json.load(f)
    return d


def speaker_map():
    """
    Read JSON data file and extract mapping between speaker numbers and names.

    Returns
    =======
    m : dict
        {[012] -> ['low', 'mid', 'high']}
    """
    m_inv = data()['speakers_numbering']
    m = {v : k for (k, v) in m_inv.items()}
    return m


def speed_of_sound():
    """
    Returns
    =======
    c : float
        Speed of sound [m/s]
    """
    c = data()['sound_speed_mps']
    return c


def sky(f):
    """
    Extract Sky Model from file name.

    Parameters
    ==========
    f : :py:class:`~pathlib.Path`
        .wav file with name of the form "<characters>_spkr[012]_angle[0:360:2].wav"

    Returns
    =======
    sky_model : :py:class:`~deepwave.tools.data_gen.source.SkyModel`
        Container with source information.
        `sky_model.intensity` will contain the speaker number {0, 1, 2}.
    """
    pattern = r'.+_spkr(\d+)_angle(\d+)\.wav'
    match = re.search(pattern, str(f))
    if match:
        idx_speaker, idx_angle = map(int, match.group(1, 2))
        if idx_speaker not in (0, 1, 2):
            raise ValueError(f'speaker number {idx_speaker} is out of bounds.')
        if idx_angle not in np.arange(0, 360, 2):
            raise ValueError(f'angle number {idx_angle} is out of bounds.')

        m_speaker = speaker_map()
        colat = data()['sources'][m_speaker[idx_speaker]]['colatitude'][str(idx_angle)]
        lon = data()['sources'][m_speaker[idx_speaker]]['azimuth'][str(idx_angle)]

        src_xyz = transform.pol2cart(1, colat, lon)
        src_I = np.r_[idx_speaker]
        sky_model = source.SkyModel(src_xyz, src_I)
        return sky_model
    else:
        raise ValueError('Parameter[f] does not have the right form.')


def merge_sky(skies):
    """
    Concatenate different source maps together.

    Parameters
    ==========
    skies : list(:py:class:`~deepwave.tools.data_gen.source.SkyModel`)

    Returns
    =======
    sky : :py:class:`~deepwave.tools.data_gen.source.SkyModel`
        Merged sky.
    """
    if not isinstance(skies, list):
        raise ValueError('Parameter[skies] must be a list.')

    sky_I = np.concatenate([_.intensity for _ in skies], axis=0)
    sky_xyz = np.concatenate([_.xyz for _ in skies], axis=1)
    sky = source.SkyModel(sky_xyz, sky_I)
    return sky


def mic_xyz():
    """
    Read JSON data file and extract calibrated microphone coordinates.

    Returns
    =======
    xyz : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian coordinates.
    """
    xyz = np.array(data()['microphones']).T
    return xyz

def merged_sky(azimuths):
    current_dir = pathlib.Path(__file__).parent
    folder_name = [current_dir / 'pyramic-dataset' / 'segmented' / f'fq_sample{_}'
                   for _ in (0, 1, 2, 3, 4)]
    file_name = []
    for folder in folder_name:
        for file in folder.iterdir():
            for lon in azimuths:
                if f'angle{lon}.wav' in str(file):
                    file_name.append(file)

    skies = [sky(_) for _ in file_name]
    all_sky = merge_sky(skies)
    return all_sky
