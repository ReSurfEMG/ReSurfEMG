"""
Copyright 2022 Netherlands eScience Center and Twente University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains one method to let the user configure
all paths for data instead of hard-coding them, as well
as methods to check data integrity, and create synthetic data.
The data integrity can be checked because this file contains
hash functions to track data. Synthetic data can be made
with several methods.
"""

import json
import logging
import os
import textwrap
import hashlib
import glob
import math
import pandas as pd
import numpy as np
import scipy
from scipy import signal
import random


class Config:
    """
    This class allows configuration on the home computer
    or remote workspace, of a file setup for data,
    which is then processed into a variable. Essentially
    by setting up and modifying a .json file in the appropriate directory
    users can avoid the need for any hardcoded paths to data.
    """

    default_locations = (
        './config.json',
        os.path.expanduser('~/.resurfemg/config.json'),
        '/etc/resurfemg/config.json',
    )

    default_layout = {
        'root_emg_directory': '{}',
        'preprocessed': '{}/preprocessed',
        'models': '{}/models',
        'output': '{}/output',
    }

    required_directories = ['root_emg_directory']

    def __init__(self, location=None):
        self._raw = None
        self._loaded = None
        self.load(location)
        self.validate()

    def usage(self):
        """
        This is essentally a corrective error message if the computer
        does not have paths configured or files made so that
        the data paths of config.json can be used
        """
        return textwrap.dedent(
            '''
            Cannot load config.

            Please create a file in either one of the locations
            listed below:
            {}

            With the contents that specifies at least the root
            directory as follows:

            {{
                "root_emg_directory": "/path/to/storage"
            }}

            The default directory layout is expected to be based on the above
            and adding subdirectories.

            You can override any individual directory (or subdirectory)
            by specifying it in the config.json file.

            "root_emg_directory" is expected to exist.
            The "models" and "preprocessed" directories need not
            exist.  They will be created if missing.
            '''
        ).format('\n'.join(self.default_locations))

    def load(self, location):
        locations = (
            [location] if location is not None else self.default_locations
        )

        for p in locations:
            try:
                with open(p) as f:
                    self._raw = json.load(f)
                    break
            except Exception as e:
                logging.info('Failed to load %s: %s', p, e)
        else:
            raise ValueError(self.usage())

        root = self._raw.get('root_emg_directory')
        self._loaded = dict(self._raw)
        if root is None:
            required = dict(self.default_layout)
            del required['root_emg_directory']
            for directory in required.keys():
                if directory not in self._raw:
                    raise ValueError(self.usage())
            # User specified all concrete directories.  Nothing for us to
            # do here.
        else:
            missing = set(self.default_layout.keys()) - set(self._raw.keys())
            # User possibly specified only a subset of directories.  We'll
            # back-fill all the not-specified directories.
            for m in missing:
                self._loaded[m] = self.default_layout[m].format(root)

    def validate(self):
        for d in self.required_directories:
            if not os.path.isdir(self._loaded[d]):
                logging.error('Directory %s must exist', self._loaded[d])
                raise ValueError(self.usage())

    def get_directory(self, directory, value=None):
        if value is None:
            return self._loaded[directory]
        return value


def hash_it_up_right_all(origin_folder1, file_extension):
    """Hashing function to check files are not corrupted or to assure
    files are changed.

    :param origin_folder1: The string of the folder with files to hash
    :type origin_folder1: str
    :param file_extension: File extension
    :type file_extension: str

    :returns: Dataframe with hashes for what is in folder
    :rtype: ~pandas.DataFrame
    """
    hash_list = []
    file_names = []
    files = '*' + file_extension
    non_suspects1 = glob.glob(os.path.join(origin_folder1, files))
    BUF_SIZE = 65536
    for file in non_suspects1:
        sha256 = hashlib.sha256()
        with open(file, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        result = sha256.hexdigest()
        hash_list.append(result)
        file_names.append(file)

    df = pd.DataFrame(hash_list, file_names)
    df.columns = ["hash"]
    df = df.reset_index()
    df = df.rename(columns={'index': 'file_name'})

    return df


def make_synth_emg(long, max_abs_volt, humps):
    """
    Function to create a synthetic EMG,
    not to add longer expirations (relatively flat)
    the results can be fed to another function.

    :param long: The legnth in samples of synth EMG created
    :type long: int
    :param max_abs_volt: desired abs voltage maximum
    :type max_abs_volt: float
    :param humps: desired number of inspiratory waves
    :type humps: int

    :returns: signal
    :rtype: ~np.array
    """
    x = np.linspace(0, (long/500), long)
    raised_sin = np.sin(x*humps/60*2*math.pi)**2
    synth_emg1 = raised_sin * np.random.randn(
        (len(x))) + 0.1 * np.random.randn(len(x))
    volt_multiplier = max_abs_volt / abs(synth_emg1).max()
    signal = synth_emg1*volt_multiplier
    return signal


def simulate_emg_with_occlusions(t_start=0,
                                 t_end=7*60,
                                 emg_sample_rate=2048,   # hertz
                                 rr=22,         # respiratory rate /min
                                 ie_ratio=1/2,  # ratio btw insp + expir phase
                                 tau_mus_up=0.3,
                                 tau_mus_down=0.3,
                                 occs_times_vals=[6*60+5, 6*60+21, 6*60+35]):
    """
    This function simulates an surface respiraotry emg with no ecg
    component but with occlusion manuevers.
    An ECG component can be added and mixed in later.
    """
    ie_fraction = ie_ratio/(ie_ratio + 1)
    occs_times = np.array(occs_times_vals)
    t_occs = np.floor(occs_times*rr/60)*60/rr
    for i, t_occ in enumerate(t_occs):
        if t_end < (t_occ + 60/rr):
            printable1 = 't=' + str(t_occ) + ':t_occ'
            printable2 = 'should be at least a full resp. cycle from t_end'
            print(printable1 + printable2)
    # time axis
    esr = emg_sample_rate
    y_emg = np.array(
        [i/esr for i in range(int(t_start*esr), int(t_end*esr))]
    )

    # reference signal pattern generator
    emg_block = (signal.square(y_emg*rr/60*2*np.pi + 0.5, ie_fraction)+1)/2
    for i, t_occ in enumerate(t_occs):
        esr = emg_sample_rate
        n_occ = int(t_occ*emg_sample_rate)
        blocker = np.arange(int(esr*60/rr)+1)/esr*rr/60*2*np.pi
        squared_wave = (signal.square(blocker, ie_fraction)+1)/2
        emg_block[n_occ:n_occ+int(esr*60/rr)+1] = squared_wave

    # simulate up- and downslope dynamics of EMG
    pattern_gen_emg = np.zeros((len(y_emg),))

    for i in range(1, len(y_emg)):
        pat = pattern_gen_emg[i-1]
        esr = emg_sample_rate
        if (emg_block[i-1]-pat) > 0:
            pattern_gen_emg[i] = pat + (emg_block[i-1]-pat)/(tau_mus_up*esr)
        else:
            pattern_gen_emg[i] = pat + (emg_block[i-1]-pat)/(tau_mus_down*esr)

    # make respiratory EMG component
    part_emg = pattern_gen_emg * np.random.normal(0, 0.5, size=(len(y_emg), ))

    # make noise and drift components
    part_noise = np.random.normal(0, 0.5, size=(len(y_emg), ))
    part_drift = np.zeros((len(y_emg),))

    # mix channels, could be remixed with an ecg
    x_emg = np.zeros((3, len(y_emg)))
    x_emg[0, :] = 0.05 * part_emg + 1 * part_noise + 20 * part_drift
    x_emg[1, :] = 4 * part_emg + 1 * part_noise + 20 * part_drift
    x_emg[2, :] = 8 * part_emg + 1 * part_noise + 20 * part_drift
    data_emg_samples = x_emg
    return data_emg_samples


def make_realistic_syn_emg(loaded_ecg, number):
    """
    This function makes realistic synthetic respiratory EMG data.
    :param loaded_ecg: synthetic emg/s as numpy array
    :type loaded_ecg: np.array

    :returns: list_emg
    :rtype: list
    """
    list_emg = []
    number = int(number)  # added for cli
    for i in list(range(number)):
        emg = simulate_emg_with_occlusions(
            t_start=0,
            t_end=7*60,
            emg_sample_rate=2048,   # hertz
            rr=22,         # respiratory rate /min
            ie_ratio=1/2,  # ratio btw insp + expir phase
            tau_mus_up=0.3,
            tau_mus_down=0.3,
            occs_times_vals=[365, 381, 395]
        )
        emg1 = emg[0][:307200]
        emg2 = emg[1][:307200]
        emg3 = emg[2][:307200]
        emg_stack = np.vstack((emg1, emg2))
        emg_stack = np.vstack((emg_stack, emg3))
        heart_line = random.randint(0, 9)
        one_line_ecg = loaded_ecg[heart_line]
        x_emg = np.zeros((3, emg_stack.shape[1]))
        ecg_out = np.array(200*one_line_ecg, dtype='float64')
        x_emg[0] = ecg_out + np.array(0.05 * emg_stack[0], dtype='float64')
        x_emg[1] = ecg_out + np.array(4 * emg_stack[1], dtype='float64')
        x_emg[2] = ecg_out + np.array(8 * emg_stack[2], dtype='float64')
        list_emg.append(x_emg)
    return list_emg


def make_realistic_syn_emg_cli(file_directory, number, made):
    """
    This function works with the cli
    module to makes realistic synthetic respiratory EMG data
    through command line.

    :param file_directory: file directory where synthetic ecg are
    :type file_directory: str
    :param number: file directory where synthetic ecg are
    :type number: int
    :param made: file directory where synthetic emg will be put
    :type made: str
    """
    file_directory_list = glob.glob(
        os.path.join(file_directory, '*.npy'),
        recursive=True,
    )
    file = file_directory_list[0]
    loaded = np.load(file)
    synthetics = make_realistic_syn_emg(loaded, number)
    number_end = 0
    for single_synth in synthetics:
        out_fname = os.path.join(made, str(number_end))
        if not (os.path.exists(made)):
            os.mkdir(made)
        np.save(out_fname, single_synth)
        number_end += 1
