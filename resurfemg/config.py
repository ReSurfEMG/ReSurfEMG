"""
Copyright 2022 Netherlands eScience Center and Twente University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains one method to let the user configure
all paths for data instead of hard-coding them, as well
as a methods to check data integrity, and create synthetic data.
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

            You can override any individual directory by specifying it
            in config.json.

            "root_emg_data" and directories are expected to exist.
            The "models" and "preprocessed" directories need not
            exist.  They will be created if missing.
            '''
        ).format('\n'.join(self.default_locations))

    def load(self, location):
        if location is not None:
            with open(location) as f:
                self._raw = json.load(f)
                return

        for p in self.default_locations:
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
