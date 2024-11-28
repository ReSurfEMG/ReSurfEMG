"""
Copyright 2022 Netherlands eScience Center and Twente University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods to let the user configure all paths for data
instead of hard-coding them, as well as methods to check data integrity.
The data integrity can be checked because this file contains hash functions
to track data. Synthetic data can be made with several methods.
"""

import json
import logging
import os
import textwrap
import hashlib
import glob
import pandas as pd


def convert_to_os_path(
    path: str,
):
    """
    This function converts a path to a os readable path.
    -----------------------------------------------------------------------
    :param path: The path to convert.
    :type path: str
    """
    readable_path = path.replace(
        os.sep if os.altsep is None else os.altsep, os.sep)
    return readable_path


def find_repo_root(marker_file='config_example.json'):
    """
    Find the root directory of the repository by looking for a marker file.
    :param marker_file: The marker file to look for (default is
    'config_example.json').
    -----------------------------------------------------------------------
    :type marker_file: str
    :return: The absolute path to the root directory of the repository.
    :rtype: str
    """
    current_dir = os.path.abspath(os.path.dirname(__file__))

    while True:
        if os.path.exists(os.path.join(current_dir, marker_file)):
            return current_dir
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if parent_dir == current_dir:
            print(f"Marker file '{marker_file}' not found in any parent "
                  + "directory.")
            return None
        current_dir = parent_dir


class Config:
    """
    This class allows configuration on the home computer or remote workspace,
    of a file setup for data, which is then processed into a variable.
    Essentially by setting up and modifying a .json file in the appropriate
    directory users can avoid the need for any hardcoded paths to data.
    -----------------------------------------------------------------------
    """

    required_directories = ['root_data']

    def __init__(self, location=None, verbose=False):
        self._raw = None
        self._loaded = None
        self.example = 'config_example_resurfemg.json'
        self.repo_root = find_repo_root(self.example)
        self.created_config = False
        # In the ResurfEMG project, the test data is stored in ./test_data
        test_path = os.path.join(self.repo_root, 'test_data')
        if len(glob.glob(test_path)) == 1:
            test_data_path = os.path.join(self.repo_root, 'test_data')
        else:
            test_data_path = '{}/test_data'
        if self.repo_root is None:
            self.default_locations = (
                './config.json',
                os.path.expanduser('~/.resurfemg/config.json'),
                '/etc/resurfemg/config.json',
            )
        else:
            self.default_locations = (
                './config.json',
                os.path.expanduser('~/.resurfemg/config.json'),
                '/etc/resurfemg/config.json',
                os.path.join(self.repo_root, 'config.json'),
            )
        self.default_layout = {
                'root_data': '{}/not_pushed',
                'test_data': test_data_path,
                'patient_data': '{}/patient_data',
                'simulated_data': '{}/simulated',
                'preprocessed_data': '{}/preprocessed',
                'output_data': '{}/output',
            }
        self.load(location, verbose=verbose)
        self.validate()

    def usage(self):
        """
        Provide feedback if the paths are not configured or not configured
        correctly. It contains instructions on how to configure the paths.
        -----------------------------------------------------------------------
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
                "root_data": "{}"
            }}

            The default directory layout is expected to be based on the above
            `root_data` directory and adding subdirectories.

            You can override any individual directory (or subdirectory) by
            specifying it in the config.json file.

            "root_data" is expected to exist.
            The "patient_data", "simulated_data", "preprocessed_data",
            "output_data" are optional. They will be created if missing.
            '''
        ).format(convert_to_os_path('\n'.join(self.default_locations)),
                 convert_to_os_path('/path/to/storage'))

    def create_config_from_example(
        self,
        location: str,
    ):
        """
        This function creates a config file from an example file.
        -----------------------------------------------------------------------
        :param location: The location of the example file.
        :type location: str
        :raises ValueError: If the example file is not found.
        """
        config_path = location.replace(self.example, 'config.json')
        with open(location, 'r') as f:
            example = json.load(f)
        with open(config_path, 'w') as f:
            json.dump(example, f, indent=4, sort_keys=True)

    def load(self, location, verbose=False):
        """
        This function loads the configuration file. If no location is specified
        it will try to load the configuration file from the default locations:
        - ./config.json
        - ~/.resurfemg/config.json
        - /etc/resurfemg/config.json
        - PROJECT_ROOT/config.json
        -----------------------------------------------------------------------
        :param location: The location of the configuration file.
        :type location: str
        :param verbose: A boolean to print the loaded configuration.
        :type verbose: bool
        :raises ValueError: If the configuration file is not found.
        """
        locations = (
            [location] if location is not None else self.default_locations
        )

        for _path in locations:
            try:
                with open(_path) as f:
                    self._raw = json.load(f)
                    break
            except Exception as e:
                logging.info('Failed to load %s: %s', _path, e)
        else:
            if (self.repo_root is not None and os.path.isfile(
                    os.path.join(self.repo_root, 'config.json'))):
                self.create_config_from_example(
                    os.path.join(self.repo_root, self.example))
                root_path = os.path.join(self.repo_root, 'not_pushed')
                if not os.path.isdir(root_path):
                    os.makedirs(root_path)
                    print(f'Created root directory at:\n {root_path}\n')
                with open(os.path.join(self.repo_root, 'config.json')) as f:
                    self._raw = json.load(f)
                self.created_config = True
            else:
                raise ValueError(self.usage())

        root = self._raw.get('root_data')
        self._loaded = dict(self._raw)
        config_path = os.path.abspath(_path.replace('config.json', ''))
        if isinstance(root, str) and root.startswith('.'):
            root = root.replace('.', config_path, 1)
        root = convert_to_os_path(root)
        self._loaded['root_data'] = root
        if root is None:
            required = dict(self.default_layout)
            del required['root_data']
            for directory in required:
                if directory not in self._raw:
                    raise ValueError(self.usage())
            # User specified all required directories.
        else:
            for key, value in self._raw.items():
                if isinstance(value, str) and value.startswith('.'):
                    new_value = value.replace('.', config_path, 1)
                    self._loaded[key] = convert_to_os_path(new_value)
                else:
                    self._loaded[key] = convert_to_os_path(value)
            # User possibly specified only a subset of optional directories.
            # The missing directories will be back-filled with the default
            # layout relative to the root directory.
            missing = set(self.default_layout.keys()) - set(self._raw.keys())
            for m in missing:
                self._loaded[m] = convert_to_os_path(
                    self.default_layout[m].format(root))

        if self.created_config:
            print(f'Created config. See and edit it at:\n {_path}\n')
        elif verbose:
            print(f'Loaded config from:\n {_path}\n')
        if verbose or self.created_config:
            print('The following paths were configured:')
            print(79*'-')
            print(f' {"Name": <15}\t{"Path": <50}')
            print(79*'-')
            print(f' {"root": <15}\t{root: <50}')
            for key, value in self._loaded.items():
                if key != 'root_data':
                    print(f' {key: <15}\t{value: <50}')

    def validate(self):
        """
        This function validates the configuration file. It checks if the
        required directories exist.
        -----------------------------------------------------------------------
        :raises ValueError: If a required directory does not exist.
        """
        for req_dir in self.required_directories:
            if not os.path.isdir(self._loaded[req_dir]):
                logging.error('Required directory %s does not exist',
                              self._loaded[req_dir])
                raise ValueError(self.usage())

    def get_directory(self, directory, value=None):
        """
        This function returns the directory specified in the configuration
        file. If the directory is not specified, it will return the value.
        -----------------------------------------------------------------------
        :param directory: The directory to return.
        :type directory: str
        :param value: The value to return if the directory is not specified.
        :type value: str
        :return: The directory specified in the configuration file.
        :rtype: str
        """
        if value is None:
            if directory in self._loaded:
                return self._loaded[directory]
            else:
                print(f"Directory `{directory}` not found in config. The "
                      "following directories are configured:"
                      )
                print(79*'-')
                print(f' {"Name": <15}\t{"Path": <50}')
                print(79*'-')
                print(f' {"root": <15}\t{self._loaded["root_data"]: <50}')
                for key, value in self._loaded.items():
                    if key != 'root_data':
                        print(f' {key: <15}\t{value: <50}')
        return value


def hash_it_up_right_all(origin_directory, file_extension):
    """Hashing function to check files are not corrupted or to assure
    files are changed. This function hashes all files in a directory.
    -----------------------------------------------------------------------
    :param origin_directory: The string of the directory with files to hash
    :type origin_directory: str
    :param file_extension: File extension
    :type file_extension: str

    :returns df: The hash values of the files
    :rtype df: pandas.DataFrame
    """
    hash_list = []
    file_names = []
    files = '*' + file_extension
    non_suspects1 = glob.glob(os.path.join(origin_directory, files))
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
