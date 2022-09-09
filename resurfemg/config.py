"""
Copyright 2022 Netherlands eScience Center and Twente University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains one method to let the user configure
all paths instead of hard-coding them.
"""

import json
import logging
import os
import textwrap


class Config:

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
