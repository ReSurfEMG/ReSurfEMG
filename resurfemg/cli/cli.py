# -*- coding: utf-8 -*-

"""
Copyright 2022 Netherlands eScience Center and U. Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions designed to help with command line
interface for reproduction of previous work. Here we are building
APIs for pre-processing and machine learning at speed.
"""

import logging

from argparse import ArgumentParser

from resurfemg.helper_functions.config import Config
from resurfemg.helper_functions.config import make_realistic_syn_emg_cli
from resurfemg.data_connector.converter_functions import save_j_as_np


def common(parser):
    """
    This function defines some arguments that can be called from any command
    line function to be defined.
    """
    parser.add_argument(
        '-i',
        '--input',
        default=None,
        help='''
        Directory containing files to be worked on
        ''',
    )
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help='''
        Directory containing algorithm output (created if doesn't exist).
        ''',
    )


def make_parser():
    """
    This is the setting up parser for our CLI.
    """
    parser = ArgumentParser('ReSurfEMG CLI')
    parser.add_argument(
        '-c',
        '--config',
        default=None,
        help='''
        Location of config.json, a file that specified directory layout.
        This file is necessary to locate the data directory,
        models and preprocessed data.
        '''
    )
    subparsers = parser.add_subparsers()
    acquire = subparsers.add_parser('acquire')
    acquire.set_defaults(action='acquire')

    acquire.add_argument(
        '-f',
        '--force',
        action='store_true',
        default=False,
        help='''
        Write over previously preprpocessed data.
        ''',
    )
    acquire.add_argument(
        '-l',
        '--lead',
        action='append',
        default=[],
        type=int,
        help='''
        Accumulate leads for chosen leads desired in preprocessing.
        ''',
    )
    common(acquire)

    synth = subparsers.add_parser('synth')
    synth.set_defaults(action='synth')
    common(synth)
    synth.add_argument(
        '-N',
        '--number',
        default=1,
        help='''
        Number of synthetic EMG to be made.
        '''
    )

    save_np = subparsers.add_parser('save_np')
    save_np.set_defaults(action='save_np')
    common(save_np)
    return parser


def main(argv):
    """
    This runs the parser and subparsers.
    """
    parser = make_parser()
    parsed = parser.parse_args()

    path_in = parsed.input
    path_out = parsed.output

    # if the paths are not specified, use the config file
    if (path_in is None) or (path_out is None):
        try:
            config = Config(parsed.config)
            path_in = config.get_directory('data', path_in)
            path_out = config.get_directory('preprocessed', path_out)
        except Exception as e:
            logging.exception(e)
            return 1

    if parsed.action == 'save_np':
        try:

            save_j_as_np(
                path_in,
                path_out,
            )
        except Exception as e:
            logging.exception(e)
            return 1

    if parsed.action == 'synth':
        try:

            make_realistic_syn_emg_cli(
                path_in,
                parsed.number,
                path_out,
            )
        except Exception as e:
            logging.exception(e)
            return 1

    return 0
