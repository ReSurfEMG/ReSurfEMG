"""
Copyright 2022 Netherlands eScience Center and U. Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions designed to help with command line interface for
reproduction of previous work. Here we are building APIs for pre-, and post-
processing.
"""

import logging

from argparse import ArgumentParser

from resurfemg.data_connector.config import Config
import resurfemg.pipelines.synthetic_data as simulate
from resurfemg.data_connector.converter_functions import save_j_as_np


def set_common_args(parser):
    """
    This function defines some arguments that can be provided to any command
    line function to be defined. See the make_parser function for more details.
    """
    parser.add_argument(
        '-i',
        '--input',
        default=None,
        help='''
        Directory containing the input files
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
        This file is necessary to locate the data directories: root_data,
        simulated_data, patient_data, preprocessed_data, and output_data.
        '''
    )
    subparsers = parser.add_subparsers()
    # Parser for simulating EMG
    sim_emg = subparsers.add_parser('simulate_emg')
    sim_emg.set_defaults(action='simulate')
    set_common_args(sim_emg)
    sim_emg.add_argument(
        '-N',
        '--number',
        default=1,
        help='''
        Number of synthetic EMG to be generated.
        '''
    )
    # Parser for simulating Ventilator data
    sim_vent = subparsers.add_parser('simulate_ventilator')
    sim_vent.set_defaults(action='simulate_ventilator')
    set_common_args(sim_vent)
    sim_vent.add_argument(
        '-N',
        '--number',
        default=1,
        help='''
        Number of synthetic EMG to be generated.
        '''
    )

    save_to_numpy = subparsers.add_parser('save_to_numpy')
    save_to_numpy.set_defaults(action='save_to_numpy')
    set_common_args(save_to_numpy)
    return parser


def main(argv):
    """
    The main function is called from the command line interface. It runs the
    parser and subparsers specified in the make_parser function.
    """
    parser = make_parser()
    parsed = parser.parse_args()

    path_in = parsed.input
    path_out = parsed.output

    # if the paths are not specified, use the config file
    if (path_in is None) or (path_out is None):
        try:
            config = Config(parsed.config)
            path_in = config.get_directory('root_data', path_in)
            path_out = config.get_directory('output_data', path_out)
        except Exception as e:
            print(e)
            logging.exception(e)
            return 1

    if parsed.action == 'save_to_numpy':
        try:
            save_j_as_np(
                path_in,
                path_out,
            )
        except Exception as e:
            print(e)
            logging.exception(e)
            return 1

    if parsed.action == 'simulate_emg':
        try:
            simulate.synthetic_emg_cli(
                int(parsed.number),
                path_out,
            )
        except Exception as e:
            print(e)
            logging.exception(e)
            return 1

    if parsed.action == 'simulate_ventilator':
        try:
            simulate.synthetic_ventilator_data_cli(
                int(parsed.number),
                path_out,
            )
        except Exception as e:
            print(e)
            logging.exception(e)
            return 1

    return 0
