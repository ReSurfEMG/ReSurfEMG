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

from .multi_lead_type import preprocess
from .ml import applu_model
from .config import Config
from .config import make_realistic_syn_emg_cli


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
    acquire.add_argument(
        '-p',
        '--preprocessing',
        default='working_pipeline_pre_ml_multi',
        choices=(
            'alternative_a_pipeline_multi',
            'alternative_b_pipeline_multi',
            'working_pipeline_pre_ml_multi'),
        type=str,
        help='''
        Pick the desired algorithm for preprocessing.
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

    ml = subparsers.add_parser('ml')
    ml.set_defaults(action='ml')
    common(ml)

    ml.add_argument(
        '-V',
        '--verbose',
        choices=tuple(range(10)),
        default=0,
        help='''
        Verbosity of mne, scikit etc. libraries.
        '''
    )
    ml.add_argument(
        '-e',
        '--features',
        action='append',
        default=['mean', 'entropy'],
        help='''
        Features used in ML. Note mean and entropy are base, add others.
        '''
    )

    ml.add_argument(
        '-m',
        '--model',
        # choices=('svm', 'dt', 'lr'),
        help='''
        ML model/algorithm to use.
        '''
    )

    # ml.add_argument(
    #     'fit',
    #     choices=('fit', 'grid_search', 'best_fit'),
    #     help='''
    #     Action performed by the selected algorithm.  If `best_fit' is
    #     selected, the algorithm will train the model using previously
    #     established best parameters.  If `grid_search' is selected,
    #     will re-run the grid search.  If `fit' is selected will run
    #     the algorithm with the default optimization strategy (using
    #     random search).
    #     '''
    # )
    return parser


def main(argv):
    """
    This runs the parser and subparsers.
    """
    parser = make_parser()
    parsed = parser.parse_args()
    config = Config(parsed.config)

    if parsed.action == 'acquire':
        try:

            preprocess(
                config.get_directory('data', parsed.input),
                parsed.lead or [0, 2],  # list of chosen leads
                parsed.preprocessing,
                config.get_directory('preprocessed', parsed.output),
                parsed.force,
            )
        except Exception as e:
            logging.exception(e)
            return 1

    if parsed.action == 'synth':
        try:

            make_realistic_syn_emg_cli(
                config.get_directory('data', parsed.input),
                parsed.number,
                config.get_directory('made', parsed.output),
            )
        except Exception as e:
            logging.exception(e)
            return 1

    if parsed.action == 'ml':
        try:
            applu_model(
                config.get_directory('data', parsed.input),
                parsed.model,
                config.get_directory('preprocessed', parsed.output),
                parsed.features,
            )
        except Exception as e:
            logging.exception(e)
            return 1

    return 0
