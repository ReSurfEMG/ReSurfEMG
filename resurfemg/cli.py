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

# from .cnt import preprocess
# from .ml import Regressions
# from .loaders import RegressionsLoader
from .config import Config


def common(parser):
    parser.add_argument(
        '-i',
        '--input',
        default=None,
        help='''
        Directory containing files generated in preprocessing step
        ''',
    )
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help='''
        Directory containing trained models (will be created if doesn't exist).
        ''',
    )
    parser.add_argument(
        '-s',
        '--size',
        default=None,
        type=int,
        help='''
        Number of samples to use instead of the entire dataset.  Note that
        grid search may be particularly slow with large datasets.  If you
        only want to try this method, it's best to limit it to a small
        number of samples.  This will set both the training and the testing
        sets to the same number.
        ''',
    )


def make_parser():
    parser = ArgumentParser('ResurfEMG CLI')
    parser.add_argument(
        '-c',
        '--config',
        default=None,
        help='''
        Location of config.json, a file that specified directory layout.
        This file is necessary to locate the data directory, metadata,
        models and preprocessed data.
        '''
    )
    subparsers = parser.add_subparsers()
    acquire = subparsers.add_parser('acquire')
    acquire.set_defaults(action='acquire')
    # acquire.add_argument(
    #     '-i',
    #     '--input',
    #     default=None,
    #     help='''
    #     Input directory
    #     (the one containing `11mnd mmn' etc. directories)''',
    # )
    acquire.add_argument(
        '-o',
        '--output',
        default=None,
        help='''
        Output directory.  Will be created if doesn't exist.
        This is where newly created files will go.
        ''',
    )
    acquire.add_argument(
        '-m',
        '--metadata',
        default=None,
        help='''
        Metadata directory.  This is the directory for metadata.
        ''',
    )
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
        '--limit',
        type=int,
        default=None,
        help='''
        Limit the preprocessed data to N first records.
        ''',
    )

    ml = subparsers.add_parser('ml')
    ml.set_defaults(action='ml')
    common(ml)

    ml.add_argument(
        '-j',
        '--jobs',
        default=-1,
        type=int,
        help='''
        Training is run using `joblib' package.  This parameter translates into
        `jobs' parameter when creating the pipeline (it controls concurrency of
        the training task).
        ''',
    )
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
        '-u',
        '--no-use-joblib',
        action='store_false',
        default=True,
        help='''
        Whether to use joblib when fitting or using searches.
        '''
    )
    ml.add_argument(
        'algo',
        choices=('dummy', 'rf', 'lsv', 'sgd', 'emrvr'),
        help='''
        Regression algorithm to use.
        '''
    )
    ml.add_argument(
        'fit',
        choices=('fit', 'grid_search', 'best_fit'),
        help='''
        Action performed by the selected algorithm.  If `best_fit' is
        selected, the algorithm will train the model using previously
        established best parameters.  If `grid_search' is selected,
        will re-run the grid search.  If `fit' is selected will run
        the algorithm with the default optimization strategy (using
        random search).
        '''
    )

    nn = subparsers.add_parser('nn')
    nn.set_defaults(action='nn')
    common(nn)
    nn.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=1500,
        help='''
        Number of training epochs to run.
        '''
    )
    nn.add_argument(
        'operation',
        choices=('fit_model', 'predict'),
    )
    nn.add_argument(
        'nth_model',
        choices=tuple(range(len(NnOptimizer.optimization_params))),
        type=int,
        help='''
        Nth model to use.
        '''
    )

    return parser


def prepare_loader(parsed, config):
    rloader = RegressionsLoader(
        config.get_directory('preprocessed', parsed.input),
        config.get_directory('models', parsed.output),
        parsed.size,
    )
    rloader.load()
    rloader.split()
    return rloader


def main(argv):
    parser = make_parser()
    parsed = parser.parse_args()
    config = Config(parsed.config)

    if parsed.action == 'acquire':
        try:
            preprocess(
                config.get_directory('data', parsed.input),
                config.get_directory('metadata', parsed.metadata),
                config.get_directory('preprocessed', parsed.output),
                parsed.limit,
                parsed.force,
            )
        except Exception as e:
            logging.exception(e)
            return 1

    if parsed.action == 'ml':
        try:
            rloader = prepare_loader(parsed, config)

            regressions = Regressions(
                rloader,
                parsed.verbose,
                parsed.no_use_joblib,
            )
            algo = regressions.algorithms[parsed.algo]
            getattr(algo, parsed.fit)()
        except Exception as e:
            logging.exception(e)
            return 1

    return 0
