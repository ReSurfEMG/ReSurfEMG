"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains data classes for standardized peak data storage and method
automation.
"""

import numpy as np
import pandas as pd

from resurfemg.postprocessing.event_detection import (
    onoffpeak_baseline_crossing, onoffpeak_slope_extrapolation,
)


class PeaksSet:
    """
    Data class to store, and process peak information.
    """
    def __init__(self, signal, t_data, peak_idxs=None):
        """
        :param signal: 1-dimensional signal data
        :type signal: ~numpy.ndarray
        :param t_data: time axis data
        :type t_data: ~numpy.ndarray
        :param peak_idxs: Indices of peaks
        :type peak_idxs: ~numpy.ndarray
        """
        if isinstance(signal, np.ndarray):
            self.signal = signal
        else:
            raise ValueError("Invalid signal type: 'signal_type'.")

        if isinstance(t_data, np.ndarray):
            self.t_data = t_data
        else:
            raise ValueError("Invalid t_data type: 't_data'.")

        if peak_idxs is None:
            peak_idxs = np.array([])
        elif (isinstance(peak_idxs, np.ndarray)
                and len(np.array(peak_idxs).shape) == 1):
            pass
        elif isinstance(peak_idxs, list):
            peak_idxs = np.array(peak_idxs)
        else:
            raise ValueError("Invalid peak indices: 'peak_s'.")

        self.peak_df = pd.DataFrame(
            data=peak_idxs, columns=['peak_idx'])
        self.quality_values_df = pd.DataFrame(
            data=peak_idxs, columns=['peak_idx'])
        self.quality_outcomes_df = pd.DataFrame(
            data=peak_idxs, columns=['peak_idx'])
        self.time_products = None

    def detect_on_offset(
        self,
        baseline=None,
        method='default',
        fs=None,
        slope_window_s=None
    ):
        """
        Detect the peak on- and offsets. See postprocessing.event_detection
        submodule.
        """
        if baseline is None:
            baseline = np.zeros(self.signal.shape)

        peak_idxs = self.peak_df['peak_idx'].to_numpy()

        if method == 'default' or method == 'baseline_crossing':
            (start_idxs, end_idxs,
                _, _, valid_list) = onoffpeak_baseline_crossing(
                self.signal,
                baseline,
                peak_idxs)

        elif method == 'slope_extrapolation':
            if fs is None:
                raise ValueError('Sampling rate is not defined.')

            if slope_window_s is None:
                # TODO Insert valid default slope window
                slope_window_s = fs // 5

            (start_idxs, end_idxs, _, _,
                valid_list) = onoffpeak_slope_extrapolation(
                self.signal, fs, peak_idxs, slope_window_s)
        else:
            raise KeyError('Detection algorithm does not exist.')

        self.peak_df['start_idx'] = start_idxs
        self.peak_df['end_idx'] = end_idxs
        self.peak_df['valid'] = valid_list
        quality_outcomes_df = self.quality_outcomes_df
        quality_outcomes_df['baseline_detection'] = valid_list

        self.evaluate_validity(quality_outcomes_df)

    def update_test_outcomes(self, tests_df_new):
        """
        Add new peak quality test to self.quality_outcomes_df, and update
        existing entries.
        -----------------------------------------------------------------------
        :param tests_df_new: Dataframe of test parameters per peak
        :type tests_df_new: pandas.DataFrame

        :returns: None
        :rtype: None
        """
        if self.quality_values_df is not None:
            df_old = self.quality_values_df
            pre_existing_keys = list(
                set(tests_df_new.keys()) & set(df_old.keys()))
            pre_existing_keys.pop(pre_existing_keys.index('peak_idx'))
            df_old = df_old.drop(columns=pre_existing_keys)
            tests_df_new = df_old.merge(
                tests_df_new,
                left_on='peak_idx',
                right_on='peak_idx',
                suffixes=(False, False))
            self.quality_values_df = tests_df_new
        else:
            self.quality_values_df = tests_df_new

    def evaluate_validity(self, tests_df_new):
        """
        Update peak validity based on previously and newly executed tests
        in self.quality_outcomes_df.
        -----------------------------------------------------------------------
        :param tests_df_new: Dataframe of passed tests per peak
        :type tests_df_new: pandas.DataFrame

        :returns: None
        :rtype: None
        """
        if self.quality_outcomes_df is not None:
            df_old = self.quality_outcomes_df
            pre_existing_keys = list(
                set(tests_df_new.keys()) & set(df_old.keys()))
            pre_existing_keys.pop(pre_existing_keys.index('peak_idx'))
            df_old = df_old.drop(columns=pre_existing_keys)
            tests_df_new = df_old.merge(
                tests_df_new,
                left_on='peak_idx',
                right_on='peak_idx',
                suffixes=(False, False))
            self.quality_outcomes_df = tests_df_new
        else:
            self.quality_outcomes_df = tests_df_new

        test_keys = list(tests_df_new.keys())
        test_keys.pop(test_keys.index('peak_idx'))
        passed_tests = np.all(
            tests_df_new.loc[:, test_keys].to_numpy(), axis=1)
        self.peak_df['valid'] = passed_tests

    def sanitize(self):
        """
        Delete invalid peak entries (self.peak_df['valid'] is False) from
        self.peak_df, self.quality_values_df, and self.quality_outcomes_df.
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        valid_idxs = np.argwhere(self.peak_df['valid'])

        self.peak_df = self.peak_df.loc[valid_idxs].reset_index(drop=True)
        self.quality_outcomes_df = \
            self.quality_outcomes_df.loc[valid_idxs].reset_index(drop=True)
        self.quality_values_df = \
            self.quality_values_df.loc[valid_idxs].reset_index(drop=True)
