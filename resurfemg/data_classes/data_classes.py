"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
This file contains data classes for standardized data storage and method
automation.
"""

import warnings

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from resurfemg.preprocessing import filtering as filt
from resurfemg.preprocessing import ecg_removal as ecg_rm
from resurfemg.pipelines.pipelines import ecg_removal_gating
from resurfemg.preprocessing import envelope as evl
from resurfemg.postprocessing.baseline import (
    moving_baseline, slopesum_baseline)
from resurfemg.postprocessing import event_detection as evt
from resurfemg.postprocessing.event_detection import (
    onoffpeak_baseline_crossing, onoffpeak_slope_extrapolation,
    find_occluded_breaths, detect_emg_breaths
)
from resurfemg.postprocessing import features as feat
from resurfemg.postprocessing import quality_assessment as qa


class TimeSeries:
    """
    Data class to store, process, and plot single channel time series data
    """
    class PeaksSet:
        """
        Data class to store, and process peak information.
        """
        def __init__(self, signal, t_data, peak_idxs=None):
            if isinstance(signal, np.ndarray):
                self.signal = signal
            else:
                raise ValueError("Invalid signal type: 'signal_type'.")

            if isinstance(t_data, np.ndarray):
                self.t_data = t_data
            else:
                raise ValueError("Invalid t_data type: 't_data'.")

            if peak_idxs is None:
                self.peak_idxs = np.array([])
            elif (isinstance(peak_idxs, np.ndarray)
                    and len(np.array(peak_idxs).shape) == 1):
                self.peak_idxs = peak_idxs
            elif isinstance(peak_idxs, list):
                self.peak_idxs = np.array(peak_idxs)
            else:
                raise ValueError("Invalid peak indices: 'peak_s'.")

            self.start_idxs = None
            self.end_idxs = None
            self.valid = None
            self.test_values_df = None
            self.test_outcomes_df = None
            self.time_products = None

        def detect_on_offset(
            self, baseline=None,
            method='default',
            fs=None,
            slope_window_s=None
        ):
            """
            Detect the peak on- and offsets. See the documentation in the
            event_detection module on the 'onoffpeak_baseline_crossing' and
            the 'slope_extrapolation' methods for a detailed description.
            """
            if baseline is None:
                baseline = np.zeros(self.signal.shape)

            if method == 'default' or method == 'baseline_crossing':
                (self.peak_idxs, self.start_idxs, self.end_idxs,
                 _, _, valid_list) = onoffpeak_baseline_crossing(
                    self.signal,
                    baseline,
                    self.peak_idxs)

            elif method == 'slope_extrapolation':
                if fs is None:
                    raise ValueError('Sampling rate is not defined.')

                if slope_window_s is None:
                    # TODO Insert valid default slope window
                    slope_window_s = 0.2 * fs

                (self.start_idxs, self.end_idxs, _, _,
                 valid_list) = onoffpeak_slope_extrapolation(
                    self.signal, fs, self.peak_idxs, slope_window_s)
            else:
                raise KeyError('Detection algorithm does not exist.')

            n_peaks = len(self.peak_idxs)
            performed_tests = ['peak_idx', 'baseline_detection']
            tests_outcomes = np.concatenate(
                (np.reshape(np.array([self.peak_idxs]), (n_peaks, 1)),
                 np.reshape(np.array(valid_list), (n_peaks, 1))), axis=1)

            test_outcomes_df = pd.DataFrame(tests_outcomes,
                                            columns=performed_tests)
            self.evaluate_validity(test_outcomes_df)

        def evaluate_validity(self, tests_df_new):
            if self.test_outcomes_df is not None:
                df_old = self.test_outcomes_df
                pre_existing_keys = list(
                    set(tests_df_new.keys()) & set(df_old.keys()))
                pre_existing_keys.pop(pre_existing_keys.index('peak_idx'))
                df_old = df_old.drop(columns=pre_existing_keys)
                tests_df_new = df_old.merge(
                    tests_df_new,
                    left_on='peak_idx',
                    right_on='peak_idx',
                    suffixes=(False, False))
                self.test_outcomes_df = tests_df_new
            else:
                self.test_outcomes_df = tests_df_new

            test_keys = list(tests_df_new.keys())
            test_keys.pop(test_keys.index('peak_idx'))

            n_tests = len(test_keys)
            passed_tests = np.sum(tests_df_new.loc[:, test_keys].values,
                                  axis=1)
            self.valid = passed_tests == n_tests

        def update_test_outcomes(self, tests_df_new):
            if self.test_values_df is not None:
                df_old = self.test_values_df
                pre_existing_keys = list(
                    set(tests_df_new.keys()) & set(df_old.keys()))
                pre_existing_keys.pop(pre_existing_keys.index('peak_idx'))
                df_old = df_old.drop(columns=pre_existing_keys)
                tests_df_new = df_old.merge(
                    tests_df_new,
                    left_on='peak_idx',
                    right_on='peak_idx',
                    suffixes=(False, False))
                self.test_values_df = tests_df_new
            else:
                self.test_values_df = tests_df_new

        def sanitize(self):
            """
            Delete invalid peak entries from the lists.
            """
            invalid_idxs = np.argwhere(self.valid)

            self.peak_idxs = np.delete(self.peak_idxs, invalid_idxs)
            self.start_idxs = np.array(self.start_idxs, invalid_idxs)
            self.end_idxs = np.array(self.end_idxs, invalid_idxs)
            self.valid = np.array(self.valid, invalid_idxs)

    def __init__(self, y_raw, t_data=None, fs=None, label=None, units=None):
        """
        Initialize the main data characteristics:
        :param y_raw: 1-dimensional raw signal data
        :type y_raw: ~numpy.ndarray
        :param t_data: time axis data, if None, generated from fs
        :type t_data: ~numpy.ndarray
        :param fs: sampling rate, if None, calculated from t_data
        :type fs: ~int
        :param labels: list of labels, one per provided channel
        :type labels: ~list of str
        :param units: list of signal units, one per provided channel
        :type units: ~list of str
        """
        self.fs = fs
        data_shape = list(np.array(y_raw).shape)
        data_dims = len(data_shape)
        if data_dims == 1:
            self.n_samp = len(y_raw)
            self.n_channel = 1
            self.y_raw = np.array(y_raw)
        elif data_dims == 2:
            self.n_samp = data_shape[np.argmax(data_shape)]
            if np.argmin(data_shape) == 0:
                self.y_raw = np.array(y_raw).reshape(self.n_samp)
            else:
                self.y_raw = np.array(y_raw)
        else:
            raise ValueError("Invalid y_raw dimensions")

        self.peaks = dict()
        self.y_clean = None
        self.y_env = None
        self.y_baseline = None

        if t_data is None and fs is None:
            self.t_data = np.arange(self.n_samp)
        elif t_data is not None:
            if len(np.array(t_data).shape) > 1:
                raise ValueError
            self.t_data = np.array(t_data)
            if fs is None:
                self.fs = int(1/(t_data[1:]-t_data[:-1]))
        else:
            self.t_data = np.array([x_i/fs for x_i in range(self.n_samp)])

        if label is None:
            self.label = ''
        else:
            self.label = label

        if units is None:
            self.units = '?'
        else:
            self.units = units

    def signal_type_data(self, signal_type=None):
        """
        Automatically select the most advanced data type eligible for a
        subprocess ('envelope' > 'clean' > 'raw')
        :param signal_type: one of 'envelope', 'clean', or 'raw'
        :type signal_type: str

        :returns: y_data
        :rtype: ~numpy.ndarray
        """
        y_data = np.zeros(self.y_raw.shape)
        if signal_type is None:
            if self.y_env is not None:
                y_data = self.y_env
            elif self.y_clean is not None:
                y_data = self.y_clean
            else:
                y_data = self.y_raw
        elif signal_type == 'env':
            if self.y_env is None:
                raise IndexError('No evelope defined for this signal.')
            y_data = self.y_env
        elif signal_type == 'clean':
            if self.y_clean is None:
                warnings.warn("Warning: No clean data availabe, using raw data"
                              + " instead.")
                y_data = self.y_raw
            else:
                y_data = self.y_clean
        else:
            y_data = self.y_raw
        return y_data

    def filter_emg(
        self,
        signal_type='raw',
        hp_cf=20.0,
        lp_cf=500.0,
    ):
        """
        Filter raw EMG signal to remove baseline wander and high frequency
        components.
        """
        y_data = self.signal_type_data(signal_type=signal_type)
        # Eliminate the baseline wander from the data using a band-pass filter
        self.y_clean = filt.emg_bandpass_butter_sample(
            y_data, hp_cf, lp_cf, self.fs, output='sos')

    def gating(
        self,
        signal_type='clean',
        gate_width_samples=None,
        ecg_peak_idxs=None,
        ecg_raw=None,
        bp_filter=True,
    ):
        """
        Eliminate ECG artifacts from the provided signal. See ecg_removal
        submodule in preprocessing.
        """
        y_data = self.signal_type_data(signal_type=signal_type)
        if ecg_peak_idxs is None:
            if ecg_raw is None:
                lp_cf = min([500.0, self.fs / 2])
                ecg_raw = filt.emg_bandpass_butter_sample(
                    self.y_raw, 1, lp_cf, self.fs, output='sos')

            ecg_peak_idxs = ecg_rm.detect_ecg_peaks(
                ecg_raw=ecg_raw,
                fs=self.fs,
                bp_filter=bp_filter,
            )

        self.set_peaks(
            signal=ecg_raw,
            peak_idxs=ecg_peak_idxs,
            peak_set_name='ecg',
        )

        if gate_width_samples is None:
            gate_width_samples = self.fs // 10

        self.y_clean = ecg_removal_gating(
            y_data,
            ecg_peak_idxs,
            gate_width_samples,
            ecg_shift=10,
        )

    def envelope(
        self,
        env_window=None,
        env_type=None,
        signal_type='clean',
    ):
        """
        Derive the moving envelope of the provided signal. See
        envelope submodule in preprocessing.
        """
        if env_window is None:
            if self.fs is None:
                raise ValueError(
                    'Evelope window and sampling rate are not defined.')
            else:
                env_window = int(0.2 * self.fs)

        y_data = self.signal_type_data(signal_type=signal_type)
        if env_type == 'rms' or env_type is None:
            self.y_env = evl.full_rolling_rms(y_data, env_window)
        elif env_type == 'arv':
            self.y_env = evl.full_rolling_arv(y_data, env_window)
        else:
            raise ValueError('Invalid envelope type')

    def baseline(
        self,
        percentile=33,
        window_s=None,
        step_s=None,
        method='default',
        signal_type=None,
        augm_percentile=25,
        ma_window=None,
        perc_window=None,
    ):
        """
        Derive the moving baseline of the provided signal. See
        baseline submodule in postprocessing.
        """
        if window_s is None:
            if self.fs is None:
                raise ValueError(
                    'Baseline window and sampling rate are not defined.')
            else:
                window_s = int(7.5 * self.fs)

        if step_s is None:
            if self.fs is None:
                step_s = 1
            else:
                step_s = self.fs // 5

        if signal_type is None:
            signal_type = 'env'

        y_baseline_data = self.signal_type_data(signal_type=signal_type)
        if method == 'default' or method == 'moving_baseline':
            self.y_baseline = moving_baseline(
                y_baseline_data,
                window_s=window_s,
                step_s=step_s,
                set_percentile=percentile,
            )
        elif method == 'slopesum_baseline':
            if self.fs is None:
                raise ValueError(
                    'Sampling rate is not defined.')
            self.y_baseline, _, _, _ = slopesum_baseline(
                    y_baseline_data,
                    window_s=window_s,
                    step_s=step_s,
                    fs=self.fs,
                    set_percentile=percentile,
                    augm_percentile=augm_percentile,
                    ma_window=ma_window,
                    perc_window=perc_window,
                )
        else:
            raise ValueError('Invalid method')

    def set_peaks(
        self,
        peak_idxs,
        signal,
        peak_set_name,
    ):
        """
        Derive the moving envelope of the provided signal. See
        envelope submodule in preprocessing.
        """
        self.peaks[peak_set_name] = self.PeaksSet(
            peak_idxs=peak_idxs,
            t_data=self.t_data,
            signal=signal)

    def detect_emg_breaths(
        self,
        threshold=0,
        prominence_factor=0.5,
        min_peak_width_s=None,
        peak_set_name='breaths',
    ):
        """
        Find breath peaks in provided EMG envelope signal. See
        event_detection submodule in postprocessing.
        """
        if self.y_env is None:
            raise ValueError('Envelope not yet defined.')

        if self.y_baseline is None:
            warnings.warn('EMG baseline not yet defined. Peak detection '
                          + 'relative to zero.')
            y_baseline = np.zeros(self.y_env.shape)
        else:
            y_baseline = self.y_baseline

        if min_peak_width_s is None:
            min_peak_width_s = self.fs // 5

        peak_idxs = detect_emg_breaths(
            self.y_env,
            y_baseline,
            threshold=threshold,
            prominence_factor=prominence_factor,
            min_peak_width_s=min_peak_width_s,
        )
        self.set_peaks(
            peak_idxs=peak_idxs,
            signal=self.y_env,
            peak_set_name=peak_set_name,
        )

    def link_peak_set(
        self,
        peak_set_name,
        t_reference_peaks,
        linked_peak_set_name=None,
    ):
        """
        Find the peaks in the PeakSet with the peak_set_name closest in time to
        the provided peak timings in t_reference_peaks
        :param peak_set_name: PeakSet name in self.peaks dict
        :type peak_set_name: str
        :param t_reference_peaks: Refernce peak timings in t_reference_peaks
        :type t_reference_peaks: ~numpy.ndarray
        :param linked_peak_set_name: Name of the new PeakSet
        :type linked_peak_set_name: str
        """
        if peak_set_name in self.peaks.keys():
            peak_set = self.peaks[peak_set_name]
        else:
            raise KeyError("Non-existent PeaksSet key")

        if linked_peak_set_name is None:
            linked_peak_set_name = peak_set_name + '_linked'

        t_peakset_peaks = peak_set.peak_idxs / self.fs
        link_peak_nrs = evt.find_linked_peaks(
            t_reference_peaks,
            t_peakset_peaks,
        )
        self.set_peaks(
            peak_idxs=peak_set.peak_idxs[link_peak_nrs],
            signal=self.y_env,
            peak_set_name=linked_peak_set_name,
        )
        linked_peak_set = self.peaks[linked_peak_set_name]
        if peak_set.start_idxs is not None:
            linked_peak_set.start_idxs = peak_set.start_idxs[link_peak_nrs]

        if peak_set.end_idxs is not None:
            linked_peak_set.end_idxs = peak_set.end_idxs[link_peak_nrs]

        if peak_set.valid is not None:
            linked_peak_set.valid = peak_set.valid[link_peak_nrs]

        if peak_set.time_products is not None:
            linked_peak_set.time_products = \
                peak_set.time_products[link_peak_nrs]

        if peak_set.test_outcomes_df is not None:
            test_outcomes_df = peak_set.test_outcomes_df.loc[link_peak_nrs]
            test_outcomes_df = test_outcomes_df.reset_index(drop=True)
            linked_peak_set.test_outcomes_df = test_outcomes_df

        if peak_set.test_values_df is not None:
            test_values_df = peak_set.test_values_df.loc[link_peak_nrs]
            test_values_df = test_values_df.reset_index(drop=True)
            linked_peak_set.test_values_df = test_values_df

    def calculate_time_products(
        self,
        peak_set_name,
        include_aub=True,
        aub_window_s=None,
        aub_reference_signal=None,
    ):
        """
        Calculate the time product, i.e. area under the curve for a PeakSet.
        :param peak_set_name: PeakSet name in self.peaks dict
        :type peak_set_name: str
        :param include_aub: Include the area under the baseline in the
        time product
        :type include_aub: bool
        :type signal_type: str
        :param aub_window_s: window length in samples in which the local
        extreme is sought.
        :param aub_window_s: int
        :param aub_reference_signal: Optional reference signal to find the
        local extreme in, else the signal underlying the peakset is taken.
        :type aub_reference_signal: ~numpy.ndarray
        """
        if peak_set_name in self.peaks.keys():
            peak_set = self.peaks[peak_set_name]
        else:
            raise KeyError("Non-existent PeaksSet key")

        if self.y_baseline is None:
            if include_aub:
                raise ValueError(
                    'Baseline in not yet defined, but is required to calculate'
                    + ' the area under the baseline.')
            else:
                warnings.warn('Baseline in not yet defined. Calculating time-'
                              + 'product with reference to 0.')
                baseline = np.zeros(peak_set.signal.shape)
        else:
            baseline = self.y_baseline

        time_products = feat.time_product(
            signal=peak_set.signal,
            fs=self.fs,
            starts_s=peak_set.start_idxs,
            ends_s=peak_set.end_idxs,
            baseline=baseline,
        )

        if include_aub:
            if aub_window_s is None:
                aub_window_s = 5 * self.fs
            aub = feat.area_under_baseline(
                signal=peak_set.signal,
                fs=self.fs,
                starts_s=peak_set.start_idxs,
                peaks_s=peak_set.peak_idxs,
                ends_s=peak_set.end_idxs,
                aub_window_s=aub_window_s,
                baseline=baseline,
                ref_signal=aub_reference_signal,
            )
            time_products += aub

        peak_set.time_products = time_products

    def test_emg_quality(
        self,
        peak_set_name,
        cutoff=None,
        skip_tests=None,
        verbose=True
    ):
        """
        Test EMG PeakSet according to quality criteria in Warnaar et al. (2024)
        Peak validity is updated in the PeakSet object.
        :param peak_set_name: PeakSet name in self.peaks dict
        :type peak_set_name: str
        :param cutoff: Cut-off criteria for passing the tests, including
        ratio between ECG and EMG interpeak time (interpeak_distance), signal-
        to-noise ratio (snr), percentage area under the baseline (aub), and
        percentage miss fit with bell curve (curve_fit). 'tolerant' and
        'strict' can also be provided instead of a dict to use the respective
        values from Warnaar et al.
        :type cutoff: dict
        :param skip_tests: List of tests to skip.
        :type skip_tests: list(str)
        :param verbose: Output the test values, and pass/fail to console.
        :type verbose: bool
        """
        if peak_set_name in self.peaks.keys():
            peak_set = self.peaks[peak_set_name]
        else:
            raise KeyError("Non-existent PeaksSet key")

        if skip_tests is None:
            skip_tests = []

        if (cutoff is None
                or (isinstance(cutoff, str) and cutoff == 'tolerant')):
            cutoff = dict()
            cutoff['interpeak_distance'] = 1.1
            cutoff['snr'] = 1.4
            cutoff['aub'] = 40
            cutoff['aub_window_s'] = 5*self.fs
            cutoff['curve_fit'] = 0.3
        elif (isinstance(cutoff, str) and cutoff == 'strict'):
            cutoff = dict()
            cutoff['interpeak_distance'] = 1.1
            cutoff['snr'] = 1.75
            cutoff['aub'] = 30
            cutoff['aub_window_s'] = 5*self.fs
            cutoff['curve_fit'] = 0.25
        elif isinstance(cutoff, dict):
            tests = ['interpeak_distance', 'snr', 'aub', 'curve_fit']
            for _, test in enumerate(tests):
                if test not in skip_tests:
                    if test not in cutoff.keys():
                        raise KeyError(
                            'No cut-off value provided for: ' + test)
                    elif isinstance(cutoff, float):
                        raise ValueError(
                            'Invalid cut-off value provided for: ' + test)

            if 'aub' not in skip_tests and 'aub_window_s' not in cutoff.keys():
                raise KeyError('No area under the baseline window provided '
                               + 'for: ' + test)

        n_peaks = len(peak_set.peak_idxs)
        performed_measures = ['peak_idx']
        test_values = np.reshape(np.array([peak_set.peak_idxs]),
                                 (n_peaks, 1))
        performed_tests = ['peak_idx']
        tests_outcomes = np.reshape(np.array([peak_set.peak_idxs]),
                                    (n_peaks, 1))
        if 'interpeak_dist' not in skip_tests:
            if 'ecg' not in self.peaks.keys():
                raise ValueError('ECG peaks not determined, but required for '
                                 + ' interpeak distance evaluation.')
            ecg_peaks = self.peaks['ecg'].peak_idxs
            valid_interpeak = qa.interpeak_dist(
                ECG_peaks=ecg_peaks,
                EMG_peaks=peak_set.peak_idxs,
                threshold=cutoff['interpeak_distance'])

            valid_interpeak_vec = np.array(n_peaks * [valid_interpeak])
            performed_tests.append('interpeak_distance')
            tests_outcomes = np.concatenate(
                (tests_outcomes,
                 np.reshape(valid_interpeak_vec, (n_peaks, 1))), axis=1)

        if 'snr' not in skip_tests:
            if self.baseline is None:
                raise ValueError('Baseline not determined, but required for '
                                 + ' SNR evaluaton.')
            snr_peaks = qa.snr_pseudo(
                src_signal=peak_set.signal,
                peaks=peak_set.peak_idxs,
                baseline=self.y_baseline,
                fs=self.fs,
            )
            valid_snr = snr_peaks > cutoff['snr']
            performed_measures.append('snr')

            test_values = np.concatenate(
                (test_values, np.reshape(snr_peaks, (n_peaks, 1))), axis=1)
            performed_tests.append('snr')
            tests_outcomes = np.concatenate(
                (tests_outcomes, np.reshape(valid_snr, (n_peaks, 1))), axis=1)

        if 'aub' not in skip_tests:
            if self.baseline is None:
                raise ValueError('Baseline not determined, but required for '
                                 + ' area under the baseline (AUB) evaluaton.')
            if peak_set.start_idxs is None:
                raise ValueError('start_idxs not determined, but required for '
                                 + ' area under the baseline (AUB) evaluaton.')
            if peak_set.end_idxs is None:
                raise ValueError('end_idxs not determined, but required for '
                                 + ' area under the baseline (AUB) evaluaton.')
            valid_timeproducts, percentages_aub = qa.percentage_under_baseline(
                signal=peak_set.signal,
                fs=self.fs,
                peaks_s=peak_set.peak_idxs,
                starts_s=peak_set.start_idxs,
                ends_s=peak_set.end_idxs,
                baseline=self.y_baseline,
                aub_window_s=None,
                ref_signal=None,
                aub_threshold=cutoff['aub'],
            )

            performed_measures.append('aub')
            test_values = np.concatenate(
                (test_values,
                 np.reshape(percentages_aub, (n_peaks, 1))), axis=1)
            performed_tests.append('aub')
            tests_outcomes = np.concatenate(
                (tests_outcomes,
                 np.reshape(valid_timeproducts, (n_peaks, 1))), axis=1)

        test_values_df = pd.DataFrame(data=test_values,
                                      columns=performed_measures)
        test_outcomes_df = pd.DataFrame(data=tests_outcomes,
                                        columns=performed_tests)

        peak_set.update_test_outcomes(test_values_df)
        peak_set.evaluate_validity(test_outcomes_df)
        if verbose:
            print('Test values:')
            print(peak_set.test_values_df)
            print('Test outcomes:')
            print(peak_set.test_outcomes_df)

    def test_pocc_quality(
        self,
        peak_set_name,
        cutoff=None,
        skip_tests=None,
        verbose=True
    ):
        """
        Test EMG PeakSet according to quality criteria in Warnaar et al. (2024)
        Peak validity is updated in the PeakSet object.
        :param peak_set_name: PeakSet name in self.peaks dict
        :type peak_set_name: str
        :param cutoff: Cut-off criteria for passing the tests, including
        consecutiveness of Pocc manoeuvre (consecutive_poccs), and Paw upslope
        (dP_up_10, dP_up_90, and dP_up_90_norm). 'tolerant' and 'strict' can
        also be provided instead of a dict to use the respective values from
        Warnaar et al.
        :type cutoff: dict
        :param skip_tests: List of tests to skip.
        :type skip_tests: list(str)
        :param verbose: Output the test values, and pass/fail to console.
        :type verbose: bool
        """
        if peak_set_name in self.peaks.keys():
            peak_set = self.peaks[peak_set_name]
        else:
            raise KeyError("Non-existent PeaksSet key")

        if skip_tests is None:
            skip_tests = []

        if (cutoff is None
                or (isinstance(cutoff, str) and cutoff == 'tolerant')
                or (isinstance(cutoff, str) and cutoff == 'strict')):
            cutoff = dict()
            cutoff['consecutive_poccs'] = 0
            cutoff['dP_up_10'] = 0.0
            cutoff['dP_up_90'] = 2.0
            cutoff['dP_up_90_norm'] = 0.8
        elif isinstance(cutoff, dict):
            tests = ['consecutive_poccs', 'pocc_upslope']
            tests_crit = dict()
            tests_crit['consecutive_poccs'] = ['consecutive_poccs']
            tests_crit['pocc_upslope'] = [
                'dP_up_10', 'dP_up_90', 'dP_up_90_norm']

            for _, test in enumerate(tests):
                if test not in skip_tests:
                    for test_crit in tests_crit[test]:
                        if test_crit not in cutoff.keys():
                            raise KeyError(
                                'No cut-off value provided for: ' + test)

        n_peaks = len(peak_set.peak_idxs)
        performed_measures = ['peak_idx']
        test_values = np.reshape(np.array([peak_set.peak_idxs]),
                                 (n_peaks, 1))
        performed_tests = ['peak_idx']
        tests_outcomes = np.reshape(np.array([peak_set.peak_idxs]),
                                    (n_peaks, 1))
        if 'consecutive_poccs' not in skip_tests:
            if 'ventilator_breaths' not in self.peaks.keys():
                raise ValueError('Ventilator breaths not determined, but '
                                 + 'required for consecutive Pocc evaluation.')

            ventilator_breath_idxs = self.peaks['ventilator_breaths'].peak_idxs
            valid_manoeuvres = qa.detect_non_consecutive_manoeuvres(
                ventilator_breath_idxs=ventilator_breath_idxs,
                manoeuvres_idxs=peak_set.peak_idxs,
            )

            performed_tests.append('consecutive_poccs')
            tests_outcomes = np.concatenate(
                (tests_outcomes,
                 np.reshape(valid_manoeuvres, (n_peaks, 1))), axis=1)

        if 'pocc_upslope' not in skip_tests:
            if peak_set.end_idxs is None:
                raise ValueError('Pocc end_idxs not determined, but required '
                                 + 'for Pocc upslope evaluaton.')
            if peak_set.time_products is None:
                raise ValueError('PTPs not determined, but required for Pocc '
                                 + 'upslope evaluaton.')
            valid_poccs, criteria_matrix = qa.pocc_quality(
                p_vent_signal=self.y_raw,
                pocc_peaks=peak_set.peak_idxs,
                pocc_ends=peak_set.end_idxs,
                ptp_occs=peak_set.time_products,
                dp_up_10_threshold=cutoff['dP_up_10'],
                dp_up_90_threshold=cutoff['dP_up_90'],
                dp_up_90_norm_threshold=cutoff['dP_up_90_norm'],
            )
            performed_measures += ['dP_up_10', 'dP_up_90', 'dP_up_90_norm']
            test_values = np.concatenate(
                (test_values,
                 np.reshape(criteria_matrix, (n_peaks, 3))), axis=1)

            performed_tests.append('pocc_upslope')
            tests_outcomes = np.concatenate(
                (tests_outcomes,
                 np.reshape(valid_poccs, (n_peaks, 1))), axis=1)

        test_values_df = pd.DataFrame(data=test_values,
                                      columns=performed_measures)
        test_outcomes_df = pd.DataFrame(data=tests_outcomes,
                                        columns=performed_tests)

        peak_set.update_test_outcomes(test_values_df)
        peak_set.evaluate_validity(test_outcomes_df)
        if verbose:
            print('Test values:')
            print(peak_set.test_values_df)
            print('Test outcomes:')
            print(peak_set.test_outcomes_df)

    def plot_full(self, axis=None, signal_type=None,
                  colors=None, baseline_bool=True):
        """
        Plot the indicated signals in the provided axes. By default the most
        advanced signal type (envelope > clean > raw) is plotted in the
        provided colours.
        :param axis: matplotlib Axis object. If none provided, a new figure is
        created.
        :type axis: matplotlib.Axis
        :type channel_idxs: list
        :param signal_type: the signal ('envelope', 'clean', 'raw') to plot
        :type signal_type: str
        :param colors: list of colors to plot the 1) signal, 2) the baseline
        :type colors: list
        :param baseline_bool: plot the baseline
        :type baseline_bool: bool

        :returns: None
        :rtype: None
        """

        if axis is None:
            _, axis = plt.subplots()

        if colors is None:
            colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:cyan',
                      'tab:green']

        y_data = self.signal_type_data(signal_type=signal_type)
        axis.grid(True)
        axis.plot(self.t_data, y_data, color=colors[0])
        axis.set_ylabel(self.label + ' (' + self.units + ')')

        if (baseline_bool is True
                and self.y_baseline is not None
                and np.any(~np.isnan(self.y_baseline), axis=0)):
            axis.plot(self.t_data, self.y_baseline, color=colors[1])

    def plot_markers(self, peak_set_name, axes, valid_only=False,
                     colors=None, markers=None):
        """
        Plot the markers for the peak set in the provided axes in the
        provided colours using the provided markers.
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param peak_set_name: PeakSet name in self.peaks dict
        :type peak_set_name: str
        :param colors: 1 color of list of up to 3 colors for the markers, peak,
        start, and end markers. If 2 colors are provided, start and end have
        the same colors
        :type colors: str or list
        :param markers: 1 markers or list of up to 3 markers for peak, start,
        and end markers. If 2 markers are provided, start and end have the same
        marker
        :type markers: str or list

        :returns: None
        :rtype: None
        """
        if peak_set_name in self.peaks.keys():
            peak_set = self.peaks[peak_set_name]
        else:
            raise KeyError("Non-existent PeaksSet key")

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        x_vals_peak = peak_set.t_data[peak_set.peak_idxs]
        y_vals_peak = peak_set.signal[peak_set.peak_idxs]
        n_peaks = len(peak_set.peak_idxs)
        if peak_set.start_idxs is not None:
            x_vals_start = peak_set.t_data[peak_set.start_idxs]
            y_vals_start = peak_set.signal[peak_set.start_idxs]
        else:
            x_vals_start = n_peaks * [None]
            y_vals_start = n_peaks * [None]
        if peak_set.end_idxs is not None:
            x_vals_end = peak_set.t_data[peak_set.end_idxs]
            y_vals_end = peak_set.signal[peak_set.end_idxs]
        else:
            x_vals_end = n_peaks * [None]
            y_vals_end = n_peaks * [None]

        if valid_only and peak_set.valid is not None:
            x_vals_peak = x_vals_peak[peak_set.valid]
            y_vals_peak = y_vals_peak[peak_set.valid]
            x_vals_start = x_vals_start[peak_set.valid]
            y_vals_start = y_vals_start[peak_set.valid]
            x_vals_end = x_vals_end[peak_set.valid]
            y_vals_end = y_vals_end[peak_set.valid]

        if colors is None:
            colors = 'tab:red'
        if isinstance(colors, str):
            peak_color = colors
            start_color = colors
            end_color = colors
        elif isinstance(colors, list) and len(colors) == 2:
            peak_color = colors[0]
            start_color = colors[1]
            end_color = colors[1]
        elif isinstance(colors, list) and len(colors) > 2:
            peak_color = colors[0]
            start_color = colors[1]
            end_color = colors[2]
        else:
            raise ValueError('Invalid color')

        if markers is None:
            markers = '*'
        if isinstance(markers, str):
            peak_marker = markers
            start_marker = markers
            end_marker = markers
        elif isinstance(markers, list) and len(markers) == 2:
            peak_marker = markers[0]
            start_marker = markers[1]
            end_marker = markers[1]
        elif isinstance(markers, list) and len(markers) > 2:
            peak_marker = markers[0]
            start_color = markers[1]
            end_marker = markers[2]
        else:
            raise ValueError('Invalid marker')

        if len(axes) > 1:
            for _, (axis, x_peak, y_peak, x_start, y_start, x_end,
                    y_end) in enumerate(zip(
                        axes, x_vals_peak, y_vals_peak, x_vals_start,
                        y_vals_start, x_vals_end, y_vals_end)):
                axis.plot(x_peak, y_peak, marker=peak_marker,
                          color=peak_color, linestyle='None')
                if x_start is not None:
                    axis.plot(x_start, y_start, marker=start_marker,
                              color=start_color, linestyle='None')
                if x_end is not None:
                    axis.plot(x_end, y_end, marker=end_marker,
                              color=end_color, linestyle='None')
        else:
            axes[0].plot(x_vals_peak, y_vals_peak, marker=peak_marker,
                         color=peak_color, linestyle='None')
            if peak_set.start_idxs is not None:
                axes[0].plot(x_vals_start, y_vals_start, marker=start_marker,
                             color=start_color, linestyle='None')
            if peak_set.end_idxs is not None:
                axes[0].plot(x_vals_end, y_vals_end, marker=end_marker,
                             color=end_color, linestyle='None')

    def plot_peaks(self, peak_set_name, axes=None, signal_type=None,
                   margin_s=None, valid_only=False, colors=None,
                   baseline_bool=True):
        """
        Plot the indicated peaks in the provided axes. By default the most
        advanced signal type (envelope > clean > raw) is plotted in the
        provided colours.
        :param peak_set_name: The name of the peak_set to be plotted.
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param signal_type: the signal ('envelope', 'clean', 'raw') to plot
        :type signal_type: str
        :param margin_s: margins in samples plotted before the peak onset and
        after the peak offset
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
        :param colors: list of colors to plot the 1) signal, 2) the baseline
        :type colors: list
        :param baseline_bool: plot the baseline
        :type baseline_bool: bool

        :returns: None
        :rtype: None
        """
        if peak_set_name in self.peaks.keys():
            peak_set = self.peaks[peak_set_name]
        else:
            raise KeyError("Non-existent PeaksSet key")

        start_idxs = peak_set.start_idxs
        end_idxs = peak_set.end_idxs

        if valid_only and peak_set.valid is not None:
            start_idxs = start_idxs[peak_set.valid]
            end_idxs = end_idxs[peak_set.valid]

        if axes is None:
            _, axes = plt.subplots(nrows=1, ncols=len(start_idxs),
                                   sharey=True)

        if colors is None:
            colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:cyan',
                      'tab:green']
        if signal_type is None:
            y_data = peak_set.signal
        else:
            y_data = self.signal_type_data(signal_type=signal_type)

        if margin_s is None:
            m_s = self.fs // 2
        else:
            m_s = margin_s

        if len(start_idxs) == 1:
            axes = np.array([axes])

        for _, (axis, x_start, x_end) in enumerate(
                zip(axes, start_idxs, end_idxs)):
            s_start = max([0, x_start - m_s])
            s_end = max([0, x_end + m_s])

            axis.grid(True)
            axis.plot(self.t_data[s_start:s_end],
                      y_data[s_start:s_end], color=colors[0])

            if (baseline_bool is True
                    and self.y_baseline is not None
                    and np.any(~np.isnan(self.y_baseline), axis=0)):
                axis.plot(self.t_data[s_start:s_end],
                          self.y_baseline[s_start:s_end], color=colors[1])

        axes[0].set_ylabel(self.label + ' (' + self.units + ')')


class TimeSeriesGroup:
    """
    Data class to store, process, and plot time series data
    """

    def __init__(self, y_raw, t_data=None, fs=None, labels=None, units=None):
        """
        Initialize the main data characteristics:
        :param y_raw: raw signal data
        :type y_raw: ~numpy.ndarray
        :param t_data: time axis data, if None, generated from fs
        :type t_data: ~numpy.ndarray
        :param fs: sampling rate, if None, calculated from t_data
        :type fs: ~int
        :param labels: list of labels, one per provided channel
        :type labels: ~list of str
        :param units: list of signal units, one per provided channel
        :type units: ~list of str
        """
        self.channels = []
        self.fs = fs
        data_shape = list(np.array(y_raw).shape)
        data_dims = len(data_shape)
        if data_dims == 1:
            self.n_samp = len(y_raw)
            self.n_channel = 1
            y_raw = np.array(y_raw).reshape((1, self.n_samp))
        elif data_dims == 2:
            self.n_samp = data_shape[np.argmax(data_shape)]
            self.n_channel = data_shape[np.argmin(data_shape)]
            if np.argmin(data_shape) == 0:
                y_raw = np.array(y_raw)
            else:
                y_raw = np.array(y_raw).T
        else:
            raise ValueError('Invalid data dimensions')

        if t_data is None and fs is None:
            t_data = np.arange(self.n_samp)
        elif t_data is not None:
            if len(np.array(t_data).shape) > 1:
                raise ValueError('Invalid time data dimensions')
            t_data = np.array(t_data)
            if fs is None:
                fs = int(1/(t_data[1:]-t_data[:-1]))
        else:
            t_data = np.array([x_i/fs for x_i in range(self.n_samp)])

        if labels is None:
            self.labels = self.n_channel * [None]
        else:
            if len(labels) != self.n_channel:
                raise ValueError('Number of labels does not match the number'
                                 + ' of data channels.')
            self.labels = labels

        if units is None:
            self.units = self.n_channel * ['N/A']
        else:
            if len(labels) != self.n_channel:
                raise ValueError
            self.units = units

        for idx in range(self.n_channel):
            new_timeseries = TimeSeries(
                y_raw=y_raw[idx, :],
                t_data=t_data,
                fs=fs,
                label=self.labels[idx],
                units=self.units[idx]
            )
            self.channels.append(new_timeseries)

    def envelope(
        self,
        env_window=None,
        env_type=None,
        signal_type='clean',
        channel_idxs=None,
    ):
        """
        Derive the moving envelope of the provided signal. See
        envelope submodule in preprocessing.
        """

        if channel_idxs is None:
            channel_idxs = np.arange(self.n_channel)
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])

        for _, channel_idx in enumerate(channel_idxs):
            self.channels[channel_idx].envelope(
                env_window=env_window,
                env_type=env_type,
                signal_type=signal_type,
            )

    def baseline(
        self,
        percentile=33,
        window_s=None,
        step_s=None,
        method='default',
        signal_type=None,
        augm_percentile=25,
        ma_window=None,
        perc_window=None,
        channel_idxs=None,
    ):
        """
        Derive the moving baseline of the provided signal. See
        baseline submodule in postprocessing.
        """
        if channel_idxs is None:
            channel_idxs = np.arange(self.n_channel)
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])

        for _, channel_idx in enumerate(channel_idxs):
            self.channels[channel_idx].baseline(
                percentile=percentile,
                window_s=window_s,
                step_s=step_s,
                method=method,
                signal_type=signal_type,
                augm_percentile=augm_percentile,
                ma_window=ma_window,
                perc_window=perc_window,
            )

    def plot_full(self, axes=None, channel_idxs=None, signal_type=None,
                  colors=None, baseline_bool=True):
        """
        Plot the indicated signals in the provided axes. By default the most
        advanced signal type (envelope > clean > raw) is plotted in the
        provided colours.
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: ~numpy.ndarray
        :param channel_idxs: list of which channels indices to plot. If none
        provided, all channels are plot.
        :type channel_idxs: list
        :param signal_type: the signal ('envelope', 'clean', 'raw') to plot
        :type signal_type: str
        :param colors: list of colors to plot the 1) signal, 2) the baseline
        :type colors: list
        :param baseline_bool: plot the baseline
        :type baseline_bool: bool

        :returns: None
        :rtype: None
        """

        if channel_idxs is None:
            channel_idxs = np.arange(self.n_channel)
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])

        if axes is None:
            _, axes = plt.subplots(
                nrows=len(channel_idxs), ncols=1, figsize=(10, 6), sharex=True)

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        if len(channel_idxs) > len(axes):
            raise ValueError('Provided axes have not enough rows for all '
                             + 'channels to plot.')
        elif len(channel_idxs) < len(axes):
            warnings.warn('More axes provided than channels to plot.')

        for idx, channel_idx in enumerate(channel_idxs):
            self.channels[channel_idx].plot_full(
                axis=axes[idx],
                signal_type=signal_type,
                colors=colors,
                baseline_bool=baseline_bool)

    def plot_peaks(self, peak_set_name, axes=None, channel_idxs=None,
                   signal_type=None, margin_s=None, valid_only=False,
                   colors=None, baseline_bool=True):
        """
        Plot the indicated peaks for all provided channels in the provided
        axes. By default the most advanced signal type (envelope > clean > raw)
        is plotted in the provided colours.
        :param peak_set_name: The name of the peak_set to be plotted.
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param channel_idxs: list of which channels indices to plot. If none
        provided, all channels are plot.
        :type channel_idxs: list
        :param signal_type: the signal ('envelope', 'clean', 'raw') to plot
        :type signal_type: str
        :param margin_s: margins in samples plotted before the peak onset and
        after the peak offset
        :type margin_s: int
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
        :param colors: list of colors to plot the 1) signal, 2) the baseline
        :type colors: list
        :param baseline_bool: plot the baseline
        :type baseline_bool: bool

        :returns: None
        :rtype: None
        """

        if channel_idxs is None:
            channel_idxs = np.arange(self.n_channel)
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])

        if len(axes.shape) == 1:
            axes = axes.reshape((1, len(axes)))

        if axes.shape[0] < len(channel_idxs):
            raise ValueError('Provided axes have not enough rows for all '
                             + 'channels to plot.')

        for idx, channel_idx in enumerate(channel_idxs):
            if peak_set_name in self.channels[channel_idx].peaks.keys():
                axes_row = axes[idx, :]
                self.channels[channel_idx].plot_peaks(
                    axes=axes_row,
                    peak_set_name=peak_set_name,
                    signal_type=signal_type,
                    margin_s=margin_s,
                    valid_only=valid_only,
                    colors=colors,
                    baseline_bool=baseline_bool
                    )
            else:
                warnings.warn('peak_set_name not occuring in channel: '
                              + self.channels[channel_idx].label
                              + ' Skipping this channel.')

    def plot_markers(self, peak_set_name, axes=None, channel_idxs=None,
                     valid_only=False, colors=None, markers=None):
        """
        Plot the indicated peak markers for all provided channels in the
        provided axes using the provided colours and markers.
        :param peak_set_name: PeakSet name in self.peaks dict
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param colors: 1 color of list of up to 3 colors for the markers, peak,
        start, and end markers. If 2 colors are provided, start and end have
        the same colors
        :type colors: str or list
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
        :param markers: 1 markers or list of up to 3 markers for peak, start,
        and end markers. If 2 markers are provided, start and end have the same
        marker
        :type markers: str or list

        :returns: None
        :rtype: None
        """

        if channel_idxs is None:
            channel_idxs = np.arange(self.n_channel)
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])

        if len(axes.shape) == 1:
            axes = axes.reshape((1, len(axes)))

        if axes.shape[0] < len(channel_idxs):
            raise ValueError('Provided axes have not enough rows for all '
                             + 'channels to plot.')

        for idx, channel_idx in enumerate(channel_idxs):
            if peak_set_name in self.channels[channel_idx].peaks.keys():
                axes_row = axes[idx, :]
                self.channels[channel_idx].plot_markers(
                    peak_set_name=peak_set_name,
                    axes=axes_row,
                    valid_only=valid_only,
                    colors=colors,
                    markers=markers
                    )
            else:
                warnings.warn('peak_set_name not occuring in channel: '
                              + self.channels[channel_idx].label
                              + '. Skipping this channel.')


class EmgDataGroup(TimeSeriesGroup):
    """
    Child-class of TimeSeriesGroup to store and handle emg data in.
    """
    def __init__(self, y_raw, t_data=None, fs=None, labels=None, units=None):
        super().__init__(
            y_raw, t_data=t_data, fs=fs, labels=labels, units=units)

        labels_lc = [label.lower() for label in labels]
        if 'ecg' in labels_lc:
            self.ecg_idx = labels_lc.index('ecg')
        else:
            self.ecg_idx = None

    def filter(
        self,
        signal_type='raw',
        hp_cf=20.0,
        lp_cf=500.0,
        channel_idxs=None,
    ):
        """
        Filter raw EMG signals to remove baseline wander and high frequency
        components.
        """
        if channel_idxs is None:
            channel_idxs = np.arange(self.n_channel)
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])

        for _, channel_idx in enumerate(channel_idxs):
            self.channels[channel_idx].filter_emg(
                signal_type=signal_type,
                hp_cf=hp_cf,
                lp_cf=lp_cf,
            )

    def gating(
        self,
        signal_type='clean',
        gate_width_samples=None,
        ecg_peak_idxs=None,
        ecg_raw=None,
        bp_filter=True,
        channel_idxs=None,
    ):
        """
        Eliminate ECG artifacts from the provided signal. See ecg_removal
        submodule in preprocessing.
        """
        if channel_idxs is None:
            channel_idxs = np.arange(self.n_channel)
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])

        if ecg_raw is None and ecg_peak_idxs is None:
            if self.ecg_idx is not None:
                ecg_raw = self.channels[self.ecg_idx].y_raw
                print('Auto-detected ECG channel from labels.')

        for _, channel_idx in enumerate(channel_idxs):
            self.channels[channel_idx].gating(
                signal_type=signal_type,
                gate_width_samples=gate_width_samples,
                ecg_peak_idxs=ecg_peak_idxs,
                ecg_raw=ecg_raw,
                bp_filter=bp_filter,
            )


class VentilatorDataGroup(TimeSeriesGroup):
    """
    Child-class of TimeSeriesGroup to store and handle ventilator data in.
    """
    def __init__(self, y_raw, t_data=None, fs=None, labels=None, units=None):
        super().__init__(
            y_raw, t_data=t_data, fs=fs, labels=labels, units=units)

        if 'Paw' in labels:
            self.p_aw_idx = labels.index('Paw')
            print('Auto-detected Paw channel from labels.')
        else:
            self.p_aw_idx = None
        if 'F' in labels:
            self.f_idx = labels.index('F')
            print('Auto-detected Flow channel from labels.')
        else:
            self.f_idx = None
        if 'Vvent' in labels:
            self.v_vent_idx = labels.index('Vvent')
            print('Auto-detected Volume channel from labels.')
        else:
            self.v_vent_idx = None

        if self.p_aw_idx is not None and self.v_vent_idx is not None:
            self.find_peep(self.p_aw_idx, self.v_vent_idx)
        else:
            self.peep = None

    def find_peep(self, pressure_idx, volume_idx):
        """
        Calculate PEEP as the median value of Paw at end-expiration
        """
        if pressure_idx is None:
            if self.p_aw_idx is not None:
                pressure_idx = self.p_aw_idx
            else:
                raise ValueError(
                    'pressure_idx and self.p_aw_idx not defined')

        if volume_idx is None:
            if self.v_vent_idx is not None:
                volume_idx = self.v_vent_idx
            else:
                raise ValueError(
                    'volume_idx and self.v_vent_idx not defined')

        v_ee_pks, _ = scipy.signal.find_peaks(-self.channels[volume_idx].y_raw)
        self.peep = np.round(np.median(
            self.channels[pressure_idx].y_raw[v_ee_pks]))

    def find_occluded_breaths(
        self,
        pressure_idx,
        peep=None,
        start_idx=0,
        end_idx=None,
        prominence_factor=0.8,
        min_width_s=None,
        distance_s=None,
    ):
        """
        Find end-expiratory occlusion manoeuvres in ventilator pressure
        timeseries data. See the documentation in the event_detection module on
        the 'find_occluded_breaths' methods for a detailed description.
        """
        if peep is None and self.peep is None:
            raise ValueError('PEEP is not defined.')
        elif peep is None:
            peep = self.peep

        peak_idxs = find_occluded_breaths(
            p_aw=self.channels[pressure_idx].y_raw,
            fs=self.fs,
            peep=peep,
            start_s=start_idx,
            end_s=end_idx,
            prominence_factor=prominence_factor,
            min_width_s=min_width_s,
            distance_s=distance_s,
        )
        peak_idxs = peak_idxs + start_idx
        self.channels[pressure_idx].set_peaks(
            signal=self.channels[pressure_idx].y_raw,
            peak_idxs=peak_idxs,
            peak_set_name='Pocc',
        )

    def find_tidal_volume_peaks(
        self,
        volume_idx=None,
        start_idx=0,
        end_idx=None,
        width_s=None,
        threshold=None,
        prominence=None,
        threshold_new=None,
        prominence_new=None,
        pressure_idx=None,
    ):
        """
        Find tidal-volume peaks in ventilator volume signal.
        """
        if volume_idx is None:
            if self.v_vent_idx is not None:
                volume_idx = self.v_vent_idx
            else:
                raise ValueError(
                    'volume_idx and v_vent_idx not defined')

        if end_idx is None:
            end_idx = len(self.channels[volume_idx].y_raw) - 1

        if width_s is None:
            width_s = self.fs // 4

        peak_idxs = evt.detect_ventilator_breath(
            V_signal=self.channels[volume_idx].y_raw,
            start_s=start_idx,
            end_s=end_idx,
            width_s=width_s,
            threshold=threshold,
            prominence=prominence,
            threshold_new=threshold_new,
            prominence_new=prominence_new
        )

        self.channels[volume_idx].set_peaks(
            signal=self.channels[volume_idx].y_raw,
            peak_idxs=peak_idxs,
            peak_set_name='ventilator_breaths',
        )

        if pressure_idx is None and self.p_aw_idx is not None:
            pressure_idx = self.p_aw_idx

        if pressure_idx is not None:
            self.channels[pressure_idx].set_peaks(
                signal=self.channels[volume_idx].y_raw,
                peak_idxs=peak_idxs,
                peak_set_name='ventilator_breaths',
            )
        else:
            warnings.warn('pressure_idx and self.p_aw_idx not defined.')
