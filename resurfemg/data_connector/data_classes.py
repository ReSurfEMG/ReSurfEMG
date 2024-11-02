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

from resurfemg.helper_functions import math_operations as mo
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

            :returns: None
            :rtype: None
            """
            valid_idxs = np.argwhere(self.peak_df['valid'])

            self.peak_df = self.peak_df.loc[valid_idxs].reset_index(drop=True)
            self.quality_outcomes_df = \
                self.quality_outcomes_df.loc[valid_idxs].reset_index(drop=True)
            self.quality_values_df = \
                self.quality_values_df.loc[valid_idxs].reset_index(drop=True)

    def __init__(self, y_raw, t_data=None, fs=None, label=None, units=None):
        """
        :param y_raw: 1-dimensional raw signal data
        :type y_raw: ~numpy.ndarray
        :param t_data: time axis data, if None, generated from fs
        :type t_data: ~numpy.ndarray
        :param fs: sampling rate, if None, calculated from t_data
        :type fs: ~int
        :param label: label of the channel
        :type label: str
        :param units: channel signal units
        :type units: str
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
        subprocess ('env' {=envelope} > 'clean' > 'raw')

        :param signal_type: one of 'env', 'clean', or 'raw'
        :type signal_type: str

        :returns: y_data
        :rtype: numpy.ndarray
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
        order=3,
    ):
        """
        Filter raw EMG signal to remove baseline wander and high frequency
        components. See preprocessing.emg_bandpass_butter submodule.

        :returns: None
        :rtype: None
        """
        y_data = self.signal_type_data(signal_type=signal_type)
        # Eliminate the baseline wander from the data using a band-pass filter
        self.y_clean = filt.emg_bandpass_butter(
            y_data,
            high_pass=hp_cf,
            low_pass=lp_cf,
            fs_emg=self.fs,
            order=order)

    def gating(
        self,
        signal_type='clean',
        gate_width_samples=None,
        ecg_peak_idxs=None,
        ecg_raw=None,
        bp_filter=True,
        fill_method=3,
    ):
        """
        Eliminate ECG artifacts from the provided signal. See
        preprocessing.ecg_removal and pipelines.ecg_removal_gating submodules.

        :returns: None
        :rtype: None
        """
        y_data = self.signal_type_data(signal_type=signal_type)
        if ecg_peak_idxs is None:
            if ecg_raw is None:
                lp_cf = min([500.0, self.fs / 2])
                ecg_raw = filt.emg_bandpass_butter(
                    self.y_raw, high_pass=1, low_pass=lp_cf, fs_emg=self.fs)

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
            method=fill_method,
        )

    def envelope(
        self,
        env_window=None,
        env_type=None,
        signal_type='clean',
    ):
        """
        Derive the moving envelope of the provided signal. See
        preprocessing.envelope submodule.

        :returns: None
        :rtype: None
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
        postprocessing.baseline submodule.

        :returns: None
        :rtype: None
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
        Store a new PeaksSet object in the self.peaks dict

        :returns: None
        :rtype: None
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
        start_idx=0,
        end_idx=None,
    ):
        """
        Find breath peaks in provided EMG envelope signal. See
        postprocessing.event_detection submodule.

        :returns: None
        :rtype: None
        """
        if self.y_env is None:
            raise ValueError('Envelope not yet defined.')

        if self.y_baseline is None:
            warnings.warn('EMG baseline not yet defined. Peak detection '
                          + 'relative to zero.')
            y_baseline = np.zeros(self.y_env.shape)
        else:
            y_baseline = self.y_baseline

        if start_idx > len(self.y_env):
            raise ValueError('Start index higher than sample length.')

        if end_idx is None:
            end_idx = len(self.y_env)

        if end_idx < start_idx:
            raise ValueError('End index smaller than start index.')

        if end_idx > len(self.y_env):
            raise ValueError('End index higher than sample length.')

        if min_peak_width_s is None:
            min_peak_width_s = self.fs // 5

        peak_idxs = detect_emg_breaths(
            self.y_env[start_idx:end_idx],
            y_baseline[start_idx:end_idx],
            threshold=threshold,
            prominence_factor=prominence_factor,
            min_peak_width_s=min_peak_width_s,
        )
        peak_idxs += start_idx
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
        Find the peaks in the PeaksSet with the peak_set_name closest in time
        to the provided peak timings in t_reference_peaks

        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param t_reference_peaks: Refernce peak timings in t_reference_peaks
        :type t_reference_peaks: ~numpy.ndarray
        :param linked_peak_set_name: Name of the new PeaksSet
        :type linked_peak_set_name: str

        :return: None
        :rtype: None
        """
        if peak_set_name in self.peaks.keys():
            peak_set = self.peaks[peak_set_name]
        else:
            raise KeyError("Non-existent PeaksSet key")

        if linked_peak_set_name is None:
            linked_peak_set_name = peak_set_name + '_linked'

        t_PeaksSet_peaks = peak_set.peak_df['peak_idx'].to_numpy() / self.fs
        link_peak_nrs = evt.find_linked_peaks(
            t_reference_peaks,
            t_PeaksSet_peaks,
        )
        self.peaks[linked_peak_set_name] = self.PeaksSet(
            peak_set.signal, peak_set.t_data, peak_idxs=None
        )
        self.peaks[linked_peak_set_name].peak_df = \
            peak_set.peak_df.loc[link_peak_nrs].reset_index(
                drop=True)
        self.peaks[linked_peak_set_name].quality_values_df = \
            peak_set.quality_values_df.loc[link_peak_nrs].reset_index(
                drop=True)
        self.peaks[linked_peak_set_name].quality_outcomes_df = \
            peak_set.quality_outcomes_df.loc[link_peak_nrs].reset_index(
                drop=True)

    def calculate_time_products(
        self,
        peak_set_name,
        include_aub=True,
        aub_window_s=None,
        aub_reference_signal=None,
        parameter_name=None,
    ):
        """
        Calculate the time product, i.e. area under the curve for a PeaksSet.
        The results are stored as
        self.peaks[peak_set_name].peak_df[parameter_name]. If no parameter_name
        is provided, parameter_name = 'time_product'

        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param include_aub: Include the area under the baseline in the
        time product
        :type include_aub: bool
        :param signal_type: one of 'env', 'clean', or 'raw'
        :type signal_type: str
        :param aub_window_s: window length in samples in which the local
        extreme is sought.
        :param aub_window_s: int
        :param aub_reference_signal: Optional reference signal to find the
        local extreme in, else the signal underlying the PeaksSet is taken.
        :type aub_reference_signal: ~numpy.ndarray
        :param parameter_name: parameter name in Dataframe
        self.peaks[peak_set_name].peak_df
        :type parameter_name: str

        :returns: None
        :rtype: None
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
            start_idxs=peak_set.peak_df['start_idx'].to_numpy(),
            end_idxs=peak_set.peak_df['end_idx'].to_numpy(),
            baseline=baseline,
        )

        if include_aub:
            if aub_window_s is None:
                aub_window_s = 5 * self.fs
            if aub_reference_signal is None:
                aub_reference_signal = peak_set.signal
            aub, y_refs = feat.area_under_baseline(
                signal=peak_set.signal,
                fs=self.fs,
                start_idxs=peak_set.peak_df['start_idx'].to_numpy(),
                peak_idxs=peak_set.peak_df['peak_idx'].to_numpy(),
                end_idxs=peak_set.peak_df['end_idx'].to_numpy(),
                aub_window_s=aub_window_s,
                baseline=baseline,
                ref_signal=aub_reference_signal,
            )
            peak_set.peak_df['AUB'] = aub
            peak_set.peak_df['aub_y_ref'] = y_refs
            time_products += aub

        if parameter_name is not None:
            peak_set.peak_df[parameter_name] = time_products
        else:
            peak_set.peak_df['time_product'] = time_products

    def test_emg_quality(
        self,
        peak_set_name,
        cutoff=None,
        skip_tests=None,
        parameter_names=None,
        verbose=True
    ):
        """
        Test EMG PeaksSet according to quality criteria in Warnaar et al.
        (2024). Peak validity is updated in the PeaksSet object.

        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param cutoff: Cut-off criteria for passing the tests, including
        ratio between ECG and EMG interpeak time (interpeak_distance), signal-
        to-noise ratio (snr), percentage area under the baseline (aub), and
        percentage miss fit with bell curve (curve_fit). 'tolerant' and
        'strict' can also be provided instead of a dict to use the respective
        values from Warnaar et al.
        :type cutoff: dict
        :param skip_tests: List of tests to skip.
        :type skip_tests: list
        :param parameter_names: Optionally refer to custom parameter names for
        default PeaksSet (ecg)
        :type parameter_names: dict
        :param verbose: Output the test values, and pass/fail to console.
        :type verbose: bool

        :returns: None
        :rtype: None
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
            cutoff['curve_fit'] = 30
            cutoff['aub_window_s'] = 5*self.fs
            cutoff['bell_window_s'] = 5*self.fs
        elif (isinstance(cutoff, str) and cutoff == 'strict'):
            cutoff = dict()
            cutoff['interpeak_distance'] = 1.1
            cutoff['snr'] = 1.75
            cutoff['aub'] = 30
            cutoff['curve_fit'] = 25
            cutoff['aub_window_s'] = 5*self.fs
            cutoff['bell_window_s'] = 5*self.fs
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

        if parameter_names is None:
            parameter_names = dict()

        for parameter in ['ecg', 'time_product']:
            if parameter not in parameter_names.keys():
                parameter_names[parameter] = parameter

        n_peaks = len(peak_set.peak_df['peak_idx'].to_numpy())
        quality_values_df = pd.DataFrame(data=peak_set.peak_df['peak_idx'],
                                         columns=['peak_idx'])
        quality_outcomes_df = pd.DataFrame(data=peak_set.peak_df['peak_idx'],
                                           columns=['peak_idx'])

        if 'interpeak_dist' not in skip_tests:
            if 'ecg' not in self.peaks.keys():
                raise ValueError('ECG peaks not determined, but required for '
                                 + ' interpeak distance evaluation.')
            ecg_peaks = self.peaks['ecg'].peak_df['peak_idx'].to_numpy()
            valid_interpeak = qa.interpeak_dist(
                ECG_peaks=ecg_peaks,
                EMG_peaks=peak_set.peak_df['peak_idx'].to_numpy(),
                threshold=cutoff['interpeak_distance'])

            valid_interpeak_vec = np.array(n_peaks * [valid_interpeak])
            quality_outcomes_df['interpeak_distance'] = valid_interpeak_vec

        if 'snr' not in skip_tests:
            if self.baseline is None:
                raise ValueError('Baseline not determined, but required for '
                                 + ' SNR evaluaton.')
            snr_peaks = qa.snr_pseudo(
                src_signal=peak_set.signal,
                peaks=peak_set.peak_df['peak_idx'].to_numpy(),
                baseline=self.y_baseline,
                fs=self.fs,
            )
            quality_values_df['snr'] = snr_peaks
            valid_snr = snr_peaks > cutoff['snr']
            quality_outcomes_df['snr'] = valid_snr

        if 'aub' not in skip_tests:
            if self.baseline is None:
                raise ValueError('Baseline not determined, but required for '
                                 + ' area under the baseline (AUB) evaluaton.')
            if 'start_idx' not in peak_set.peak_df.columns:
                raise ValueError('start_idxs not determined, but required for '
                                 + ' area under the baseline (AUB) evaluaton.')
            if 'end_idx' not in peak_set.peak_df.columns:
                raise ValueError('end_idxs not determined, but required for '
                                 + ' area under the baseline (AUB) evaluaton.')
            outputs = qa.percentage_under_baseline(
                signal=peak_set.signal,
                fs=self.fs,
                peak_idxs=peak_set.peak_df['peak_idx'].to_numpy(),
                start_idxs=peak_set.peak_df['start_idx'].to_numpy(),
                end_idxs=peak_set.peak_df['end_idx'].to_numpy(),
                baseline=self.y_baseline,
                aub_window_s=None,
                ref_signal=None,
                aub_threshold=cutoff['aub'],
            )
            (valid_timeproducts, percentages_aub, y_refs) = outputs
            quality_values_df['aub'] = percentages_aub
            quality_values_df['aub_y_refs'] = y_refs
            quality_outcomes_df['aub'] = valid_timeproducts

        if 'curve_fit' not in skip_tests:
            if self.baseline is None:
                raise ValueError('Baseline not determined, but required for '
                                 + ' area under the baseline (AUB) evaluaton.')
            if 'start_idx' not in peak_set.peak_df.columns:
                raise ValueError('start_idxs not determined, but required for '
                                 + ' curve fit evaluaton.')
            if 'end_idx' not in peak_set.peak_df.columns:
                raise ValueError('end_idxs not determined, but required for '
                                 + ' curve fit evaluaton.')
            if parameter_names['time_product'] not in peak_set.peak_df.columns:
                raise ValueError('ETPs not determined, but required for curve '
                                 + 'fit evaluaton.')
            outputs = qa.evaluate_bell_curve_error(
                peak_idxs=peak_set.peak_df['peak_idx'].to_numpy(),
                start_idxs=peak_set.peak_df['start_idx'].to_numpy(),
                end_idxs=peak_set.peak_df['end_idx'].to_numpy(),
                signal=peak_set.signal,
                fs=self.fs,
                time_products=peak_set.peak_df[
                    parameter_names['time_product']].to_numpy(),
                bell_window_s=cutoff['bell_window_s'],
                bell_threshold=cutoff['curve_fit'],
            )
            (valid_bell_shape,
             _,
             percentage_bell_error,
             y_min,
             parameters) = outputs

            peak_set.peak_df['bell_y_min'] = y_min
            for idx, parameter in enumerate(['a', 'b', 'c']):
                peak_set.peak_df['bell_' + parameter] = parameters[:, idx]

            quality_values_df['bell'] = percentage_bell_error
            quality_outcomes_df['bell'] = valid_bell_shape

        peak_set.update_test_outcomes(quality_values_df)
        peak_set.evaluate_validity(quality_outcomes_df)
        if verbose:
            print('Test values:')
            print(peak_set.quality_values_df)
            print('Test outcomes:')
            print(peak_set.quality_outcomes_df)

    def test_pocc_quality(
        self,
        peak_set_name,
        cutoff=None,
        skip_tests=None,
        parameter_names=None,
        verbose=True,
    ):
        """
        Test EMG PeaksSet according to quality criteria in Warnaar et al.
        (2024). Peak validity is updated in the PeaksSet object.

        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param cutoff: Cut-off criteria for passing the tests, including
        consecutiveness of Pocc manoeuvre (consecutive_poccs), and p_vent
        upslope (dP_up_10, dP_up_90, and dP_up_90_norm). 'tolerant' and
        'strict' can also be provided instead of a dict to use the respective
         values from Warnaar et al.
        :type cutoff: dict
        :param skip_tests: List of tests to skip.
        :type skip_tests: list
        :param parameter_names: Optionally refer to custom parameter names for
        default PeaksSet and parameter names (ventilator_breaths,
        time_product, AUB, )
        :type parameter_names: dict
        :param verbose: Output the test values, and pass/fail to console.
        :type verbose: bool

        :returns: None
        :rtype: None
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

        if parameter_names is None:
            parameter_names = dict()

        for parameter in ['ventilator_breaths', 'time_product', 'AUB']:
            if parameter not in parameter_names.keys():
                parameter_names[parameter] = parameter

        quality_values_df = pd.DataFrame(data=peak_set.peak_df['peak_idx'],
                                         columns=['peak_idx'])
        quality_outcomes_df = pd.DataFrame(data=peak_set.peak_df['peak_idx'],
                                           columns=['peak_idx'])

        if 'consecutive_poccs' not in skip_tests:
            if parameter_names['ventilator_breaths'] not in self.peaks.keys():
                raise ValueError('Ventilator breaths not determined, but '
                                 + 'required for consecutive Pocc evaluation.')
            vent_breaths = parameter_names['ventilator_breaths']
            ventilator_breath_idxs = \
                self.peaks[vent_breaths].peak_df['peak_idx'].to_numpy()
            valid_manoeuvres = qa.detect_non_consecutive_manoeuvres(
                ventilator_breath_idxs=ventilator_breath_idxs,
                manoeuvres_idxs=peak_set.peak_df['peak_idx'].to_numpy(),
            )
            quality_outcomes_df['consecutive_poccs'] = valid_manoeuvres

        if 'pocc_upslope' not in skip_tests:
            if 'end_idx' not in peak_set.peak_df.columns:
                raise ValueError('Pocc end_idx not determined, but required '
                                 + 'for Pocc upslope evaluaton.')
            if parameter_names['time_product'] not in peak_set.peak_df.columns:
                raise ValueError('PTPs not determined, but required for Pocc '
                                 + 'upslope evaluaton.')
            valid_poccs, criteria_matrix = qa.pocc_quality(
                p_vent_signal=self.y_raw,
                pocc_peaks=peak_set.peak_df['peak_idx'].to_numpy(),
                pocc_ends=peak_set.peak_df['end_idx'].to_numpy(),
                ptp_occs=peak_set.peak_df[
                    parameter_names['time_product']].to_numpy(),
                dp_up_10_threshold=cutoff['dP_up_10'],
                dp_up_90_threshold=cutoff['dP_up_90'],
                dp_up_90_norm_threshold=cutoff['dP_up_90_norm'],
            )
            quality_values_df['dP_up_10'] = criteria_matrix[0, :]
            quality_values_df['dP_up_90'] = criteria_matrix[1, :]
            quality_values_df['dP_up_90_norm'] = criteria_matrix[2, :]

            quality_outcomes_df['pocc_upslope'] = valid_poccs

        peak_set.update_test_outcomes(quality_values_df)
        peak_set.evaluate_validity(quality_outcomes_df)
        if verbose:
            print('Test values:')
            print(peak_set.quality_values_df)
            print('Test outcomes:')
            print(peak_set.quality_outcomes_df)

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
        :param signal_type: the signal ('env', 'clean', 'raw') to plot
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

        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
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

        x_vals_peak = peak_set.t_data[
            peak_set.peak_df['peak_idx'].to_numpy()]
        y_vals_peak = peak_set.signal[
            peak_set.peak_df['peak_idx'].to_numpy()]
        n_peaks = len(peak_set.peak_df['peak_idx'].to_numpy())
        if 'start_idx' in peak_set.peak_df.columns:
            x_vals_start = peak_set.t_data[
                peak_set.peak_df['start_idx'].to_numpy()]
            y_vals_start = peak_set.signal[
                peak_set.peak_df['start_idx'].to_numpy()]
        else:
            x_vals_start = n_peaks * [None]
            y_vals_start = n_peaks * [None]
        if 'end_idx' in peak_set.peak_df.columns:
            x_vals_end = peak_set.t_data[
                peak_set.peak_df['end_idx'].to_numpy()]
            y_vals_end = peak_set.signal[
                peak_set.peak_df['end_idx'].to_numpy()]
        else:
            x_vals_end = n_peaks * [None]
            y_vals_end = n_peaks * [None]

        if valid_only and 'valid' in peak_set.peak_df.columns:
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
            if 'start_idx' in peak_set.peak_df.columns:
                axes[0].plot(x_vals_start, y_vals_start, marker=start_marker,
                             color=start_color, linestyle='None')
            if 'end_idx' in peak_set.peak_df.columns:
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

        start_idxs = peak_set.peak_df['start_idx'].to_numpy()
        end_idxs = peak_set.peak_df['end_idx'].to_numpy()

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

    def plot_curve_fits(self, peak_set_name, axes, valid_only=False,
                        colors=None):
        """
        Plot the curve-fits for the peak set in the provided axes in the
        provided colours using the provided markers.

        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
        :param colors: 1 color or list of colors for the fitted curve peak
        :type colors: str or list

        :returns: None
        :rtype: None
        """
        if peak_set_name in self.peaks.keys():
            peak_set = self.peaks[peak_set_name]
        else:
            raise KeyError("Non-existent PeaksSet key")

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for parameter in ['y_min', 'a', 'b', 'c']:
            if 'bell_' + parameter not in peak_set.peak_df.columns:
                raise KeyError('bell_' + parameter + 'not included in PeaksSet'
                               + ', curve fit is not evaluated yet.')

        if valid_only and 'valid' in peak_set.peak_df.columns:
            plot_peak_df = peak_set.peak_df.loc[peak_set.peak_df]
        else:
            plot_peak_df = peak_set.peak_df

        if colors is None:
            colors = ['tab:green']

        if isinstance(colors, str):
            color = colors
        elif isinstance(colors, list) and len(colors) >= 1:
            color = colors[0]
        else:
            raise ValueError('Invalid color')

        if len(axes) > 1:
            for _, (axis, (_, row)) in enumerate(zip(
                        axes, plot_peak_df.iterrows())):
                y_bell = mo.bell_curve(
                    peak_set.t_data[row.start_idx:row.end_idx],
                    a=row.bell_a,
                    b=row.bell_b,
                    c=row.bell_c,
                )
                axis.plot(peak_set.t_data[row.start_idx:row.end_idx],
                          row.bell_y_min + y_bell, color=color)
        else:
            for _, row in plot_peak_df.iterrows():
                y_bell = mo.bell_curve(
                    peak_set.t_data[row.start_idx:row.end_idx],
                    a=row.bell_a,
                    b=row.bell_b,
                    c=row.bell_c,
                )
                axes[0].plot(peak_set.t_data[row.start_idx:row.end_idx],
                             row.bell_y_min + y_bell, color=color)

    def plot_aub(self, peak_set_name, axes, signal_type, valid_only=False,
                 colors=None):
        """
        Plot the area under the baseline (AUB) for the peak set in the provided
        axes in the provided colours using the provided markers.

        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param signal_type: the signal ('env', 'clean', 'raw') to plot
        :type signal_type: str
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
        :param colors: 1 color or list of up to 3 colors for the peak
        :type colors: str or list

        :returns: None
        :rtype: None
        """
        if peak_set_name in self.peaks.keys():
            peak_set = self.peaks[peak_set_name]
        else:
            raise KeyError("Non-existent PeaksSet key")

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for parameter in ['y_ref']:
            if 'aub_' + parameter not in peak_set.peak_df.columns:
                raise KeyError('aub_' + parameter + 'not included in PeaksSet'
                               + ', area under the baseline is not evaluated'
                               + ' yet.')

        if signal_type is None:
            y_data = peak_set.signal
        else:
            y_data = self.signal_type_data(signal_type=signal_type)

        if valid_only and 'valid' in peak_set.peak_df.columns:
            plot_peak_df = peak_set.peak_df.loc[peak_set.peak_df]
        else:
            plot_peak_df = peak_set.peak_df

        if colors is None:
            colors = ['tab:cyan']

        if isinstance(colors, str):
            color = colors
        elif isinstance(colors, list) and len(colors) >= 1:
            color = colors[0]
        else:
            raise ValueError('Invalid color')

        if len(axes) > 1:
            for _, (axis, (_, row)) in enumerate(zip(
                        axes, plot_peak_df.iterrows())):
                axis.plot(peak_set.t_data[[row.start_idx, row.end_idx]],
                          [row.aub_y_ref, row.aub_y_ref], color=color)
                axis.plot(peak_set.t_data[[row.start_idx, row.start_idx]],
                          [y_data[row.start_idx], row.aub_y_ref], color=color)
                axis.plot(peak_set.t_data[[row.end_idx, row.end_idx]],
                          [y_data[row.end_idx], row.aub_y_ref], color=color)
        else:
            for _, row in plot_peak_df.iterrows():
                axes[0].plot(peak_set.t_data[[row.start_idx, row.end_idx]],
                             [row.aub_y_ref, row.aub_y_ref], color=color)
                axes[0].plot(peak_set.t_data[[row.start_idx, row.start_idx]],
                             [y_data[row.start_idx], row.aub_y_ref],
                             color=color)
                axes[0].plot(peak_set.t_data[[row.end_idx, row.end_idx]],
                             [y_data[row.end_idx], row.aub_y_ref],
                             color=color)


class TimeSeriesGroup:
    """
    Data class to store, process, and plot time series data
    """

    def __init__(self, y_raw, t_data=None, fs=None, labels=None, units=None):
        """
        :param y_raw: raw signal data
        :type y_raw: ~numpy.ndarray
        :param t_data: time axis data, if None, generated from fs
        :type t_data: ~numpy.ndarray
        :param fs: sampling rate, if None, calculated from t_data
        :type fs: ~int
        :param labels: list of labels, one per provided channel
        :type labels: ~list
        :param units: list of signal units, one per provided channel
        :type units: ~list

        :returns: None
        :rtype: None
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
        TimeSeries.envelope.

        :returns: None
        :rtype: None
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
        TimeSeries.baseline.

        :returns: None
        :rtype: None
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
        provided colours. See TimeSeries.plot_full.

        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: ~numpy.ndarray
        :param channel_idxs: list of which channels indices to plot. If none
        provided, all channels are plot.
        :type channel_idxs: list
        :param signal_type: the signal ('env', 'clean', 'raw') to plot
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
        axes. By default the most advanced signal type (env > clean > raw)
        is plotted in the provided colours. See TimeSeries.plot_peaks

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
        See TimeSeries.plot_markers

        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param channel_idxs: list of which channels indices to plot. If none
        provided, all channels are plot.
        :type channel_idxs: list
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
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

    :returns: None
    :rtype: None
    """
    def __init__(self, y_raw, t_data=None, fs=None, labels=None, units=None):
        super().__init__(
            y_raw, t_data=t_data, fs=fs, labels=labels, units=units)

        labels_lc = [label.lower() for label in labels]
        if 'ecg' in labels_lc:
            self.ecg_idx = labels_lc.index('ecg')
            print('Auto-detected ECG channel from labels.')
        else:
            self.ecg_idx = None

    def filter(
        self,
        signal_type='raw',
        hp_cf=20.0,
        lp_cf=500.0,
        order=3,
        channel_idxs=None,
    ):
        """
        Filter raw EMG signals to remove baseline wander and high frequency
        components. See TimeSeries.filter_emg.

        :returns: None
        :rtype: None
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
                order=order,
            )

    def gating(
        self,
        signal_type='clean',
        gate_width_samples=None,
        ecg_peak_idxs=None,
        ecg_raw=None,
        bp_filter=True,
        fill_method=3,
        channel_idxs=None,
    ):
        """
        Eliminate ECG artifacts from the provided signal. See
        TimeSeries.gating.
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
                fill_method=fill_method,
            )


class VentilatorDataGroup(TimeSeriesGroup):
    """
    Child-class of TimeSeriesGroup to store and handle ventilator data in.
    """
    def __init__(self, y_raw, t_data=None, fs=None, labels=None, units=None):
        super().__init__(
            y_raw, t_data=t_data, fs=fs, labels=labels, units=units)

        if 'Paw' in labels:
            self.p_vent_idx = labels.index('Paw')
            print('Auto-detected Pvent channel from labels.')
        elif 'Pvent' in labels:
            self.p_vent_idx = labels.index('Pvent')
            print('Auto-detected Pvent channel from labels.')
        else:
            self.p_vent_idx = None
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

        if self.p_vent_idx is not None and self.v_vent_idx is not None:
            self.find_peep(self.p_vent_idx, self.v_vent_idx)
        else:
            self.peep = None

    def find_peep(self, pressure_idx, volume_idx):
        """
        Calculate PEEP as the median value of p_vent at end-expiration.

        :param pressure_idx: Channel index of the ventilator pressure data
        :type pressure_idx: int
        :param volume_idx: Channel index of the ventilator volume data
        :type volume_idx: int

        :returns: None
        :rtype: None
        """
        if pressure_idx is None:
            if self.p_vent_idx is not None:
                pressure_idx = self.p_vent_idx
            else:
                raise ValueError(
                    'pressure_idx and self.p_vent_idx not defined')

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
        timeseries data. See postprocessing.event_detection submodule.

        :param pressure_idx: Channel index of the ventilator pressure data
        :type pressure_idx: int
        For other arguments, see postprocessing.event_detection submodule.

        :returns: None
        :rtype: None
        """
        if peep is None and self.peep is None:
            raise ValueError('PEEP is not defined.')
        elif peep is None:
            peep = self.peep

        peak_idxs = find_occluded_breaths(
            p_vent=self.channels[pressure_idx].y_raw,
            fs=self.fs,
            peep=peep,
            start_idx=start_idx,
            end_idx=end_idx,
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
        Find tidal-volume peaks in ventilator volume signal. Peaks are stored
        in PeaksSet named 'ventilator_breaths' in ventilator pressure and
        volume TimeSeries.

        :param volume_idx: Channel index of the ventilator volume data
        :type volume_idx: int
        For other arguments, see postprocessing.event_detection submodule.

        :returns: None
        :rtype: None
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
            v_vent=self.channels[volume_idx].y_raw,
            start_idx=start_idx,
            end_idx=end_idx,
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

        if pressure_idx is None and self.p_vent_idx is not None:
            pressure_idx = self.p_vent_idx

        if pressure_idx is not None:
            self.channels[pressure_idx].set_peaks(
                signal=self.channels[pressure_idx].y_raw,
                peak_idxs=peak_idxs,
                peak_set_name='ventilator_breaths',
            )
        else:
            warnings.warn('pressure_idx and self.p_vent_idx not defined.')
