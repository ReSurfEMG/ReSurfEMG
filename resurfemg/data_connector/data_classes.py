"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains data classes for standardized data storage and method
automation.
"""

import warnings

import numpy as np
import scipy
import matplotlib.pyplot as plt

from resurfemg.helper_functions import math_operations as mo
from resurfemg.helper_functions import \
    data_classes_quality_assessment as data_qa
from resurfemg.preprocessing import filtering as filt
from resurfemg.preprocessing import ecg_removal as ecg_rm
from resurfemg.pipelines.processing import ecg_removal_gating
from resurfemg.preprocessing import envelope as evl
from resurfemg.postprocessing import baseline as bl
from resurfemg.postprocessing import event_detection as evt
from resurfemg.postprocessing import features as feat

from resurfemg.data_connector.peakset_class import PeaksSet


class TimeSeries:
    """
    Data class to store, process, and plot single channel time series data
    """

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
        self.param = dict()
        self.param['fs'] = fs
        y_raw = np.array(y_raw)
        data_dims = y_raw.ndim
        if data_dims == 1:
            self.param['n_samp'] = len(y_raw)
        elif data_dims == 2:
            if y_raw.shape[0] < y_raw.shape[1]:
                y_raw = y_raw.T
            self.param['n_samp'] = y_raw.shape[0]
        else:
            raise ValueError("Invalid y_raw dimensions")
        self.y_raw = y_raw.flatten()

        self.peaks = dict()
        self.y_filt = None
        self.y_clean = None
        self.y_env = None
        self.y_baseline = None

        if t_data is None and fs is None:
            self.t_data = np.arange(self.param['n_samp'])
        elif t_data is not None:
            if len(np.array(t_data).shape) > 1:
                raise ValueError
            self.t_data = np.array(t_data)
            if fs is None:
                self.param['fs'] = int(1/(t_data[1:]-t_data[:-1]))
        else:
            self.t_data = np.array(
                [x_i/fs for x_i in range(self.param['n_samp'])])

        self.label = label or ''
        self.y_units = units or '?'

    def signal_type_data(self, signal_type=None):
        """
        Automatically select the most advanced data type eligible for a
        subprocess ('env' {=envelope} > 'clean' > 'raw')
        -----------------------------------------------------------------------
        :param signal_type: one of 'env', 'clean', or 'raw'
        :type signal_type: str

        :returns y_data: data of the selected signal type
        :rtype y_data: ~numpy.ndarray
        """
        y_data = np.zeros(self.y_raw.shape)
        signal_map = {
            None: self.y_env if self.y_env is not None else (
                self.y_clean if self.y_clean is not None else (
                    self.y_filt if self.y_filt is not None else self.y_raw)),
            'env': self.y_env,
            'clean': self.y_clean if self.y_clean is not None else (
                self.y_filt if self.y_filt is not None else self.y_raw),
            'filt': self.y_filt if self.y_filt is not None else self.y_raw,
            'raw': self.y_raw
        }
        y_data = signal_map.get(signal_type, self.y_raw)
        if signal_type == 'env' and self.y_env is None:
            raise IndexError('No envelope defined for this signal.')
        if signal_type == 'clean' and self.y_clean is None:
            warnings.warn(
                "Warning: No clean data available, using raw data instead.")
        if signal_type == 'filt' and self.y_filt is None:
            warnings.warn(
                "Warning: No filtered data available, using raw data instead.")

        return y_data

    def filter_emg(self, signal_type='raw', hp_cf=20.0, lp_cf=500.0, order=3):
        """
        Filter raw EMG signal to remove baseline wander and high frequency
        components. See preprocessing.emg_bandpass_butter submodule.
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        y_data = self.signal_type_data(signal_type=signal_type)
        # Eliminate the baseline wander from the data using a band-pass filter
        self.y_filt = filt.emg_bandpass_butter(
            y_data,
            high_pass=hp_cf,
            low_pass=lp_cf,
            fs_emg=self.param['fs'],
            order=order)

    def get_ecg_peaks(self, ecg_raw=None, bp_filter=True, overwrite=False):
        """
        Detect ECG peaks in the provided signal. See preprocessing.ecg_removal
        submodule.
        -----------------------------------------------------------------------
        :param ecg_raw: ECG signal, if None, the raw signal is used
        :type ecg_raw: ~numpy.ndarray
        :bp_filter: Apply band-pass filter to the ECG signal
        :type bp_filter: bool
        :overwrite: Overwrite existing peaks
        :type overwrite: bool

        :returns: None
        :rtype: None
        """
        if 'ecg' in self.peaks and not overwrite:
            raise UserWarning('ECG peaks already detected. Use overwrite=True')
        else:
            if ecg_raw is None:
                lp_cf = min([500.0, 0.95 * self.param['fs'] / 2])
                ecg_raw = filt.emg_bandpass_butter(
                    self.y_raw,
                    high_pass=1,
                    low_pass=lp_cf,
                    fs_emg=self.param['fs'])

            ecg_peak_idxs = ecg_rm.detect_ecg_peaks(
                ecg_raw=ecg_raw,
                fs=self.param['fs'],
                bp_filter=bp_filter,
            )

            self.set_peaks(
                signal=ecg_raw,
                peak_idxs=ecg_peak_idxs,
                peak_set_name='ecg',
            )

    def gating(self, signal_type='filt', gate_width_samples=None,
               ecg_peak_idxs=None, ecg_raw=None, bp_filter=True, fill_method=3,
               overwrite=False):
        """
        Eliminate ECG artifacts from the provided signal. See
        preprocessing.ecg_removal and pipelines.ecg_removal_gating submodules.
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        y_data = self.signal_type_data(signal_type=signal_type)
        if ecg_peak_idxs is None:
            self.get_ecg_peaks(
                ecg_raw=ecg_raw, bp_filter=bp_filter, overwrite=overwrite)
            ecg_peak_idxs = self.peaks['ecg']['peak_idx']

        if gate_width_samples is None:
            gate_width_samples = self.param['fs'] // 10

        self.y_clean = ecg_removal_gating(
            y_data,
            ecg_peak_idxs,
            gate_width_samples,
            ecg_shift=10,
            method=fill_method,
        )

    def wavelet_denoising(self, signal_type='filt', ecg_peak_idxs=None,
                          ecg_raw=None, n=None, fixed_threshold=None,
                          bp_filter=True, overwrite=False):
        """
        Eliminate ECG artifacts from the provided signal. See
        preprocessing.wavelet_denoising submodules.
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        y_data = self.signal_type_data(signal_type=signal_type)
        if ecg_peak_idxs is None:
            self.get_ecg_peaks(
                ecg_raw=ecg_raw, bp_filter=bp_filter, overwrite=overwrite)
            ecg_peak_idxs = self.peaks['ecg']['peak_idx']

        if n is None:
            n = int(np.log(self.param['fs']/20) // np.log(2))

        if fixed_threshold is None:
            fixed_threshold = 4.5

        self.y_clean, *_ = ecg_rm.wavelet_denoising(
            y_data, ecg_peak_idxs, fs=self.param['fs'], hard_thresholding=True,
            n=n, fixed_threshold=fixed_threshold, wavelet_type='db2')

    def envelope(self, env_window=None, env_type=None, signal_type='clean'):
        """
        Derive the moving envelope of the provided signal. See
        preprocessing.envelope submodule.
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        if env_window is None:
            if 'fs' in self.param:
                env_window = int(0.2 * self.param['fs'])
            else:
                raise ValueError(
                    'Evelope window and sampling rate are not defined.')

        y_data = self.signal_type_data(signal_type=signal_type)
        if env_type == 'rms' or env_type is None:
            self.y_env = evl.full_rolling_rms(y_data, env_window)
        elif env_type == 'arv':
            self.y_env = evl.full_rolling_arv(y_data, env_window)
        else:
            raise ValueError('Invalid envelope type')

    def baseline(self, percentile=33, window_s=None, step_s=None,
                 method='default', signal_type=None, augm_percentile=25,
                 ma_window=None, perc_window=None):
        """
        Derive the moving baseline of the provided signal. See
        postprocessing.baseline submodule.
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        window_s = window_s or int(7.5 * self.param.get('fs', 1))
        step_s = step_s or self.param.get('fs', 5) // 5
        signal_type = signal_type or 'env'

        y_baseline_data = self.signal_type_data(signal_type=signal_type)
        if method in ('default', 'moving_baseline'):
            self.y_baseline = bl.moving_baseline(
                y_baseline_data, window_s=window_s, step_s=step_s,
                set_percentile=percentile)
        elif method == 'slopesum_baseline':
            if 'fs' not in self.param:
                raise ValueError(
                    'Sampling rate is not defined.')
            self.y_baseline, _, _, _ = bl.slopesum_baseline(
                    y_baseline_data, window_s=window_s, step_s=step_s,
                    fs=self.param['fs'], set_percentile=percentile,
                    augm_percentile=augm_percentile, ma_window=ma_window,
                    perc_window=perc_window)
        else:
            raise ValueError('Invalid method')

    def set_peaks(self, peak_idxs, signal, peak_set_name):
        """
        Store a new PeaksSet object in the self.peaks dict
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        self.peaks[peak_set_name] = PeaksSet(
            peak_idxs=peak_idxs, t_data=self.t_data, signal=signal)

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
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        if self.y_env is None:
            raise ValueError('Envelope not yet defined.')

        y_baseline = (self.y_baseline if self.y_baseline is not None
                      else np.zeros(self.y_env.shape))
        if self.y_baseline is None:
            warnings.warn("EMG baseline not yet defined. Peak detection "
                          "relative to zero.")

        if ((end_idx is not None and end_idx > len(self.y_env))
                or start_idx > len(self.y_env)):
            raise ValueError('Index out of range.')

        end_idx = end_idx or len(self.y_env)
        if end_idx < start_idx:
            raise ValueError('End index smaller than start index.')

        min_peak_width_s = min_peak_width_s or self.param['fs'] // 5

        peak_idxs = evt.detect_emg_breaths(
            self.y_env[start_idx:end_idx], y_baseline[start_idx:end_idx],
            threshold=threshold, prominence_factor=prominence_factor,
            min_peak_width_s=min_peak_width_s)
        peak_idxs += start_idx
        self.set_peaks(peak_idxs=peak_idxs, signal=self.y_env,
                       peak_set_name=peak_set_name)

    def link_peak_set(self, peak_set_name, t_reference_peaks,
                      linked_peak_set_name=None):
        """
        Find the peaks in the PeaksSet with the peak_set_name closest in time
        to the provided peak timings in t_reference_peaks
        -----------------------------------------------------------------------
        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param t_reference_peaks: Refernce peak timings in t_reference_peaks
        :type t_reference_peaks: ~numpy.ndarray
        :param linked_peak_set_name: Name of the new PeaksSet
        :type linked_peak_set_name: str

        :return: None
        :rtype: None
        """
        if peak_set_name not in self.peaks:
            raise KeyError("Non-existent PeaksSet key")

        peak_set = self.peaks[peak_set_name]
        linked_peak_set_name = \
            linked_peak_set_name or peak_set_name + '_linked'
        t_peakset_peaks = peak_set['peak_idx'] / self.param['fs']
        link_peak_nrs = evt.find_linked_peaks(
            t_reference_peaks, t_peakset_peaks)

        self.peaks[linked_peak_set_name] = PeaksSet(
            peak_set.signal, peak_set.t_data, peak_idxs=None
        )
        for attr in ['peak_df', 'quality_values_df', 'quality_outcomes_df']:
            setattr(self.peaks[linked_peak_set_name], attr,
                    getattr(peak_set, attr).loc[link_peak_nrs].reset_index(
                        drop=True))

    def calculate_time_products(
            self, peak_set_name, include_aub=True, aub_window_s=None,
            aub_reference_signal=None, parameter_name=None):
        """
        Calculate the time product, i.e. area under the curve for a PeaksSet.
        The results are stored as
        self.peaks[peak_set_name].peak_df[parameter_name]. If no parameter_name
        is provided, parameter_name = 'time_product'
        -----------------------------------------------------------------------
        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param include_aub: Include the area under the baseline in the
        time product
        :type include_aub: bool
        :param signal_type: one of 'env', 'clean', 'filt', or 'raw'
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
        peak_set = self.peaks.get(peak_set_name)
        if peak_set is None:
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
            signal=peak_set.signal, fs=self.param['fs'],
            start_idxs=peak_set['start_idx'], end_idxs=peak_set['end_idx'],
            baseline=baseline)

        if include_aub:
            aub_window_s = aub_window_s or 5 * self.param['fs']
            aub_reference_signal = (
                peak_set.signal if aub_reference_signal is None
                else aub_reference_signal)
            aub, y_refs = feat.area_under_baseline(
                signal=peak_set.signal, fs=self.param['fs'],
                start_idxs=peak_set['start_idx'],
                peak_idxs=peak_set['peak_idx'], end_idxs=peak_set['end_idx'],
                aub_window_s=aub_window_s, baseline=baseline,
                ref_signal=aub_reference_signal)
            peak_set.peak_df['AUB'] = aub
            peak_set.peak_df['aub_y_ref'] = y_refs
            time_products += aub

        peak_set.peak_df[parameter_name or 'time_product'] = time_products

    def test_emg_quality(self, peak_set_name, cutoff=None, skip_tests=None,
                         parameter_names=None, verbose=True):
        """See helper_functions.data_classes_quality_assessment submodule."""
        data_qa.test_emg_quality(
            self, peak_set_name, cutoff, skip_tests, parameter_names, verbose)

    def test_pocc_quality(self, peak_set_name, cutoff=None, skip_tests=None,
                          parameter_names=None, verbose=True):
        """See helper_functions.data_classes_quality_assessment submodule."""
        data_qa.test_pocc_quality(
            self, peak_set_name, cutoff, skip_tests, parameter_names, verbose)

    def test_linked_peak_sets(
            self, peak_set_name, linked_timeseries, linked_peak_set_name,
            parameter_names=None, cutoff=None, skip_tests=None, verbose=True):
        """See helper_functions.data_classes_quality_assessment submodule."""
        data_qa.test_linked_peak_sets(
            self, peak_set_name, linked_timeseries, linked_peak_set_name,
            parameter_names, cutoff, skip_tests, verbose)

    def plot_full(self, axis=None, signal_type=None, colors=None,
                  baseline_bool=True):
        """Plot the indicated signals in the provided axes. By default the most
        advanced signal type (envelope > clean > filt > raw) is plotted in the
        provided colours.
        -----------------------------------------------------------------------
        :param axis: matplotlib Axis object. If none provided, a new figure is
        created.
        :type axis: matplotlib.Axis
        :param signal_type: the signal ('env', 'clean', 'filt', 'raw') to plot
        :type signal_type: str
        :param colors: list of colors to plot the 1) signal, 2) the baseline
        :type colors: list
        :param baseline_bool: plot the baseline
        :type baseline_bool: bool

        :returns: None
        :rtype: None
        """
        axis = axis or plt.subplots()[1]
        colors = colors or [
            'tab:blue', 'tab:orange', 'tab:red', 'tab:cyan', 'tab:green']

        y_data = self.signal_type_data(signal_type=signal_type)
        axis.grid(True)
        axis.plot(self.t_data, y_data, color=colors[0])
        axis.set_ylabel(self.label + ' (' + self.y_units + ')')

        if (baseline_bool is True
                and self.y_baseline is not None
                and np.any(~np.isnan(self.y_baseline), axis=0)):
            axis.plot(self.t_data, self.y_baseline, color=colors[1])

    def plot_markers(self, peak_set_name, axes, valid_only=False,
                     colors=None, markers=None):
        """Plot the markers for the peak set in the provided axes in the
        provided colours using the provided markers.
        -----------------------------------------------------------------------
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
        if peak_set_name not in self.peaks:
            raise KeyError("Non-existent PeaksSet key")

        peak_set = self.peaks[peak_set_name]
        valid = (peak_set['valid'] if valid_only and
                 'valid' in peak_set.peak_df.columns
                 else np.ones(len(peak_set['peak_idx']), dtype=bool))

        def get_values(column):
            return ((peak_set.t_data[peak_set[column]][valid],
                     peak_set.signal[peak_set[column]][valid])
                    if column in peak_set.peak_df.columns
                    else (len(peak_set) * [None], len(peak_set) * [None]))

        x_vals_peak, y_vals_peak = get_values('peak_idx')
        x_vals_start, y_vals_start = get_values('start_idx')
        x_vals_end, y_vals_end = get_values('end_idx')

        def get_color_marker(values, default):
            if values is None:
                return [default] * 3
            if isinstance(values, str):
                return [values] * 3
            if isinstance(values, list):
                return values[:3] if len(values) > 2 else [
                    values[0], values[1], values[1]]
            raise ValueError(f'Invalid {default}')

        peak_color, start_color, end_color = get_color_marker(
            colors, 'tab:red')
        peak_marker, start_marker, end_marker = get_color_marker(markers, '*')

        if len(np.atleast_1d(axes)) == 1 and len(x_vals_peak) > 1:
            axes = np.matlib.repmat(axes, len(x_vals_peak), 1).flatten()
        for axis, x_peak, y_peak, x_start, y_start, x_end, y_end in zip(
                np.atleast_1d(axes), x_vals_peak, y_vals_peak, x_vals_start,
                y_vals_start, x_vals_end, y_vals_end):

            axis.plot(x_peak, y_peak, marker=peak_marker, color=peak_color,
                      linestyle='None')
            if x_start is not None:
                axis.plot(x_start, y_start, marker=start_marker,
                          color=start_color, linestyle='None')
            if x_end is not None:
                axis.plot(x_end, y_end, marker=end_marker, color=end_color,
                          linestyle='None')

    def plot_peaks(self, peak_set_name, axes=None, signal_type=None,
                   margin_s=None, valid_only=False, colors=None,
                   baseline_bool=True):
        """Plot the indicated peaks in the provided axes. By default the most
        advanced signal type (envelope > clean > filt > raw) is plotted in the
        provided colours.
        -----------------------------------------------------------------------
        :param peak_set_name: The name of the peak_set to be plotted.
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param signal_type: the signal ('env', 'clean', 'filt' 'raw') to plot
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
        peak_set = self.peaks.get(peak_set_name)
        if peak_set is None:
            raise KeyError("Non-existent PeaksSet key")

        start_idxs, end_idxs = peak_set['start_idx'], peak_set['end_idx']
        if valid_only and peak_set.valid is not None:
            valid = peak_set.valid
            start_idxs, end_idxs = start_idxs[valid], end_idxs[valid]

        if axes is None:
            _, axes = plt.subplots(nrows=1, ncols=len(start_idxs), sharey=True)

        colors = colors or [
            'tab:blue', 'tab:orange', 'tab:red', 'tab:cyan', 'tab:green']
        y_data = (peak_set.signal if signal_type is None
                  else self.signal_type_data(signal_type=signal_type))
        m_s = margin_s or self.param['fs'] // 2

        for axis, x_start, x_end in zip(
                np.atleast_1d(axes), start_idxs, end_idxs):
            s_start, s_end = max(0, x_start - m_s), max(0, x_end + m_s)
            axis.grid(True)
            axis.plot(self.t_data[s_start:s_end], y_data[s_start:s_end],
                      color=colors[0])
            if baseline_bool and self.y_baseline is not None and np.any(
                    ~np.isnan(self.y_baseline), axis=0):
                axis.plot(self.t_data[s_start:s_end],
                          self.y_baseline[s_start:s_end], color=colors[1])

        axes[0].set_ylabel(f"{self.label} ({self.y_units})")

    def plot_curve_fits(self, peak_set_name, axes, valid_only=False,
                        colors=None):
        """Plot the curve-fits for the peak set in the provided axes in the
        provided colours using the provided markers.
        -----------------------------------------------------------------------
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
        peak_set = self.peaks.get(peak_set_name)
        if peak_set is None:
            raise KeyError("Non-existent PeaksSet key")

        axes = np.atleast_1d(axes)
        required_params = ['y_min', 'a', 'b', 'c']
        missing_params = [
            param for param in required_params
            if f'bell_{param}' not in peak_set.peak_df.columns]
        if missing_params:
            raise KeyError(
                f"Missing parameters in PeaksSet: {', '.join(missing_params)}")

        plot_peak_df = (peak_set.peak_df.loc[peak_set.peak_df['valid']]
                        if valid_only and 'valid' in peak_set.peak_df.columns
                        else peak_set.peak_df)
        color = \
            colors[0] if isinstance(colors, list) and colors else 'tab:green'

        for axis, (_, row) in zip(axes, plot_peak_df.iterrows()):
            y_bell = mo.bell_curve(
                peak_set.t_data[row.start_idx:row.end_idx], a=row.bell_a,
                b=row.bell_b, c=row.bell_c)
            axis.plot(peak_set.t_data[row.start_idx:row.end_idx],
                      row.bell_y_min + y_bell, color=color)

        if len(axes) > 1:
            for _, (axis, (_, row)) in enumerate(zip(
                        axes, plot_peak_df.iterrows())):
                y_bell = mo.bell_curve(
                    peak_set.t_data[row.start_idx:row.end_idx],
                    a=row.bell_a, b=row.bell_b, c=row.bell_c)
                axis.plot(peak_set.t_data[row.start_idx:row.end_idx],
                          row.bell_y_min + y_bell, color=color)
        else:
            for _, row in plot_peak_df.iterrows():
                y_bell = mo.bell_curve(
                    peak_set.t_data[row.start_idx:row.end_idx],
                    a=row.bell_a, b=row.bell_b, c=row.bell_c)
                axes[0].plot(peak_set.t_data[row.start_idx:row.end_idx],
                             row.bell_y_min + y_bell, color=color)

    def plot_aub(self, peak_set_name, axes, signal_type, valid_only=False,
                 colors=None):
        """Plot the area under the baseline (AUB) for the peak set in the
        provided axes in the provided colours using the provided markers.
        -----------------------------------------------------------------------
        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param signal_type: the signal ('env', 'clean', 'filt', 'raw') to plot
        :type signal_type: str
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
        :param colors: 1 color or list of up to 3 colors for the peak
        :type colors: str or list

        :returns: None
        :rtype: None
        """
        peak_set = self.peaks.get(peak_set_name)
        if peak_set is None:
            raise KeyError("Non-existent PeaksSet key")

        axes = np.atleast_1d(axes)

        if 'aub_y_ref' not in peak_set.peak_df.columns:
            raise KeyError("aub_y_ref not included in PeaksSet, area under the"
                           " baseline is not evaluated yet.")

        y_data = (peak_set.signal if signal_type is None
                  else self.signal_type_data(signal_type=signal_type))

        plot_peak_df = (peak_set.peak_df.loc[peak_set.peak_df['valid']]
                        if valid_only and 'valid' in peak_set.peak_df.columns
                        else peak_set.peak_df)

        colors = colors[0] if isinstance(colors, list) else colors

        color = (colors if isinstance(colors, str)
                 else colors[0] if isinstance(colors, list) and colors
                 else 'tab:cyan')

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
    Data class to store, process, and plot time series data. Channels can be
    accessed by index or by label."""

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
        self.param = dict()
        self.param['fs'] = fs
        self._available_methods = [
            'envelope',
            'baseline',
        ]
        data_shape = list(np.array(y_raw).shape)
        data_dims = len(data_shape)
        if data_dims == 1:
            self.param['n_samp'] = len(y_raw)
            self.param['n_channel'] = 1
            y_raw = np.array(y_raw).reshape((1, self.param['n_samp']))
        elif data_dims == 2:
            self.param['n_samp'] = data_shape[np.argmax(data_shape)]
            self.param['n_channel'] = data_shape[np.argmin(data_shape)]
            if np.argmin(data_shape) == 0:
                y_raw = np.array(y_raw)
            else:
                y_raw = np.array(y_raw).T
        else:
            raise ValueError('Invalid data dimensions')

        if t_data is None and fs is None:
            t_data = np.arange(self.param['n_samp'])
        elif t_data is not None:
            if len(np.array(t_data).shape) > 1:
                raise ValueError('Invalid time data dimensions')
            t_data = np.array(t_data)
            if fs is None:
                fs = int(1/(t_data[1:]-t_data[:-1]))
        else:
            t_data = np.array(
                [x_i/fs for x_i in range(self.param['n_samp'])])

        if labels is None:
            self.labels = self.param['n_channel'] * [None]
        else:
            if len(labels) != self.param['n_channel']:
                raise ValueError('Number of labels does not match the number'
                                 + ' of data channels.')
            self.labels = labels

        if units is None:
            self.y_units = self.param['n_channel'] * ['N/A']
        else:
            if len(labels) != self.param['n_channel']:
                raise ValueError
            self.y_units = units

        for idx in range(self.param['n_channel']):
            new_timeseries = TimeSeries(
                y_raw=y_raw[idx, :],
                t_data=t_data,
                fs=fs,
                label=self.labels[idx],
                units=self.y_units[idx]
            )
            self.channels.append(new_timeseries)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.channels[key]
        elif isinstance(key, str):
            for channel in self.channels:
                if channel.label == key:
                    return channel
            raise KeyError('Channel not found')
        else:
            raise ValueError('Invalid key type')

    def __iter__(self):
        return iter(self.channels)

    def run(self, method, channel_idxs=None, **kwargs):
        """Run a TimeSeries function on the provided channels in the group. The
        function is run with the provided keyword arguments."""
        if channel_idxs is None:
            channel_idxs = np.arange(self.param['n_channel'])
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])

        if method in self._available_methods:
            if method in ['gating', 'wavelet_denoising']:
                if 'ecg_peak_idxs' in kwargs:
                    print('Provided ECG peak indices used for ECG removal.')
                elif 'ecg_raw' in kwargs:
                    print('Provided raw ECG used for ECG removal.')
                else:
                    if self.ecg_idx is not None:
                        kwargs['ecg_raw'] = self[self.ecg_idx].y_raw
                        print('Set ECG channel used for ECG removal.')
                    else:
                        raise ValueError("No ECG channel and peak indices "
                                         "provided")

            for _, channel_idx in enumerate(channel_idxs):
                getattr(self.channels[channel_idx], method)(
                    **kwargs
                )
        else:
            raise ValueError('Invalid method')

    def plot_full(self, axes=None, channel_idxs=None, signal_type=None,
                  colors=None, baseline_bool=True):
        """
        Plot the indicated signals in the provided axes. By default the most
        advanced signal type (envelope > clean > filt > raw) is plotted in the
        provided colours. See TimeSeries.plot_full for colors, baseline_bool.
        -----------------------------------------------------------------------
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: ~numpy.ndarray
        :param channel_idxs: list of which channels indices to plot. If none
        provided, all channels are plot.
        :type channel_idxs: list
        :param signal_type: the signal ('env', 'clean', 'filt', 'raw') to plot
        :type signal_type: str

        :returns: None
        :rtype: None
        """
        channel_idxs = (np.arange(self.param['n_channel'])
                        if channel_idxs is None
                        else np.atleast_1d(channel_idxs))
        if axes is None:
            _, axes = plt.subplots(
                nrows=len(channel_idxs), ncols=1, figsize=(10, 6), sharex=True)
        axes = np.atleast_1d(axes)

        if len(channel_idxs) > len(axes):
            raise ValueError(
                'Provided axes have not enough rows for all channels to plot.')
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
        axes. By default the most advanced signal type (env > clean > filt >
        raw) is plotted in the provided colours. See TimeSeries.plot_peaks for
        signal_type, margin_s, valid_only, colors, baseline_bool.
        -----------------------------------------------------------------------
        :param peak_set_name: The name of the peak_set to be plotted.
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param channel_idxs: list of which channels indices to plot. If none
        provided, all channels are plot.
        :type channel_idxs: list

        :returns: None
        :rtype: None
        """
        channel_idxs = (np.arange(self.param['n_channel'])
                        if channel_idxs is None
                        else np.atleast_1d(channel_idxs))
        if axes is None:
            _, axes = plt.subplots(
                nrows=len(channel_idxs), ncols=1, figsize=(10, 6), sharex=True)
        axes = np.atleast_2d(axes)

        if axes.shape[0] < len(channel_idxs):
            raise ValueError(
                'Provided axes have not enough rows for all channels to plot.')

        for idx, channel_idx in enumerate(channel_idxs):
            if peak_set_name in self.channels[channel_idx].peaks:
                self.channels[channel_idx].plot_peaks(
                    axes=axes[idx, :],
                    peak_set_name=peak_set_name,
                    signal_type=signal_type,
                    margin_s=margin_s,
                    valid_only=valid_only,
                    colors=colors,
                    baseline_bool=baseline_bool)
            else:
                warnings.warn(f"""peak_set_name not occurring in channel:
                              {self.channels[channel_idx].label}. Skipping this
                                channel.""")

    def plot_markers(self, peak_set_name, axes=None, channel_idxs=None,
                     valid_only=False, colors=None, markers=None):
        """
        Plot the indicated peak markers for all provided channels in the
        provided axes using the provided colours and markers.
        See TimeSeries.plot_markers for valid_only, colors, markers.
        -----------------------------------------------------------------------
        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param channel_idxs: list of which channels indices to plot. If none
        provided, all channels are plot.
        :type channel_idxs: list

        :returns: None
        :rtype: None
        """
        channel_idxs = (np.arange(self.param['n_channel'])
                        if channel_idxs is None
                        else np.atleast_1d(channel_idxs))
        if axes is None:
            _, axes = plt.subplots(
                nrows=len(channel_idxs), ncols=1, figsize=(10, 6), sharex=True)
        axes = np.atleast_2d(axes)

        if axes.shape[0] < len(channel_idxs):
            raise ValueError(
                'Provided axes have not enough rows for all channels to plot.')

        for idx, channel_idx in enumerate(channel_idxs):
            if peak_set_name in self.channels[channel_idx].peaks:
                self.channels[channel_idx].plot_markers(
                    peak_set_name=peak_set_name,
                    axes=axes[idx, :],
                    valid_only=valid_only,
                    colors=colors,
                    markers=markers)
            else:
                warnings.warn("peak_set_name not occurring in channel: "
                              f"{self.channels[channel_idx].label}. "
                              "Skipping this channel.")


class EmgDataGroup(TimeSeriesGroup):
    """Child-class of TimeSeriesGroup to store and handle emg data in with the
    additional methods filter_emg, gating, and wavelet_denoising."""
    def __init__(self, y_raw, t_data=None, fs=None, labels=None, units=None):
        super().__init__(
            y_raw, t_data=t_data, fs=fs, labels=labels, units=units)
        self._available_methods.append('filter_emg')
        self._available_methods.append('gating')
        self._available_methods.append('wavelet_denoising')

        labels_lc = [label.lower() for label in labels]
        if 'ecg' in labels_lc:
            self.ecg_idx = labels_lc.index('ecg')
            print('Auto-detected ECG channel from labels.')
        else:
            print("No ECG channel detected. Set ECG channel index with "
                  "`EmgDataGroup.set_ecg_idx(arg)` method.")
            self.ecg_idx = None

    def set_ecg_idx(self, ecg_idx):
        """
        Set the ECG channel index in the group.
        -----------------------------------------------------------------------
        :param ecg_idx: ECG channel index or label
        :type ecg_idx: int

        :returns: None
        :rtype: None
        """
        if isinstance(ecg_idx, int):
            self.ecg_idx = ecg_idx
        elif isinstance(ecg_idx, str):
            self.ecg_idx = self.labels.index(ecg_idx)


class VentilatorDataGroup(TimeSeriesGroup):
    """
    Child-class of TimeSeriesGroup to store and handle ventilator data in.
    Default channels are 'Paw'/ 'Pvent', 'F', and 'Vvent'.
    """
    def __init__(self, y_raw, t_data=None, fs=None, labels=None, units=None):
        super().__init__(
            y_raw, t_data=t_data, fs=fs, labels=labels, units=units)

        self.p_vent_idx = next((labels.index(label)
                                for label in ['Paw', 'Pvent']
                                if label in labels), None)
        self.f_idx = labels.index('F') if 'F' in labels else None
        self.v_vent_idx = labels.index('Vvent') if 'Vvent' in labels else None

        if self.p_vent_idx is not None:
            print('Auto-detected Pvent channel from labels.')
        if self.f_idx is not None:
            print('Auto-detected Flow channel from labels.')
        if self.v_vent_idx is not None:
            print('Auto-detected Volume channel from labels.')

        if self.p_vent_idx is not None and self.v_vent_idx is not None:
            self.find_peep(self.p_vent_idx, self.v_vent_idx)
        else:
            self.peep = None

    def find_peep(self, pressure_idx, volume_idx):
        """
        Calculate PEEP as the median value of p_vent at end-expiration.
        -----------------------------------------------------------------------
        :param pressure_idx: Channel index of the ventilator pressure data
        :type pressure_idx: int
        :param volume_idx: Channel index of the ventilator volume data
        :type volume_idx: int

        :returns: None
        :rtype: None
        """
        pressure_idx = pressure_idx or self.p_vent_idx
        if pressure_idx is None:
            raise ValueError('pressure_idx and self.p_vent_idx not defined')

        volume_idx = volume_idx or self.v_vent_idx
        if volume_idx is None:
            raise ValueError('volume_idx and self.v_vent_idx not defined')

        v_ee_pks, _ = scipy.signal.find_peaks(-self.channels[volume_idx].y_raw)
        self.peep = np.round(np.median(
            self.channels[pressure_idx].y_raw[v_ee_pks]))

    def find_occluded_breaths(self, pressure_idx=None, peep=None, **kwargs):
        """
        Find end-expiratory occlusion manoeuvres in ventilator pressure
        timeseries data. See postprocessing.event_detection submodule.
        -----------------------------------------------------------------------
        :param pressure_idx: Channel index of the ventilator pressure data
        :type pressure_idx: int
        For other arguments, see postprocessing.event_detection submodule.

        :returns: None
        :rtype: None
        """
        pressure_idx = pressure_idx or self.p_vent_idx
        kwargs['p_vent'] = self.channels[pressure_idx].y_raw
        kwargs['fs'] = self.param['fs']

        kwargs['peep'] = peep or self.peep
        if kwargs['peep'] is None:
            raise ValueError('PEEP is not defined.')

        peak_idxs = evt.find_occluded_breaths(**kwargs)
        peak_idxs = peak_idxs + kwargs['start_idx']
        self.channels[pressure_idx].set_peaks(
            signal=self.channels[pressure_idx].y_raw, peak_idxs=peak_idxs,
            peak_set_name='Pocc')

    def find_tidal_volume_peaks(
        self, volume_idx=None, pressure_idx=None, **kwargs,
    ):
        """
        Find tidal-volume peaks in ventilator volume signal. Peaks are stored
        in PeaksSet named 'ventilator_breaths' in ventilator pressure and
        volume TimeSeries.
        -----------------------------------------------------------------------
        :param volume_idx: Channel index of the ventilator volume data
        :type volume_idx: int
        :param pressure_idx: Channel index of the ventilator pressure data
        :type pressure_idx: int
        For other arguments, see postprocessing.event_detection submodule.

        :returns: None
        :rtype: None
        """
        volume_idx = (kwargs.pop('volume_idx') if 'volume_idx' in kwargs
                      else self.v_vent_idx)
        if volume_idx is None:
            raise ValueError('volume_idx and v_vent_idx not defined')
        kwargs['v_vent'] = self.channels[volume_idx].y_raw

        kwargs.setdefault('start_idx', 0)
        kwargs.setdefault('end_idx', len(self.channels[volume_idx].y_raw) - 1)
        kwargs.setdefault('width_s', self.param['fs'] // 4)

        peak_idxs = evt.detect_ventilator_breath(**kwargs)

        self.channels[volume_idx].set_peaks(
            signal=self.channels[volume_idx].y_raw, peak_idxs=peak_idxs,
            peak_set_name='ventilator_breaths')

        pressure_idx = pressure_idx or self.p_vent_idx
        if pressure_idx is not None:
            self.channels[pressure_idx].set_peaks(
                signal=self.channels[pressure_idx].y_raw, peak_idxs=peak_idxs,
                peak_set_name='ventilator_breaths')
        else:
            warnings.warn('pressure_idx and self.p_vent_idx not defined.')
