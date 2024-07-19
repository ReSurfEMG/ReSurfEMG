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

from resurfemg.preprocessing import filtering as filt
from resurfemg.preprocessing import ecg_removal as ecg_rm
from resurfemg.pipelines.pipelines import ecg_removal_gating
from resurfemg.preprocessing import envelope as evl
from resurfemg.postprocessing.baseline import (
    moving_baseline, slopesum_baseline)
from resurfemg.postprocessing.event_detection import (
    onoffpeak_baseline_crossing, onoffpeak_slope_extrapolation,
    find_occluded_breaths
)


class TimeSeries:
    """
    Data class to store, process, and plot single channel time series data
    """
    class PeaksSet:
        """
        Data class to store, and process peak information.
        """
        def __init__(self, signal, t_data, peaks_s=None):
            if isinstance(signal, np.ndarray):
                self.signal = signal
            else:
                raise ValueError("Invalid signal type: 'signal_type'.")

            if isinstance(t_data, np.ndarray):
                self.t_data = t_data
            else:
                raise ValueError("Invalid t_data type: 't_data'.")

            if peaks_s is None:
                self.peaks_s = np.array([])
            elif (isinstance(peaks_s, np.ndarray)
                    and len(np.array(peaks_s).shape) == 1):
                self.peaks_s = peaks_s
            elif isinstance(peaks_s, list):
                self.peaks_s = np.array(peaks_s)
            else:
                raise ValueError("Invalid peak indices: 'peak_s'.")

            self.starts_s = None
            self.ends_s = None
            self.valid = None

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
                (self.peaks_s, self.starts_s, self.ends_s,
                 _, _, self.valid) = onoffpeak_baseline_crossing(
                    self.signal,
                    baseline,
                    self.peaks_s)

            elif method == 'slope_extrapolation':
                if fs is None:
                    raise ValueError('Sampling rate is not defined.')

                if slope_window_s is None:
                    # TODO Insert valid default slope window
                    slope_window_s = 0.2 * fs

                (self.starts_s, self.ends_s, _, _,
                 self.valid) = onoffpeak_slope_extrapolation(
                    self.signal, fs, self.peaks_s, slope_window_s)

        def sanitize(self):
            """
            Delete invalid peak entries from the lists.
            """
            invalid_idxs = np.argwhere(self.valid)

            self.peaks_s = np.delete(self.peaks_s, invalid_idxs)
            self.starts_s = np.array(self.starts_s, invalid_idxs)
            self.ends_s = np.array(self.ends_s, invalid_idxs)
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
            peaks_s=ecg_peak_idxs,
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
        peaks_s,
        signal,
        peak_set_name,
    ):
        """
        Derive the moving envelope of the provided signal. See
        envelope submodule in preprocessing.
        """
        self.peaks[peak_set_name] = self.PeaksSet(
            peaks_s=peaks_s,
            t_data=self.t_data,
            signal=signal)

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

        x_vals_peak = peak_set.t_data[peak_set.peaks_s]
        y_vals_peak = peak_set.signal[peak_set.peaks_s]
        x_vals_start = peak_set.t_data[peak_set.starts_s]
        y_vals_start = peak_set.signal[peak_set.starts_s]
        x_vals_end = peak_set.t_data[peak_set.ends_s]
        y_vals_end = peak_set.signal[peak_set.ends_s]

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

        if isinstance(axes, np.ndarray):
            for _, (axis, x_peak, y_peak, x_start, y_start, x_end,
                    y_end) in enumerate(zip(
                        axes, x_vals_peak, y_vals_peak, x_vals_start,
                        y_vals_start, x_vals_end, y_vals_end)):
                axis.plot(x_peak, y_peak, marker=peak_marker,
                          color=peak_color, linestyle='None')
                axis.plot(x_start, y_start, marker=start_marker,
                          color=start_color, linestyle='None')
                axis.plot(x_end, y_end, marker=end_marker,
                          color=end_color, linestyle='None')
        else:
            axes.plot(x_vals_peak, y_vals_peak, marker=peak_marker,
                      color=peak_color, linestyle='None')
            axes.plot(x_vals_start, y_vals_start, marker=start_marker,
                      color=start_color, linestyle='None')
            axes.plot(x_vals_end, y_vals_end, marker=end_marker,
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

        starts_s = peak_set.starts_s
        ends_s = peak_set.ends_s

        if valid_only and peak_set.valid is not None:
            starts_s = starts_s[peak_set.valid]
            ends_s = ends_s[peak_set.valid]

        if axes is None:
            _, axes = plt.subplots(nrows=1, ncols=len(starts_s),
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

        if len(starts_s) == 1:
            axes = np.array([axes])

        for _, (axis, x_start, x_end) in enumerate(
                zip(axes, starts_s, ends_s)):
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
        :type peak_set_name: str
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
        if channel_idxs is None:
            channel_idxs = np.arange(self.n_channel)
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])

        if ecg_raw is None and ecg_peak_idxs is None:
            if self.ecg_idx is not None:
                ecg_raw = self.channels[self.ecg_idx].y_raw
                print('Auto-detected ECG channel.')

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
        else:
            self.p_aw_idx = None
        if 'F' in labels:
            self.f_idx = labels.index('F')
        else:
            self.f_idx = None
        if 'Vvent' in labels:
            self.v_vent_idx = labels.index('Vvent')
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
        v_ee_pks, _ = scipy.signal.find_peaks(-self.channels[volume_idx].y_raw)
        self.peep = np.round(np.median(
            self.channels[pressure_idx].y_raw[v_ee_pks]))

    def find_occluded_breaths(
        self,
        pressure_idx,
        peep=None,
        start_s=0,
        end_s=None,
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

        peaks_s = find_occluded_breaths(
            p_aw=self.channels[pressure_idx].y_raw,
            fs=self.fs,
            peep=peep,
            start_s=start_s,
            end_s=end_s,
            prominence_factor=prominence_factor,
            min_width_s=min_width_s,
            distance_s=distance_s,
        )
        peaks_s = peaks_s + start_s
        self.channels[pressure_idx].set_peaks(
            signal=self.channels[pressure_idx].y_raw,
            peaks_s=peaks_s,
            peak_set_name='Pocc',
        )

    def find_tidal_volume_peaks(self, threshold):
        """
        Find tidal-volume peaks in ventilator volume signal.
        """
