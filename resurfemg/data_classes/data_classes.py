import warnings

import numpy as np
import matplotlib.pyplot as plt

from resurfemg.postprocessing.baseline import (
    moving_baseline, slopesum_baseline)
from resurfemg.preprocessing.envelope import full_rolling_rms


class TimeSeriesData:
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
        self.fs = fs
        data_shape = list(np.array(y_raw).shape)
        data_dims = len(data_shape)
        if data_dims == 1:
            self.n_samp = len(y_raw)
            self.n_channel = 1
            self.y_raw = np.array(y_raw).reshape((1, self.n_samp))
        elif data_dims == 2:
            self.n_samp = data_shape[np.argmax(data_shape)]
            self.n_channel = data_shape[np.argmin(data_shape)]
            if np.argmin(data_shape) == 0:
                self.y_raw = np.array(y_raw)
            else:
                self.y_raw = np.array(y_raw).T
        else:
            raise ValueError

        none_array = np.array([self.n_channel*[None]]).T
        self.y_clean = none_array
        self.y_env = none_array
        self.y_baseline = none_array

        if t_data is None and fs is None:
            self.t_data = np.arange(self.n_samp)
        elif t_data is not None:
            if len(np.array(t_data)) > 1:
                raise ValueError
            self.t_data = np.array(t_data)
            if fs is None:
                self.fs = int(1/(t_data[1:]-t_data[:-1]))
        else:
            self.t_data = np.array([x_i/fs for x_i in range(self.n_samp)])

        if labels is None:
            self.labels = self.n_channel * ['']
        else:
            if len(labels) != self.n_channel:
                raise ValueError
            self.labels = labels

        if units is None:
            self.units = self.n_channel * ['?']
        else:
            if len(labels) != self.n_channel:
                raise ValueError
            self.units = units

    def signal_type_data(self, channel_idxs=None, signal_type=None):
        """
        Automatically select the most advanced data type eligible for a
        subprocess ('envelope' > 'clean' > 'raw')
        :param channel_idxs: list of which channels indices to plot. If none
        provided, all channels are returned.
        :type channel_idxs: list
        :param signal_type: one of 'envelope', 'clean', or 'raw'
        :type signal_type: str

        :returns: y_data
        :rtype: ~numpy.ndarray
        """
        if channel_idxs is None:
            channel_idxs = np.arange(self.n_channel)

        y_data = np.zeros((len(channel_idxs), self.n_samp))
        for it_idx, channel_idx in enumerate(channel_idxs):
            if signal_type is None:
                if not self.y_env[channel_idx, 0] is None:
                    y_data[it_idx, :] = self.y_env[channel_idx, :]
                elif not self.y_clean[channel_idx, 0] is None:
                    y_data[it_idx, :] = self.y_clean[channel_idx, :]
                else:
                    y_data[it_idx, :] = self.y_raw[channel_idx, :]
            elif signal_type == 'env':
                if self.y_env[channel_idx, 0] is None:
                    raise IndexError('No evelope defined for this signal.')
                y_data[it_idx, :] = self.y_env[channel_idx, :]
            elif signal_type == 'clean':
                if self.y_clean[channel_idx, 0] is None:
                    warnings.warn("Warning: No clean data availabe, " +
                                  "using raw data instead.")
                    y_data[it_idx, :] = self.y_raw[channel_idx, :]
                else:
                    y_data[it_idx, :] = self.y_clean[channel_idx, :]
            else:
                y_data[it_idx, :] = self.y_raw[channel_idx, :]
        return y_data

    def envelope(
        self,
        rms_window=None,
        signal_type='clean',
        channel_idxs=None,
    ):  
        """
        Derive the moving envelope of the provided signal. See
        envelope submodule in preprocessing.
        """
        if rms_window is None:
            if self.fs is None:
                raise ValueError(
                    'Evelope window and sampling rate are not defined.')
            else:
                rms_window = int(0.2 * self.fs)

        if channel_idxs is None:
            channel_idxs = np.arange(self.n_channel)

        y_data = self.signal_type_data(channel_idxs=channel_idxs,
                                       signal_type=signal_type)
        self.y_env = np.zeros((self.n_channel, self.n_samp)) * np.nan
        for it_idx, channel_idx in enumerate(channel_idxs):
            self.y_env[channel_idx, :] = full_rolling_rms(
                y_data[it_idx, :],
                rms_window)

    def baseline(
        self,
        percentile=33,
        window_s=None,
        step_s=None,
        channel_idxs=None,
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

        if channel_idxs is None:
            channel_idxs = np.arange(self.n_channel)

        if signal_type is None:
            signal_type = 'env'

        y_baseline_data = self.signal_type_data(
            channel_idxs=channel_idxs, signal_type=signal_type)
        self.y_baseline = np.zeros((self.n_channel, self.n_samp)) * np.nan
        for it_idx, channel_idx in enumerate(channel_idxs):
            if method == 'default' or method == 'moving_baseline':
                self.y_baseline[channel_idx, :] = moving_baseline(
                    y_baseline_data[it_idx, :],
                    window_s=window_s,
                    step_s=step_s,
                    set_percentile=percentile,
                )
            elif method == 'slopesum_baseline':
                if self.fs is None:
                    raise ValueError(
                        'Sampling rate is not defined.')
                self.y_baseline[channel_idx, :], _, _, _ = slopesum_baseline(
                        y_baseline_data[it_idx, :],
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

    def plot_full(self, axes=None, channel_idxs=None, signal_type=None,
                  colors=None, baseline_bool=True):
        """
        Plot the indicated signals in the provided axes. By default the most
        advanced signal type (envelope > clean > raw) is plotted in the
        provided colours.
        axes

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

        if axes is None:
            _, axes = plt.subplots(
                nrows=len(channel_idxs), ncols=1, figsize=(10, 6), sharex=True)

        if len(channel_idxs) != len(axes):
            raise ValueError

        if colors is None:
            colors = ['tab:blue', 'tab:orange', 'tab:cyan', 'tab:green']

        y_data = self.signal_type_data(channel_idxs=channel_idxs,
                                       signal_type=signal_type)
        for plot_idx, (signal_idx, axis) in enumerate(zip(channel_idxs, axes)):
            y_plot_data = y_data[plot_idx, :]

            axis.grid(True)
            axis.plot(self.t_data,
                      y_plot_data, color=colors[0])
            axis.set_ylabel(self.labels[signal_idx]
                            + ' (' + self.units[signal_idx] + ')')

            y_baseline_sub = self.y_baseline[signal_idx, :]
            if (baseline_bool is True
                    and y_baseline_sub[0] is not None
                    and np.any(~np.isnan(y_baseline_sub), axis=0)):
                axis.plot(self.t_data, y_baseline_sub, color=colors[1])
