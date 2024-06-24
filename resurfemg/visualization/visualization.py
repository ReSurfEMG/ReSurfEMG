"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to visualize the power spectrum of EMG arrays.
"""

import numpy as np
from scipy.signal import welch, periodogram
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


def show_my_power_spectrum(signal, fs_emg, t_window_s, axis_spec=1,
                           show_plot=True, signal_unit='uV'):
    """This function plots a power spectrum of the frequencies
    comtained in an EMG based on a Fourier transform.  It does not
    return the graph, rather the values but plots the graph before it
    return.  Sample should be one single row (1-dimensional array.)

    :param signal: The signal array
    :type signal: ~numpy.ndarray
    :param fs_emg: Number of samples per second
    :type fs_emg: int
    :param t_window_s: The end of window over which values will be plotted
    :type t_window_s: int
    :param axis_spec: 1 for logaritmic axis, 0 for linear axis
    :type axis_spec: int
    :param show_plot: If True, display the plot. If False, don't display  plot.
    :type show_plot: bool
    :param signal_unit: Unit of y-axis, default is uV
    :type signal_unit: str

    :return: :code:`yf, xf` tuple of fourier transformed array and
        frequencies (the values for plotting the power spectrum)
    :rtype: Tuple[float, float]
    """
    n_samples = len(signal)
    # for our emgs sample rate is usually 2048
    y_f = np.abs(fft(signal))**2
    x_f = fftfreq(n_samples, 1 / fs_emg)

    idx = [i for i, v in enumerate(x_f) if 0 <= v <= t_window_s]
    psd_label = f'PSD [{signal_unit}**2/Hz]'

    if show_plot:
        if axis_spec == 1:
            plt.semilogy(x_f[idx], y_f[idx])
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(psd_label)
            plt.title('Power Spectral Density (Logarithmic Scale)')
            plt.show()
        if axis_spec == 0:
            plt.plot(x_f[idx], y_f[idx])
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(psd_label)
            plt.title('Power Spectral Density (Linear Scale)')
            plt.show()
        else:
            raise ValueError("Invalid axis_spec value. Please use 1"
                             "for logarithmic axis or 0 for linear axis.")
    return y_f, x_f


def show_psd_welch(signal, fs_emg, t_window_s, axis_spec=1, show_plot=True,
                   signal_unit='uV'):
    """This function calculates the power spectrum density using the Welch
    method. Tis method involves dividing the signal into overlapping segments,
    copmuting a modified periodogram for each segment, and then averaging
    these periodograms.

    :param signal: the signal array
    :type signal: ~numpy.ndarray
    :param fs_emg: Number of samples per second
    :type fs_emg: int
    :param t_window_s:Length of segments in which  original signal is divided
    :type t_window_s: int
    :param axis_spec: 1 for logaritmic axis, 0 for linear axis
    :type axis_spec: int
    :param show_plot: If True, display the plot. If False, don't display  plot.
    :type show_plot: bool
    :param signal_unit: Unit of signal for labeling the PSD axis, default uV
    :type signal_unit: str
    :return: 'f, Pxx_den'
    """
    if signal.ndim != 1:
        raise ValueError("Sample array must be 1-dimensional")

    window = np.hanning(t_window_s)
    f, Pxx_den = welch(signal, fs_emg, window=window, nperseg=t_window_s)
    psd_label = f'PSD [{signal_unit}**2/Hz]'

    if show_plot:
        if axis_spec == 1:
            plt.semilogy(f, Pxx_den)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(psd_label)
            plt.title('Power Spectral Density (Logarithmic Scale)')
            plt.show()
        elif axis_spec == 0:
            plt.plot(f, Pxx_den)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(psd_label)
            plt.title('Power Spectral Density (Linear Scale)')
            plt.show()
        else:
            raise ValueError("Invalid axis_spec value. Please use 1"
                             "for logarithmic axis or 0 for linear axis.")

    return f, Pxx_den


def show_periodogram(signal, fs_emg, axis_spec=1, show_plot=True,
                     signal_unit='uV'):
    """This function calculates the periodogram.

    :param signal: the signal array
    :type signal: ~numpy.ndarray
    :param fs_emg: Number of samples per second
    :type fs_emg: int
    :param axis_spec: 1 for logaritmic axis, 0 for linear axis
    :type axis_spec: int
    :param show_plot: If True, display the plot. If False, don't display  plot.
    :type show_plot: bool
    :param signal_unit: Unit of y-axis, default is uV
    :type signl_unit: str
    :return: 'f, Pxx_den'
    """

    f, Pxx_den = periodogram(signal, fs_emg)
    psd_label = f'PSD [{signal_unit}**2/Hz]'

    if show_plot:
        if axis_spec == 1:
            plt.semilogy(f, Pxx_den)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(psd_label)
            plt.title('Periodogram (Logarithmic Scale)')
            plt.show()
        elif axis_spec == 0:
            plt.plot(f, Pxx_den)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(psd_label)
            plt.title('Periodogram (Linear Scale)')
            plt.show()
        else:
            raise ValueError("Invalid axis_spec value. Please use 1 for"
                             "logarithmic axis or 0 for linear axis.")
    return f, Pxx_den
