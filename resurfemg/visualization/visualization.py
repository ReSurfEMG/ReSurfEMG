"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to visualize with various EMG arrays
and other types of data arrays e.g. ventilator signals.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


def show_my_power_spectrum(sample, sample_rate, upper_window):
    """This function plots a power spectrum of the frequencies
    comtained in an EMG based on a Fourier transform.  It does not
    return the graph, rather the values but plots the graph before it
    return.  Sample should be one single row (1-dimensional array.)

    :param sample: The sample array
    :type sample: ~numpy.ndarray
    :param sample_rate: Number of samples per second
    :type sample_rate: int
    :param upper_window: The end of window over which values will be plotted
    :type upper_window: int

    :return: :code:`yf, xf` tuple of fourier transformed array and
        frequencies (the values for plotting the power spectrum)
    :rtype: Tuple[float, float]
    """
    n_samples = len(sample)
    # for our emgs sample rate is usually 2048
    y_f = np.abs(fft(sample))**2
    x_f = fftfreq(n_samples, 1 / sample_rate)

    idx = [i for i, v in enumerate(x_f) if 0 <= v <= upper_window]

    plt.plot(x_f[idx], y_f[idx])
    plt.show()
    return y_f, x_f


def show_psd_welch(sample, sample_rate, nperseg, axis_spec):
    """This function calculates the power spectrum density using the Welch
    method.

    :param sample: the sample array
    :type sample: ~numpy.ndarray
    :param sample_rate: Number of samples per second
    :type sample_rate: int
    :param nperseg:Length of each segment for Welch's method
    :type nperseg: int
    :param axis_spec: 1 for logaritmic axis, 0 for linear axis
    :type axis_spec: int
    :return: 'f, Pxx_den'
    """
    if sample.ndim != 1:
        raise ValueError("Sample array must be 1-dimensional")

    window = np.hanning(nperseg)
    f, Pxx_den = signal.welch(sample, sample_rate,
                              window=window, nperseg=nperseg)

    if axis_spec == 1:
        plt.semilogy(f, Pxx_den)
        plt.ylim([0.5e-3, 1])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('Power Spectral Density (Logarithmic Scale)')
        plt.show()
    elif axis_spec == 0:
        plt.plot(f, Pxx_den)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('Power Spectral Density (Linear Scale)')
        plt.show()
    else:
        raise ValueError("Invalid axis_spec value. Please use 1"
                         " for logarithmic axis or 0 for linear axis.")

    return f, Pxx_den


def show_periodogram(sample, sample_rate, axis_spec):
    """This function calculates the periodogram.

    :param sample: the sample array
    :type sample: ~numpy.ndarray
    :param sample_rate: Number of samples per second
    :type sample_rate: int
    :param axis_spec: 1 for logaritmic axis, 0 for linear axis
    :type axis_spec: int
    :return: 'f, Pxx_den'
    """

    f, Pxx_den = signal.periodogram(sample, sample_rate)

    if axis_spec == 1:
        plt.semilogy(f, Pxx_den)
        plt.ylim([0.5e-3, 1])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('Periodogram (Logarithmic Scale)')
        plt.show()
    elif axis_spec == 0:
        plt.plot(f, Pxx_den)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('Periodogram (Linear Scale)')
        plt.show()
    else:
        raise ValueError("Invalid axis_spec value. Please use 1 for"
                         "logarithmic axis or 0 for linear axis.")
    return f, Pxx_den
