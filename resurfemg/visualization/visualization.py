"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to visualize with various EMG arrays
and other types of data arrays e.g. ventilator signals.
"""

import numpy as np
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
