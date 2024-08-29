"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions extract the envelopes from EMG arrays.
"""

import numpy as np
from scipy import signal
from scipy.signal import savgol_filter


def full_rolling_rms(data_emg, window_length):
    """This function computes a root mean squared envelope over an
    array :code:`data_emg`.  To do this it uses number of sample values
    :code:`window_length`.

    :param data_emg: Samples from the EMG
    :type data_emg: ~numpy.ndarray
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns: The root-mean-squared EMG sample data
    :rtype: ~numpy.ndarray
    """
    padded_samples = int(np.floor(window_length/2))
    window = np.ones(window_length)/float(window_length)
    emg_rms_padded = np.sqrt(
        np.convolve(np.power(data_emg, 2), window, 'full'))
    emg_rms = emg_rms_padded[padded_samples:-padded_samples]

    return emg_rms


def hi_envelope(our_signal, dmax=24):
    """
    Takes a 1d signal array, and extracts 'high'envelope,
    then makes high envelope, based on connecting peaks
    dmax: int, size of chunks,

    :param our_signal: 1d signal array usually of emg
    :type our_signal: ~numpy.ndarray
    :param dmax: length of chunk to look for local max in
    :type dmax: int

    :returns: src_signal_gated, the gated result
    :rtype: ~numpy.ndarray
    """
    # locals max is lmax
    lmax = (np.diff(np.sign(np.diff(our_signal))) < 0).nonzero()[0] + 1
    lmax = lmax[
        [i+np.argmax(
            our_signal[lmax[i:i+dmax]]
        ) for i in range(0, len(lmax), dmax)]
    ]
    smoothed = savgol_filter(our_signal[lmax], int(0.8 * (len(lmax))), 3)
    smoothed_interped = signal.resample(smoothed, len(our_signal))

    return smoothed_interped


def naive_rolling_rms(data_emg, window_length):
    """This function computes a root mean squared envelope over an
    array :code:`data_emg`. To do this it uses number of sample values
    :code:`window_length`.

    :param data_emg: Samples from the EMG
    :type data_emg: ~numpy.ndarray
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns: The root-mean-squared EMG sample data
    :rtype: ~numpy.ndarray
    """
    x_c = np.cumsum(abs(data_emg)**2)
    emg_rms = np.sqrt((x_c[window_length:] - x_c[:-window_length])
                      / window_length)
    return emg_rms


def running_smoother(array):
    """
    This is the smoother to use in time calculations
    """
    n_samples = len(array) // 10
    new_list = np.convolve(abs(array), np.ones(n_samples), "valid") / n_samples
    zeros = np.zeros(n_samples - 1)
    smoothed_array = np.hstack((new_list, zeros))
    return smoothed_array


def full_rolling_arv(data_emg, window_length):
    """This function computes an average rectified value envelope over an
    array :code:`data_emg`.  To do this it uses number of sample values
    :code:`window_length`.

    :param data_emg: Samples from the EMG
    :type data_emg: ~numpy.ndarray
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns: The arv envelope of the EMG sample data
    :rtype: ~numpy.ndarray
    """
    padded_samples = int(np.floor(window_length/2))
    window = np.ones(window_length)/float(window_length)
    emg_arv_padded = np.convolve(np.abs(data_emg), window, 'full')
    emg_arv = emg_arv_padded[padded_samples:-padded_samples]

    return emg_arv
