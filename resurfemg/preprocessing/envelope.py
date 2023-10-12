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
    :code:`window_length`. It differs from :func:`naive_rolling_rms`
    by that the     output is the same length as the input vector.

    :param data_emg: Samples from the EMG
    :type data_emg: ~numpy.ndarray
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns: The root-mean-squared EMG sample data
    :rtype: ~numpy.ndarray
    """
    x_pad = np.pad(
        data_emg,
        (0, window_length-1),
        'constant',
        constant_values=(0, 0)
    )

    x_2 = np.power(x_pad, 2)
    window = np.ones(window_length)/float(window_length)
    emg_rms = np.sqrt(np.convolve(x_2, window, 'valid'))
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


def smooth_for_baseline(
    single_filtered_array, start=None, end=None, smooth=100
):
    """
    This is an adaptive smoothing a series that overvalues closer numbers.

    :param single_filtered_array: Array.
    :type single_filtered_array: ~numpy.ndarray
    :param start: The number of samples to work from
    :type start: int
    :param end: The number of samples to work until
    :type end: int
    :param smooth: The number of samples to work over
    :type smooth: int

    :return: tuple of arrays
    :rtype: tuple
    """
    array = single_filtered_array[start:end]
    dists = np.zeros(len(array))
    wmax, wmin = 0, 0
    nwmax, nwmin = 0, 0
    tail = (smooth - 1) / smooth

    for i, elt in enumerate(array[1:]):
        if elt > 0:
            nwmax = wmax * tail + elt / smooth
        else:
            nwmin = wmin * tail + elt / smooth
        dist = nwmax - nwmin
        dists[i] = dist
        wmax, wmin = nwmax, nwmin
    return array, dists


def smooth_for_baseline_with_overlay(
    my_own_array, threshold=10, start=None, end=None, smooth=100
):
    """This is the same as smooth for baseline, but we also get an
    overlay 0 or 1 mask tagging the baseline.

    :param my_own_array: Array
    :type  my_own_array: ~numpy.ndarray
    :param threshold: Number where to cut the mask for overlay
    :type threshold: int
    :param start: The number of samples to work from
    :type start: int
    :param end: The number of samples to work until
    :type end: int
    :param smooth: The number of samples to work over
    :type smooth: int

    :return: tuple of arrays
    :rtype: tuple
    """
    array = my_own_array[start:end]
    overlay = np.zeros(len(array)).astype('int8')
    dists = np.zeros(len(array))
    wmax, wmin = 0, 0
    nwmax, nwmin = 0, 0
    count, filler = 0, False
    tail = (smooth - 1) / smooth
    switched = 0

    for i, elt in enumerate(array[1:]):
        if elt > 0:
            nwmax = wmax * tail + elt / smooth
        else:
            nwmin = wmin * tail + elt / smooth
        dist = nwmax - nwmin
        if (i > smooth) and (i - switched > smooth):
            vodist = dists[i - smooth]
            if (vodist / dist > threshold) or (dist / vodist > threshold):
                filler = not filler
                # Now we need to go back and repaing the values in the overlay
                # because the change was detected after `smooth' interval
                overlay[i - smooth:i] = filler
                count += 1
                switched = i
        overlay[i] = filler
        dists[i] = dist
        wmax, wmin = nwmax, nwmin
    return array, overlay, dists


def vect_naive_rolling_rms(x, N):
    """This function computes a root mean squared envelope over an
    array :code:`x`.  To do this it uses number of sample values
    :code:`N`. It differs from :func:`naive_rolling_rms` by the way
    the signal is put in.

    :param xc: Samples from the EMG
    :type xc: ~numpy.ndarray

    :param N: Legnth of the sample use as window for function
    :type N: int

    :return: The root-mean-squared EMG sample data
    :rtype: ~numpy.ndarray
    """
    x_c = np.cumsum(np.abs(x)**2)
    emg_rms = np.sqrt((x_c[N:] - x_c[:-N])/N)
    return emg_rms
