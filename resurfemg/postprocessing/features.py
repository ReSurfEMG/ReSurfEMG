"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to extract features from preprocessed EMG arrays.
"""

# import collections
# import math
import warnings
# import logging
# import scipy
# from scipy.signal import savgol_filter
# from scipy.stats import entropy
from scipy.integrate import trapezoid
import numpy as np

from resurfemg.preprocessing.envelope import running_smoother


def times_under_curve(
    array,
    start_index,
    end_index,
):
    """
    This function is meant to calculate the length of time to peak in
    an absolute and relative sense

    :param array: an array e.g. single lead EMG recording
    :type array: np.array
    :param start_index: which index number the breath starts on
    :type start_index: int
    :param end_index: which index number the breath ends on
    :type end_index: int

    :returns: times; a tuple of absolute and relative times
    :rtype: tuple
    """
    breath_arc = array[start_index:end_index]
    smoothed_breath = running_smoother(breath_arc)
    abs_time = smoothed_breath.argmax()
    percent_time = abs_time / len(breath_arc)
    times = ((abs_time, percent_time))
    return times


def pseudo_slope(
    array,
    start_index,
    end_index,
    smoothing=True,
):
    """
    This is a function to get the shape/slope of the take-off angle
    of the resp. surface EMG signal, however we are returning values of
    mV divided by samples (in abs values), not a true slope
    and the number will depend on sampling rate
    and pre-processing, therefore it is recommended
    only to compare across the same single sample run

    :param array: an array e.g. single lead EMG recording
    :type array: np.array
    :param start_index: which index number the breath starts on
    :type start_index: int
    :param end_index: which index number the breath ends on
    :type end_index: int
    :param smoothing: smoothing which can or can not run before calculations
    :type smoothing: bool

    :returns: pseudoslope
    :rtype: float
    """
    breath_arc = array[start_index:end_index]
    pos_arc = abs(breath_arc)
    if smoothing:
        smoothed_breath = running_smoother(pos_arc)
        abs_time = smoothed_breath.argmax()
    else:
        abs_time = pos_arc.argmax()
    abs_height = pos_arc[abs_time]
    pseudoslope = abs_height / abs_time
    return pseudoslope


def time_product(
    signal,
    fs,
    start_idxs,
    ends_s,
    baseline=None,
):
    """
    Calculate the time product between the signal and the baseline for the
    windows defined by the start_idx and end_s sample pairs.
    :param signal: signal to calculate the time product over
    :type signal: ~numpy.ndarray
    :param fs: sampling frequency
    :type fs: ~int
    :param start_idxs: list of individual peak start indices
    :type start_idxs: ~list
    :param ends_s: list of individual peak end indices
    :type ends_s: ~list
    :param baseline: running Baseline of the signal
    :type baseline: ~numpy.ndarray
    :returns: time_products
    :rtype: list
    """
    if baseline is None:
        baseline = np.zeros(signal.shape)

    time_products = np.zeros(np.asarray(start_idxs).shape)
    for idx, (start_idx, end_s) in enumerate(zip(start_idxs, ends_s)):
        y_delta = signal[start_idx:end_s+1]-baseline[start_idx:end_s+1]
        if (not np.all(np.sign(y_delta[1:]) >= 0)
                and not np.all(np.sign(y_delta[1:]) <= 0)):
            warnings.warn("Warning: Curve for peak idx" + str(idx)
                          + " not entirely above or below baseline. The "
                          + "calculated integrals will cancel out.")

        time_products[idx] = np.abs(trapezoid(y_delta, dx=1/fs))

    return time_products


def area_under_baseline(
    signal,
    fs,
    peak_idxs,
    start_idxs,
    ends_s,
    aub_window_s,
    baseline,
    ref_signal=None,
):
    """
    Calculate the time product between the baseline and the nadir of the
    reference signal in the aub_window_s for the windows defined by the
    start_idx and end_s sample pairs.
    :param signal: signal to calculate the time product over
    :type signal: ~numpy.ndarray
    :param fs: sampling frequency
    :type fs: ~int
    :param peak_idxs: list of individual peak indices
    :type peak_idxs: ~list
    :param start_idxs: list of individual peak start indices
    :type start_idxs: ~list
    :param ends_s: list of individual peak end indices
    :type ends_s: ~list
    :param aub_window_s: number of samples before and after peak_idxs to look
    for the nadir
    :type aub_window_s: ~int
    :param baseline: running baseline of the signal
    :type baseline: ~numpy.ndarray
    :param ref_signal: signal in which the nadir is searched
    :type ref_signal: ~numpy.ndarray
    :returns: aubs
    :rtype: list
    """
    if ref_signal is None:
        ref_signal = signal

    aubs = np.zeros(np.asarray(peak_idxs).shape)
    for idx, (start_idx, peak_idx, end_s) in enumerate(
            zip(start_idxs, peak_idxs, ends_s)):
        y_delta_curve = signal[start_idx:end_s+1]-baseline[start_idx:end_s+1]
        ref_start_idx = max([0, peak_idx - aub_window_s])
        ref_end_s = min([len(signal) - 1, peak_idx + aub_window_s])
        if (not np.all(np.sign(y_delta_curve[1:]) >= 0)
                and not np.all(np.sign(y_delta_curve[1:]) <= 0)):
            warnings.warn("Warning: Curve for peak idx" + str(idx)
                          + " not entirely above or below baseline. The "
                          + "calculated integrals will cancel out.")

        if np.median(np.sign(y_delta_curve[1:]) >= 0):
            # Positively deflected signal: Baseline below peak
            y_ref = min(ref_signal[ref_start_idx:ref_end_s])
            y_delta = baseline[start_idx:end_s+1] - y_ref
        elif np.median(np.sign(y_delta_curve[1:]) <= 0):
            # Negatively deflected signal: Baseline above peak
            y_ref = max(ref_signal[ref_start_idx:ref_end_s])
            y_delta = y_ref - baseline[start_idx:end_s+1]

        aubs[idx] = np.abs(trapezoid(y_delta, dx=1/fs))

    return aubs
