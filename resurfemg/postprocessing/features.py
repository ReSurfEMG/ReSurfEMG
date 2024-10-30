"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to extract features from preprocessed EMG arrays.
"""

import warnings
from scipy.integrate import trapezoid
import numpy as np

from resurfemg.helper_functions.math_operations import running_smoother


def times_under_curve(
    emg_env,
    start_index,
    end_index,
):
    """
    This function is meant to calculate the length of time to peak in
    an absolute and relative sense

    :param emg_env: an single lead EMG envelope
    :type emg_env: np.array
    :param start_index: which index number the breath starts on
    :type start_index: int
    :param end_index: which index number the breath ends on
    :type end_index: int

    :returns: times; a tuple of absolute and relative times
    :rtype: tuple
    """
    breath_arc = emg_env[start_index:end_index]
    smoothed_breath = running_smoother(breath_arc)
    abs_time = smoothed_breath.argmax()
    percent_time = abs_time / len(breath_arc)
    times = ((abs_time, percent_time))
    return times


def pseudo_slope(
    emg_env,
    start_index,
    end_index,
    smoothing=True,
):
    """
    This is a function to get the shape/slope of the take-off angle of the
    EMG signal. However, the slope is returned in units/samples (in abs values)
    , not a true slope. The slope will depend on sampling rate and pre-
    processing. Therefore, only comparison across the same sample is
    recommended.

    :param emg_env: an single lead EMG envelope
    :type emg_env: np.array
    :param start_index: which index number the breath starts on
    :type start_index: int
    :param end_index: which index number the breath ends on
    :type end_index: int
    :param smoothing: smoothing which can or can not run before calculations
    :type smoothing: bool

    :returns: pseudoslope
    :rtype: float
    """
    breath_arc = emg_env[start_index:end_index]
    pos_arc = abs(breath_arc)
    if smoothing:
        smoothed_breath = running_smoother(pos_arc)
        abs_time = smoothed_breath.argmax()
    else:
        abs_time = pos_arc.argmax()
    abs_height = pos_arc[abs_time]
    pseudoslope = abs_height / abs_time
    return pseudoslope


def amplitude(
    signal,
    peak_idxs,
    baseline=None,
):
    """
    Calculate the peak height of signal and the baseline for the windows
    at the peak_idxs relative to the baseline. If no baseline is provided, the
    peak height relative to zero is determined.
    :param signal: signal to determine the peak heights in
    :type signal: ~numpy.ndarray
    :param peak_idxs: list of individual peak start indices
    :type peak_idxs: ~np.ndarray
    :param baseline: running baseline of the signal
    :type baseline: ~numpy.ndarray
    :returns amplitudes: list of peak amplitudes
    :rtype: ~np.ndarray
    """
    if baseline is None:
        baseline = np.zeros(signal.shape)
    amplitudes = np.array(signal[peak_idxs] - baseline[peak_idxs])

    return amplitudes


def time_product(
    signal,
    fs,
    start_idxs,
    end_idxs,
    baseline=None,
):
    """
    Calculate the time product between the signal and the baseline for the
    windows defined by the start_idx and end_idx sample pairs.
    :param signal: signal to calculate the time product over
    :type signal: ~numpy.ndarray
    :param fs: sampling frequency
    :type fs: ~int
    :param start_idxs: list of individual peak start indices
    :type start_idxs: ~list
    :param end_idxs: list of individual peak end indices
    :type end_idxs: ~list
    :param baseline: running Baseline of the signal
    :type baseline: ~numpy.ndarray
    :returns: time_products
    :rtype: list
    """
    if baseline is None:
        baseline = np.zeros(signal.shape)

    time_products = np.zeros(np.asarray(start_idxs).shape)
    for idx, (start_idx, end_idx) in enumerate(zip(start_idxs, end_idxs)):
        y_delta = signal[start_idx:end_idx+1]-baseline[start_idx:end_idx+1]
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
    end_idxs,
    aub_window_s,
    baseline,
    ref_signal=None,
):
    """
    Calculate the time product between the baseline and the nadir of the
    reference signal in the aub_window_s for the windows defined by the
    start_idx and end_idx sample pairs.
    :param signal: signal to calculate the time product over
    :type signal: ~numpy.ndarray
    :param fs: sampling frequency
    :type fs: ~int
    :param peak_idxs: list of individual peak indices
    :type peak_idxs: ~list
    :param start_idxs: list of individual peak start indices
    :type start_idxs: ~list
    :param end_idxs: list of individual peak end indices
    :type end_idxs: ~list
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
    y_refs = np.zeros(np.asarray(peak_idxs).shape)
    for idx, (start_idx, peak_idx, end_idx) in enumerate(
            zip(start_idxs, peak_idxs, end_idxs)):
        y_delta_curve = (signal[start_idx:end_idx+1]
                         - baseline[start_idx:end_idx+1])
        ref_start_idx = max([0, peak_idx - aub_window_s])
        ref_end_idx = min([len(signal) - 1, peak_idx + aub_window_s])
        if (not np.all(np.sign(y_delta_curve[1:]) >= 0)
                and not np.all(np.sign(y_delta_curve[1:]) <= 0)):
            warnings.warn("Warning: Curve for peak idx" + str(idx)
                          + " not entirely above or below baseline. The "
                          + "calculated integrals will cancel out.")

        if np.median(np.sign(y_delta_curve[1:]) >= 0):
            # Positively deflected signal: Baseline below peak
            y_ref = min(ref_signal[ref_start_idx:ref_end_idx])
            y_delta = baseline[start_idx:end_idx+1] - y_ref
        else:
            # Negatively deflected signal: Baseline above peak
            y_ref = max(ref_signal[ref_start_idx:ref_end_idx])
            y_delta = y_ref - baseline[start_idx:end_idx+1]

        aubs[idx] = np.abs(trapezoid(y_delta, dx=1/fs))
        y_refs[idx] = y_ref

    return aubs, y_refs


def respiratory_rate(
    breath_idxs,
    fs,
    outlier_percentile=33,
    outlier_factor=3,
):
    """ Estimate respiratory rate based from breath indices. Breath-by-breath
    respiratory rate larger than the outlier_percentile * outlier_factor are
    excluded.
    :param breath_idxs: breath indices
    :type breath_idxs: ~numpy.ndarray[int]
    :param fs: sampling frequency
    :type fs: ~int
    :param outlier_percentile: Respiratory rate outlier percentile
    :type outlier_percentile: ~float
    :param outlier_percentile: Respiratory rate outlier factor
    :type outlier_percentile: ~float
    :returns: median respiratory rate, breath-to-breath respiratory rate.
    :rtype: (~float, ~numpy.ndarray[~float]
    """
    breath_interval = np.array(breath_idxs[1:]) - np.array(breath_idxs[:-1])
    rr_b2b = 60 * fs / breath_interval
    outlier_threshold = outlier_factor * np.percentile(
        rr_b2b, outlier_percentile)
    rr_b2b[rr_b2b > outlier_threshold] = np.nan
    rr_median = float(np.nanmedian(rr_b2b))

    return rr_median, rr_b2b
