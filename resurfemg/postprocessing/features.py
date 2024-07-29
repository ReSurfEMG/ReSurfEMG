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


# def simple_area_under_curve(
#     array,
#     start_index,
#     end_index,
# ):
#     """
#     This function is just a wrapper over np.sum written because it isn't
#     apparent to some clinically oriented people that an area under the curve
#     will be a sum of all the numbers

#     :param array: an array e.g. single lead EMG recording
#     :type array: np.array
#     :param start_index: which index number the breath starts on
#     :type start_index: int
#     :param end_index: which index number the breath ends on
#     :type end_index: int

#     :returns: area; area under the curve
#     :rtype: float
#     """
#     breath = array[start_index:end_index]
#     area = np.sum(abs(breath))
#     return area


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


# def area_under_curve(
#     array,
#     start_index,
#     end_index,
#     end_curve=70,
#     smooth_algorithm='none',
# ):
#     """
#     This algorithm should be applied to breaths longer than 60 values
#     on an index. The mid_savgol assumes a parabolic fit. It is
#     recommended to test a smoothing algorithm first, apply,
#     then run the area_under the curve with none for smooth_algortihm.
#     If a cutoff of the curve before it hits bottom is desired then a value
#     other than zero must be in end_curve variable. This variable
#     should be written from 0 to 100 for the percentage of the max value
#     at which to cut off after the peak.
#     :param array: an array e.g. single lead EMG recording
#     :type array: np.array
#     :param start_index: which index number the breath starts on
#     :type start_index: int
#     :param end_index: which index number the breath ends on
#     :type end_index: int
#     :param end_curve: percentage of peak value to stop summing at
#     :type end_curve: float
#     :param smooth_algorithm: algorithm for smoothing
#     :type smooth_algorithm: str
#     :returns: area; area under the curve
#     :rtype: float
#     """
#     if not (0 <= end_curve <= 100):
#         raise ValueError(
#             'end_curve must be between 0 and 100, '
#             'but {} given'.format(end_curve),
#         )
#     if smooth_algorithm not in ('none', 'mid_savgol'):
#         raise ValueError(
#             'Possible values for smooth_algorithm are none and mid_savgol, '
#             'but {} given'.format(smooth_algorithm),
#         )

#     if array[start_index] < array[end_index]:
#         logging.warning(
#             'You picked an end point above baseline, '
#             'as defined by the last value on the whole curve, '
#             'caution with end_curve variable!',
#         )

#     new_array = array[start_index:end_index + 1]
#     max_ind = new_array.argmax()
#     end_curve = end_curve / 100

#     if smooth_algorithm == 'mid_savgol':
#         new_array = savgol_filter(
#             new_array,
#             len(new_array),
#             2,
#             deriv=0,
#             delta=1.0,
#             axis=- 1,
#             mode='interp',
#             cval=0.0,
#         )

#     tail = new_array[max_ind:] < new_array.max() * end_curve
#     nonzero = np.nonzero(tail)[0]
#     end = nonzero[0] if len(nonzero) else new_array.shape[0] - 1
#     return np.sum(new_array[:(max_ind + end)])


# def find_peak_in_breath(
#     array,
#     start_index,
#     end_index,
#     smooth_algorithm='none'
# ):
#     """
#     This algorithm locates peaks on a breath. It is assumed
#     an array of absolute values for electrophysiological signals
#     will be used as the array. The mid_savgol assumes a parabolic fit.
#     The convy option uses a convolution to essentially
#     smooth values with those around it as in function
#     running_smoother() in the same module.
#     It is recommended to test a smoothing
#     algorithm first, apply, then run the find peak algorithm.

#     :param array: an array e.g. single lead EMG recording
#     :type array: np.array
#     :param start_index: which index number the breath starts on
#     :type start_index: int
#     :param end_index: which index number the breath ends on
#     :type end_index: int
#     :param smooth_algorithm: algorithm for smoothing (none or
#         'mid-savgol' or 'convy')
#     :type smooth_algorithm: str

#     :returns: index of max point, value at max point, smoothed value
#     :rtype: tuple
#     """
#     new_array = array[start_index: (end_index+1)]
#     if smooth_algorithm == 'mid_savgol':
#         new_array2 = savgol_filter(
#             abs(new_array), int(len(new_array)),
#             2,
#             deriv=0,
#             delta=1.0,
#             axis=- 1,
#             mode='interp',
#             cval=0.0,
#         )
#         max_ind = new_array2.argmax()
#         max_val = new_array[max_ind]
#         smooth_max = new_array2[max_ind]
#     elif smooth_algorithm == 'convy':
#         abs_new_array = abs(new_array)
#         new_array2 = running_smoother(abs_new_array)
#         max_ind = new_array2.argmax()
#         max_val = new_array[max_ind]
#         smooth_max = new_array2[max_ind]
#     else:
#         abs_new_array = abs(new_array)
#         max_ind = abs_new_array.argmax()
#         max_val = abs_new_array[max_ind]
#         smooth_max = max_val
#     return (max_ind, max_val, smooth_max)


# def variability_maker(
#         array,
#         segment_size,
#         method='variance',
#         fill_method='avg',
# ):
#     """
#     Calculate variability of segments of an array according to a specific
#     method, then interpolate the values back to the original legnth of array


#     :param array: the input array
#     :type array: ~numpy.ndarray

#     :param segment_size: length over which variabilty calculated
#     :type segment_size: int

#     :param method: method for calculation i.e. variance or standard deviation
#     :type method: str

#     :param fill_method: method to fill missing values at end result array,
#         'avg' will fill with average of last values, 'zeros' fills zeros, and
#         'resample' will resample (not fill) and strech array
#         to the full 'correct' length of the original signal
#     :type method: str

#     :returns: variability_values array showing variability over segments
#     :rtype: ~numpy.ndarray

#     """
#     variability_values = []
#     if method == 'variance':
#         variability_values = [
#             np.var(array[i:i + segment_size])
#             for i in range(1, len(array) - segment_size)
#         ]

#         values_out = np.array(variability_values)

#     elif method == 'std':
#         # Calculate the standard deviation of each segment
#         variability_values = [
#             np.std(array[i:i+segment_size])
#             for i in range(0, len(array), segment_size)
#         ]
#         values_out = np.array(variability_values)

#     else:
#         print("You did not choose an exisitng method")
#     num_missing_values = len(array) - len(values_out)
#     avg_values_end = np.sum(
#         values_out[(len(values_out) - num_missing_values):]
#         ) \
#         / num_missing_values
#     last_values_avg = np.full(num_missing_values, avg_values_end)

#     if fill_method == 'avg':
#         variability_array = np.hstack((values_out, last_values_avg))
#     elif fill_method == 'zeros':
#         variability_array = np.hstack(
#             (values_out, np.zeros(num_missing_values))
#         )
#     elif fill_method == 'resample':
#         variability_array = scipy.signal.resample(values_out, len(array))
#     else:
#         print("You did not choose an exisitng method")
#         variability_array = np.hstack((values_out, last_values_avg))

#     return variability_array


def time_product(
    signal,
    fs,
    starts_s,
    ends_s,
    baseline=None,
):
    """
    Calculate the time product between the signal and the baseline for the
    windows defined by the start_s and end_s sample pairs.
    :param signal: signal to calculate the time product over
    :type signal: ~numpy.ndarray
    :param fs: sampling frequency
    :type fs: ~int
    :param starts_s: list of individual peak start indices
    :type starts_s: ~list
    :param ends_s: list of individual peak end indices
    :type ends_s: ~list
    :param baseline: running Baseline of the signal
    :type baseline: ~numpy.ndarray
    :returns: time_products
    :rtype: list
    """
    if baseline is None:
        baseline = np.zeros(signal.shape)

    time_products = np.zeros(np.asarray(starts_s).shape)
    for idx, (start_s, end_s) in enumerate(zip(starts_s, ends_s)):
        y_delta = signal[start_s:end_s+1]-baseline[start_s:end_s+1]
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
    peaks_s,
    starts_s,
    ends_s,
    aub_window_s,
    baseline,
    ref_signal=None,
):
    """
    Calculate the time product between the baseline and the nadir of the
    reference signal in the aub_window_s for the windows defined by the start_s
    and end_s sample pairs.
    :param signal: signal to calculate the time product over
    :type signal: ~numpy.ndarray
    :param fs: sampling frequency
    :type fs: ~int
    :param peaks_s: list of individual peak indices
    :type peaks_s: ~list
    :param starts_s: list of individual peak start indices
    :type starts_s: ~list
    :param ends_s: list of individual peak end indices
    :type ends_s: ~list
    :param aub_window_s: number of samples before and after peaks_s to look for
    the nadir
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

    aubs = np.zeros(np.asarray(peaks_s).shape)
    for idx, (start_s, peak_s, end_s) in enumerate(
            zip(starts_s, peaks_s, ends_s)):
        y_delta_curve = signal[start_s:end_s+1]-baseline[start_s:end_s+1]
        ref_start_s = max([0, peak_s - aub_window_s])
        ref_end_s = min([len(signal) - 1, peak_s + aub_window_s])
        if (not np.all(np.sign(y_delta_curve[1:]) >= 0)
                and not np.all(np.sign(y_delta_curve[1:]) <= 0)):
            warnings.warn("Warning: Curve for peak idx" + str(idx)
                          + " not entirely above or below baseline. The "
                          + "calculated integrals will cancel out.")

        if np.median(np.sign(y_delta_curve[1:]) >= 0):
            # Positively deflected signal: Baseline below peak
            y_ref = min(ref_signal[ref_start_s:ref_end_s])
            y_delta = baseline[start_s:end_s+1] - y_ref
        elif np.median(np.sign(y_delta_curve[1:]) <= 0):
            # Negatively deflected signal: Baseline above peak
            y_ref = max(ref_signal[ref_start_s:ref_end_s])
            y_delta = y_ref - baseline[start_s:end_s+1]

        aubs[idx] = np.abs(trapezoid(y_delta, dx=1/fs))

    return aubs
