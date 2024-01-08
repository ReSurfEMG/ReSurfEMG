"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to extract features from preprocessed EMG arrays.
"""

import collections
import math
import warnings
import logging
import scipy
from scipy.signal import savgol_filter
from scipy.stats import entropy
import numpy as np

from ..preprocessing.envelope import running_smoother
from ..helper_functions.helper_functions import delay_embedding


def entropical(sig):
    """This function computes something close to certain type of entropy
    of a series signal array.  Input is sig, the signal, and output is an
    array of entropy measurements. The function can be used inside a generator
    to read over slices. Note it is not a true entropy, and works best with
    very small numbers.

    :param sig: array containin the signal
    :type sig: ~numpy.ndarray

    :returns: number for an entropy-like signal using math.log w/base 2
    :rtype: float

    """
    probabilit = [n_x/len(sig) for x, n_x in collections.Counter(sig).items()]
    e_x = [-p_x*math.log(p_x, 2) for p_x in probabilit]
    return sum(e_x)


def entropy_scipy(sli, base=None):
    """
    This function wraps scipy.stats entropy  (which is a Shannon entropy)
    for use in the resurfemg library, it can be used in a slice iterator
    as a drop-in substitute for the hf.entropical but it is a true entropy.

    :param sli: array
    :type sli: ~numpy.ndarray

    :returns: entropy_count
    :rtype: float
    """

    _, counts = np.unique(sli, return_counts=True)
    entropy_count = entropy(counts/len(counts), base=base)
    return entropy_count


def simple_area_under_curve(
    array,
    start_index,
    end_index,
):
    """
    This function is just a wrapper over np.sum written because it isn't
    apparent to some clinically oriented people that an area under the curve
    will be a sum of all the numbers

    :param array: an array e.g. single lead EMG recording
    :type array: np.array
    :param start_index: which index number the breath starts on
    :type start_index: int
    :param end_index: which index number the breath ends on
    :type end_index: int

    :returns: area; area under the curve
    :rtype: float
    """
    breath = array[start_index:end_index]
    area = np.sum(abs(breath))
    return area


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


def area_under_curve(
    array,
    start_index,
    end_index,
    end_curve=70,
    smooth_algorithm='none',
):
    """
    This algorithm should be applied to breaths longer than 60 values
    on an index. The mid_savgol assumes a parabolic fit. It is
    recommended to test a smoothing algorithm first, apply,
    then run the area_under the curve with none for smooth_algortihm.
    If a cutoff of the curve before it hits bottom is desired then a value
    other than zero must be in end_curve variable. This variable
    should be written from 0 to 100 for the percentage of the max value
    at which to cut off after the peak.
    :param array: an array e.g. single lead EMG recording
    :type array: np.array
    :param start_index: which index number the breath starts on
    :type start_index: int
    :param end_index: which index number the breath ends on
    :type end_index: int
    :param end_curve: percentage of peak value to stop summing at
    :type end_curve: float
    :param smooth_algorithm: algorithm for smoothing
    :type smooth_algorithm: str
    :returns: area; area under the curve
    :rtype: float
    """
    if not (0 <= end_curve <= 100):
        raise ValueError(
            'end_curve must be between 0 and 100, '
            'but {} given'.format(end_curve),
        )
    if smooth_algorithm not in ('none', 'mid_savgol'):
        raise ValueError(
            'Possible values for smooth_algorithm are none and mid_savgol, '
            'but {} given'.format(smooth_algorithm),
        )

    if array[start_index] < array[end_index]:
        logging.warning(
            'You picked an end point above baseline, '
            'as defined by the last value on the whole curve, '
            'caution with end_curve variable!',
        )

    new_array = array[start_index:end_index + 1]
    max_ind = new_array.argmax()
    end_curve = end_curve / 100

    if smooth_algorithm == 'mid_savgol':
        new_array = savgol_filter(
            new_array,
            len(new_array),
            2,
            deriv=0,
            delta=1.0,
            axis=- 1,
            mode='interp',
            cval=0.0,
        )

    tail = new_array[max_ind:] < new_array.max() * end_curve
    nonzero = np.nonzero(tail)[0]
    end = nonzero[0] if len(nonzero) else new_array.shape[0] - 1
    return np.sum(new_array[:(max_ind + end)])


def find_peak_in_breath(
    array,
    start_index,
    end_index,
    smooth_algorithm='none'
):
    """
    This algorithm locates peaks on a breath. It is assumed
    an array of absolute values for electrophysiological signals
    will be used as the array. The mid_savgol assumes a parabolic fit.
    The convy option uses a convolution to essentially
    smooth values with those around it as in function
    running_smoother() in the same module.
    It is recommended to test a smoothing
    algorithm first, apply, then run the find peak algorithm.

    :param array: an array e.g. single lead EMG recording
    :type array: np.array
    :param start_index: which index number the breath starts on
    :type start_index: int
    :param end_index: which index number the breath ends on
    :type end_index: int
    :param smooth_algorithm: algorithm for smoothing (none or
        'mid-savgol' or 'convy')
    :type smooth_algorithm: str

    :returns: index of max point, value at max point, smoothed value
    :rtype: tuple
    """
    new_array = array[start_index: (end_index+1)]
    if smooth_algorithm == 'mid_savgol':
        new_array2 = savgol_filter(
            abs(new_array), int(len(new_array)),
            2,
            deriv=0,
            delta=1.0,
            axis=- 1,
            mode='interp',
            cval=0.0,
        )
        max_ind = new_array2.argmax()
        max_val = new_array[max_ind]
        smooth_max = new_array2[max_ind]
    elif smooth_algorithm == 'convy':
        abs_new_array = abs(new_array)
        new_array2 = running_smoother(abs_new_array)
        max_ind = new_array2.argmax()
        max_val = new_array[max_ind]
        smooth_max = new_array2[max_ind]
    else:
        abs_new_array = abs(new_array)
        max_ind = abs_new_array.argmax()
        max_val = abs_new_array[max_ind]
        smooth_max = max_val
    return (max_ind, max_val, smooth_max)


def variability_maker(
        array,
        segment_size,
        method='variance',
        fill_method='avg',
):
    """
    Calculate variability of segments of an array according to a specific
    method, then interpolate the values back to the original legnth of array


    :param array: the input array
    :type array: ~numpy.ndarray

    :param segment_size: length over which variabilty calculated
    :type segment_size: int

    :param method: method for calculation i.e. variance or standard deviation
    :type method: str

    :param fill_method: method to fill missing values at end result array,
        'avg' will fill with average of last values, 'zeros' fills zeros, and
        'resample' will resample (not fill) and strech array
        to the full 'correct' length of the original signal
    :type method: str

    :returns: variability_values array showing variability over segments
    :rtype: ~numpy.ndarray

    """
    variability_values = []
    if method == 'variance':
        variability_values = [
            np.var(array[i:i + segment_size])
            for i in range(1, len(array) - segment_size)
        ]

        values_out = np.array(variability_values)

    elif method == 'std':
        # Calculate the standard deviation of each segment
        variability_values = [
            np.std(array[i:i+segment_size])
            for i in range(0, len(array), segment_size)
        ]
        values_out = np.array(variability_values)

    else:
        print("You did not choose an exisitng method")
    num_missing_values = len(array) - len(values_out)
    avg_values_end = np.sum(
        values_out[(len(values_out) - num_missing_values):]
        ) \
        / num_missing_values
    last_values_avg = np.full(num_missing_values, avg_values_end)

    if fill_method == 'avg':
        variability_array = np.hstack((values_out, last_values_avg))
    elif fill_method == 'zeros':
        variability_array = np.hstack(
            (values_out, np.zeros(num_missing_values))
        )
    elif fill_method == 'resample':
        variability_array = scipy.signal.resample(values_out, len(array))
    else:
        print("You did not choose an exisitng method")
        variability_array = np.hstack((values_out, last_values_avg))

    return variability_array


def rowwise_chebyshev(x, y):
    return np.max(np.abs(x - y), axis=1)


def sampen(
        data,
        emb_dim=2,
        tolerance=None,
        dist=rowwise_chebyshev,
        closed=False,
):
    """
    The following code is adapted from openly licensed code written by
    Christopher Schölzel in his package
    nolds (NOnLinear measures for Dynamical Systems).
    It computes the sample entropy of time sequence data.
    Returns
    the sample entropy of the data (negative logarithm of ratio between
    similar template vectors of length emb_dim + 1 and emb_dim)
    [c_m, c_m1]:
    list of two floats: count of similar template vectors of length emb_dim
    (c_m) and of length emb_dim + 1 (c_m1)
    [float list, float list]:
    Lists of lists of the form ``[dists_m, dists_m1]`` containing the
    distances between template vectors for m (dists_m)
    and for m + 1 (dists_m1).
    Reference
    .. [se_1] J. S. Richman and J. R. Moorman, “Physiological time-series
    analysis using approximate entropy and sample entropy,”
    American Journal of Physiology-Heart and Circulatory Physiology,
    vol. 278, no. 6, pp. H2039-H2049, 2000.

    Kwargs are
    emb_dim (int):
    the embedding dimension (length of vectors to compare)
    tolerance (float):
    distance threshold for two template vectors to be considered equal
    (default: 0.2 * std(data) at emb_dim = 2, corrected for
    dimension effect for other values of emb_dim)
    dist (function (2d-array, 1d-array) -> 1d-array):
    distance function used to calculate the distance between template
    vectors. Sampen is defined using ``rowwise_chebyshev``. You should only
    use something else, if you are sure that you need it.
    closed (boolean):
    if True, will check for vector pairs whose distance is in the closed
    interval [0, r] (less or equal to r), otherwise the open interval
    [0, r) (less than r) will be used

    :param data: array-like
    :type data: array
    :param emb_dim: the embedded dimension
    :type emb_dim: int
    :param tolerance: distance threshold for two template vectors
    :type tolerance: float
    :param distance: function to calculate distance
    :type distance: function

    :returns: saen
    :rtype: float
    """
    data = np.asarray(data)

    if tolerance is None:
        lint_helper = (0.5627 * np.log(emb_dim) + 1.3334)
        tolerance = np.std(data, ddof=1) * 0.1164 * lint_helper
    n = len(data)

    # build matrix of "template vectors"
    # (all consecutive subsequences of length m)
    # x0 x1 x2 x3 ... xm-1
    # x1 x2 x3 x4 ... xm
    # x2 x3 x4 x5 ... xm+1
    # ...
    # x_n-m-1     ... xn-1

    # since we need two of these matrices for m = emb_dim and
    #  m = emb_dim +1,
    # we build one that is large enough => shape (emb_dim+1, n-emb_dim)

    # note that we ignore the last possible template vector with
    #  length emb_dim,
    # because this vector has no corresponding vector of length m+
    # 1 and thus does
    # not count towards the conditional probability
    # (otherwise first dimension would be n-emb_dim+1 and not n-emb_dim)
    t_vecs = delay_embedding(np.asarray(data), emb_dim + 1, lag=1)
    counts = []
    for m in [emb_dim, emb_dim + 1]:
        counts.append(0)
        # get the matrix that we need for the current m
        t_vecs_m = t_vecs[:n - m + 1, :m]
        # successively calculate distances between each pair of templ vectrs
        for i in range(len(t_vecs_m) - 1):
            dsts = dist(t_vecs_m[i + 1:], t_vecs_m[i])
            # count how many distances are smaller than the tolerance
            if closed:
                counts[-1] += np.sum(dsts <= tolerance)
            else:
                counts[-1] += np.sum(dsts < tolerance)
    if counts[0] > 0 and counts[1] > 0:
        saen = -np.log(1.0 * counts[1] / counts[0])
    else:
        # log would be infinite or undefined => cannot determine saen
        zcounts = []
        if counts[0] == 0:
            zcounts.append("emb_dim")
        if counts[1] == 0:
            zcounts.append("emb_dim + 1")
        print_message = (
            "Zero vectors are within tolerance for {}. "
            "Consider raising tolerance parameter to avoid {} result."
        )
        warnings.warn(
            print_message.format(
                " and ".join(zcounts),
                "NaN" if len(zcounts) == 2 else "inf",
            ),
            RuntimeWarning
        )
        if counts[0] == 0 and counts[1] == 0:
            saen = np.nan
        elif counts[0] == 0:
            saen = -np.inf
        else:
            saen = np.inf
    return saen


def sampen_optimized(
        data,
        tolerance=None,
        closed=False,
):
    """

    The following code is adapted from openly licensed code written by
    Christopher Schölzel in his package
    nolds (NOnLinear measures for Dynamical Systems).
    It computes the sample entropy of time sequence data.
    emb_dim has been set to 1 (not parameterized)
    Returns
    the sample entropy of the data (negative logarithm of ratio between
    similar template vectors of length emb_dim + 1 and emb_dim)
    [c_m, c_m1]:
    list of two floats: count of similar template vectors of length emb_dim
    (c_m) and of length emb_dim + 1 (c_m1)
    [float list, float list]:
    Lists of lists of the form ``[dists_m, dists_m1]`` containing the
    distances between template vectors for m (dists_m)
    and for m + 1 (dists_m1).
    Reference:
    .. [se_1] J. S. Richman and J. R. Moorman, “Physiological time-series
    analysis using approximate entropy and sample entropy,”
    American Journal of Physiology-Heart and Circulatory Physiology,
    vol. 278, no. 6, pp. H2039–H2049, 2000.

    Kwargs are pre-set and not available. For more extensive
    you should use the sampen function.

    :param data: array-like
    :type data: array
    :param tolerance: distance threshold for two template vectors
    :type tolerance: float
    :param distance: function to calculate distance
    :type distance: function

    :returns: saen
    :rtype: float
    """
    # TODO: this function can still be further optimized
    data = np.asarray(data)
    if tolerance is None:
        lint_helper = (0.5627 * np.log(1) + 1.3334)
        tolerance = np.std(data, ddof=1) * 0.1164 * lint_helper
    n = len(data)

    # TODO(): This can be done with just using NumPy
    t_vecs = delay_embedding(np.asarray(data), 3, lag=1)

    if closed:
        counts = calc_closed_sampent(t_vecs, n, tolerance)
    else:
        counts = calc_open_sampent(t_vecs, n, tolerance)

    if counts[0] > 0 and counts[1] > 0:
        saen = -np.log(1.0 * counts[1] / counts[0])
    else:
        # log would be infinite or undefined => cannot determine saen
        zcounts = []
        if counts[0] == 0:
            zcounts.append("1")
        if counts[1] == 0:
            zcounts.append("2")
        print_message = (
            "Zero vectors are within tolerance for {}. "
            "Consider raising tolerance parameter to avoid {} result."
        )
        warnings.warn(
            print_message.format(
                " and ".join(zcounts),
                "NaN" if len(zcounts) == 2 else "inf",
            ),
            RuntimeWarning
        )
        if counts[0] == 0 and counts[1] == 0:
            saen = np.nan
        elif counts[0] == 0:
            saen = -np.inf
        else:
            saen = np.inf
    return saen


def calc_closed_sampent(t_vecs, n, tolerance):
    # TODO(someone?): Analogous to calc_open_sampent
    return np.nan, np.nan


def calc_open_sampent(t_vecs, n, tolerance):
    triplets = t_vecs[:n - 2, :3]

    raw_dsts = tuple(
        triplets[i + 1:] - triplets[i]
        for i in range(len(triplets) - 1)
    )
    dsts = np.concatenate(raw_dsts)
    dsts_abs = np.abs(dsts)
    dsts_gt = dsts_abs < tolerance
    dsts_max_a = np.logical_and(dsts_gt[:, 0], dsts_gt[:, 1])
    dsts_max = np.logical_and(dsts_max_a, dsts_gt[:, 2])
    return np.sum(dsts_max_a), np.sum(dsts_max)


def entropy_maker(
        array,
        method='sample_entropy',
        base=None,
):
    """
    The following code allows a user to input an array and calculate either
    a time-series specific entropy i.e. the nolds or a more general
    Shannon entropy as calculated in scipy.
    It calls entropy functions in the file.

    """
    if method == 'scipy':
        output = entropy_scipy(array, base=base)
    elif method == 'nolds':
        output = sampen(array)
    elif method == 'sample_entropy':
        output = sampen_optimized(array)
    else:
        print('your method is not an option,')
        print('we defaulted to a slow unoptimized sample entropy')
        output = sampen(array)
    return output


def snr_pseudo(
        src_signal,
        peaks,
        baseline=np.array([]),
):
    """
    Approximate the signal-to-noise ratio (SNR) of the signal based
    on the peak height relative to the baseline.

    :param signal: Signal to evaluate
    :type signal: ~numpy.ndarray
    :param peaks: list of individual peak indices
    :type gate_peaks: ~list
    :param signal: Signal to evaluate
    :type baseline: ~numpy.ndarray


    :returns: snr_peaks, the SNR per peak
    :rtype: ~numpy.ndarray
    """

    if len(baseline) != len(src_signal):
        baseline = np.zeros(
            (len(src_signal), ))

        baseline_w_emg = 5 * 2048  # window length
        for idx in range(len(src_signal)):
            start_i = max([0, idx-int(baseline_w_emg/2)])
            end_i = min([len(src_signal), idx+int(baseline_w_emg/2)])
            baseline[idx] = np.percentile(src_signal[start_i:end_i], 33)

    peak_heights = np.zeros((len(peaks),))
    noise_heights = np.zeros((len(peaks),))

    for peak_nr, idx in enumerate(peaks):
        peak_heights[peak_nr] = src_signal[idx]
        start_i = max([0, idx-2048])
        end_i = min([len(src_signal), idx+2048])
        noise_heights[peak_nr] = np.median(baseline[start_i:end_i])

    snr_peaks = np.divide(peak_heights, noise_heights)
    return snr_peaks
