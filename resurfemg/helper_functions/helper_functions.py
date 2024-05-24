"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
This file contains general functions to support the functions in this
repository
"""

import os
from collections import namedtuple
import glob
import numpy as np
import textdistance
import pandas as pd
import scipy
from ..data_connector.tmsisdk_lite import Poly5Reader
from ..pipelines.pipelines import alternative_a_pipeline_multi
from ..pipelines.pipelines import alternative_b_pipeline_multi
from ..pipelines.pipelines import working_pipeline_pre_ml_multi


class Range(namedtuple('RangeBase', 'start,end')):

    """Utility class for working with ranges (intervals).
    :ivar start: Start of the range
    :type start: ~number.Number
    :ivar end: End of the range
    :type end: ~number.Number
    """

    def intersects(self, other):
        """Returns :code:`True` if this range intersects :code:`other` range.
        :param other: Another range to compare this one to
        :type other: ~resurfemg.helper_functions.Range
        :returns: :code:`True` if this range intersects another range
        :rtype: bool
        """
        return (
            (self.end >= other.end) and (self.start < other.end) or
            (self.end >= other.start) and (self.start < other.start) or
            (self.end < other.end) and (self.start >= other.start)
        )

    def precedes(self, other):
        """Returns :code:`True` if this range precedes :code:`other` range.
        :param other: Another range to compare this one to
        :type other: ~resurfemg.helper_functions.Range
        :returns: :code:`True` if this range strictly precedes another range
        :rtype: bool
        """
        return self.end < other.start

    def to_slice(self):
        """Converts this range to a :class:`slice`.
        :returns: A slice with its start set to this range's start and end set
        to this range's end
        :rtype: slice
        """
        return slice(*map(int, self))   # maps whole tuple set


def zero_one_for_jumps_base(array, cut_off):
    """This function takes an array and makes it binary (0, 1) based
    on a cut-off value.
    :param array: An array
    :type array: ~numpy.ndarray
    :param cut_off: The number defining a cut-off line for binarization
    :type cut_off: float
    :returns: Binarized list that can be turned into array
    :rtype: list
    """
    array_list = []
    for i in array:
        if i < cut_off:
            i = 0
        else:
            i = 1
        array_list.append(i)
    return array_list


def slices_slider(array_sample, slice_len):
    """This function produces continuous sequential slices over an
    array of a certain length.  The inputs are the following -
    :code:`array_sample`, the signal and :code:`slice_len` - the
    window which you wish to slide with.  The function yields, does
    not return these slices.
    :param array_sample: array containing the signal
    :type array_sample: ~numpy.ndarray
    :param slice_len: the length of window on the array
    :type slice_len: int
    :returns: Actually yields, no return
    :rtype: ~numpy.ndarray
    """
    for i in range(len(array_sample) - slice_len + 1):
        yield array_sample[i:i + slice_len]


def slices_jump_slider(array_sample, slice_len, jump):
    """
    This function produces continuous sequential slices over an
    array of a certain length spaced out by a 'jump'.
    The function yields, does
    not return these slices.
    :param array_sample: array containing the signal
    :type array_sample: ~numpy.ndarray
    :param slice_len: the length of window on the array
    :type slice_len: int
    :param jump: the amount by which the window is moved at iteration
    :type jump: int
    :returns: Actually yields, no return
    :rtype: ~numpy.ndarray
    """
    for i in range(len(array_sample) - (slice_len)):
        yield array_sample[(jump*i):((jump*i) + slice_len)]


def count_decision_array(decision_array):
    """This is a function that, practically speaking, counts events
    on a time series array that has been reduced down to a binary
    (0, 1) output. It counts changes then divides by two.
    :param decision_array: Array.
    :type decisions_array: ~numpy.ndarray
    :returns: Number of events
    :rtype: float
    """
    ups_and_downs = np.logical_xor(decision_array[1:], decision_array[:-1])
    count = ups_and_downs.sum()/2
    return count


def ranges_of(array):
    """This function is made to work with :class:`Range` class objects, such
    that is selects ranges and returns tuples of boundaries.
    :param my_own_array: array
    :type  my_own_array: ~numpy.ndarray
    :return: range_return
    :rtype: tuple
    """
    marks = np.logical_xor(array[1:], array[:-1])
    boundaries = np.hstack(
        (np.zeros(1), np.where(marks != 0)[0], np.zeros(1) + len(array) - 1)
    )
    if not array[0]:
        boundaries = boundaries[1:]
    if len(boundaries) % 2 != 0:
        boundaries = boundaries[:-1]
    range_return = tuple(
        Range(*boundaries[i:i+2]) for i in range(0, len(boundaries), 2)
    )
    return range_return


def intersections(left, right):
    """This function works over two arrays, :code:`left` and
    :code:`right`, and allows a picking based on intersections.  It
    only takes ranges on the left that intersect ranges on the right.
    :param left: List of ranges
    :type left: List[Range]
    :param right: List of ranges
    :type right: List[Range]
    :returns: Ranges from the :code:`left` that intersect ranges from
    the :code:`right`.
    :rtype: List[Range]
    """
    i, j = 0, 0
    result = []
    while i < len(left) and j < len(right):
        lelt, relt = left[i], right[j]
        if lelt.intersects(relt):
            result.append(lelt)
            i += 1
        elif relt.precedes(lelt):
            j += 1
        elif lelt.precedes(relt):
            i += 1
    return result


def raw_overlap_percent(signal1, signal2):
    """This function takes two binary 0 or 1 signal arrays and gives
    the percentage of overlap.
    :param signal1: Binary signal 1
    :type signal1: ~numpy.ndarray
    :param rsignal2: Binary signal 2
    :type rsignal2: ~numpy.ndarray
    :returns: Raw overlap percent
    :rtype: float
    """
    if len(signal1) != len(signal2):
        print('Warning: length of arrays is not matched')
        longer_signal_len = np.max([len(signal1), len(signal2)])
    else:
        longer_signal_len = len(signal1)

    _raw_overlap_percent = sum(
        signal1.astype(int) & signal2.astype(int)
    ) / longer_signal_len
    return _raw_overlap_percent


def relative_levenshtein(signal1, signal2):
    """
    Here we take two arrays, and create an edit distance based on Levelshtien
    edit distance The distance is then normalized between 0 and one regardless
    of signal length
    """
    signal1_list = []
    signal2_list = []
    for element in signal1:
        signal1_list.append(element)
    for element in signal2:
        signal2_list.append(element)
    distance = textdistance.levenshtein.similarity(signal1_list, signal2_list)
    if len(signal1) != len(signal2):
        print('Warning: length of arrays is not matched')
    longer_signal_len = np.max([len(signal1), len(signal2)])
    normalized_distance = distance / longer_signal_len
    return normalized_distance


def merge(left, right):
    """
    Mergey function
    """
    # Initialize an empty list output that will be populated
    # with sorted elements.
    # Initialize two variables i and j which are used pointers when
    # iterating through the lists.
    output = []
    i = j = 0

    # Executes the while loop if both pointers i and j are less than
    # the length of the left and right lists
    while i < len(left) and j < len(right):
        # Compare the elements at every position of both
        # lists during each iteration
        if left[i] < right[j]:
            # output is populated with the lesser value
            output.append(left[i])
            # 10. Move pointer to the right
            i += 1
        else:
            output.append(right[j])
            j += 1
    # The remnant elements are picked from the current
    # pointer value to the end of the respective list
    output.extend(left[i:])
    output.extend(right[j:])

    return output


def scale_arrays(array, maximumn, minimumn):
    """
    This function will scale all arrays along
    the vertical axis to have an absolute maximum
    value of the maximum parameter
    :param array: Original signal array with any number iflayers
    :type array: ~numpy.ndarray
    :param maximumn: the absolute maximum below which the new array exists
    :type maximumn: float
    :param minimumn: the absolute maximum below which the new array exists
    :type minimumn: float
    :returns: reformed, a new array with absolute max of maximum
    :rtype: ~numpy.ndarray
    """
    reformed = np.interp(
        array,
        (array.min(), array.max()),
        (maximumn, minimumn)
    )
    return reformed


def distance_matrix(array_a, array_b):
    """
    :param array_a: an array of same size as other parameter array
    :type array_a: array or list
    :param array_b: an array of same size as other parameter array
    :type array_b: array or list
    :returns: distances
    :rtype: pd.DataFrame
    """
    if len(array_a) != len(array_b):
        print('Your arrays do not match in length, caution!')
    array_a_list = array_a.tolist()
    array_b_list = array_b.tolist()
    distance_earthmover = scipy.stats.wasserstein_distance(array_a, array_b)
    distance_edit_distance = textdistance.levenshtein.similarity(
        array_a_list,
        array_b_list,
    )
    distance_euclidian = scipy.spatial.distance.euclidean(array_a, array_b)
    distance_hamming = scipy.spatial.distance.hamming(array_a, array_b)
    distance_chebyshev = scipy.spatial.distance.cityblock(array_a, array_b)
    distance_cosine = scipy.spatial.distance.cosine(array_a, array_b)
    data_made = {
        'earthmover': distance_earthmover,
        'edit_distance': distance_edit_distance,
        'euclidean': distance_euclidian,
        'hamming': distance_hamming,
        'chebyshev': distance_chebyshev,
        'cosine': distance_cosine,
    }
    distances = pd.DataFrame(data=data_made, index=[0])
    return distances


def delay_embedding(data, emb_dim, lag=1):
    """
    The following code is adapted from openly licensed code written by
    Christopher Schölzel in his package nolds
    (NOnLinear measures for Dynamical Systems).
    It performs a time-delay embedding of a time series
    :param data: array-like
    :type data: array
    :param emb_dim: the embedded dimension
    :type emb_dim: int
    :param lag: the lag between elements in the embedded vectors
    :type lag: int
    :returns: matrix_vectors
    :rtype: ~nd.array
    """
    data = np.asarray(data)
    min_len = (emb_dim - 1) * lag + 1
    if len(data) < min_len:
        msg = "cannot embed data of length {} with embedding dimension {} " \
            + "and lag {}, minimum required length is {}"
        raise ValueError(msg.format(len(data), emb_dim, lag, min_len))
    m = len(data) - min_len + 1
    indices = np.repeat([np.arange(emb_dim) * lag], m, axis=0)
    indices += np.arange(m).reshape((m, 1))
    matrix_vectors = data[indices]
    return matrix_vectors


def save_preprocessed(array, out_fname, force):
    """
    This function is written to be called by the cli module.
    It stores arrays in a directory.
    """
    if not force:
        if os.path.isfile(out_fname):
            return
    try:
        os.makedirs(os.path.dirname(out_fname))
    except FileExistsError:
        pass
    np.save(out_fname, array, allow_pickle=False)


def preprocess(
    file_directory,
    our_chosen_leads,
    algorithm,
    processed,
    force=False
):
    """
    This function is written to be called by the cli module.
    The cli module supports command line pre-processing.
    This function is currently written to accomodate Poly5 files types.
    It can be refactored later.

    :param file_directory: the directory with EMG files
    :type file_directory: str
    :param processed: the output directory
    :type processed: str
    :param our_chosen_leads: the leads selected for the pipeline to run over
    :type our_chosen_leads: list

    """
    file_directory_list = glob.glob(
        os.path.join(file_directory, '**/*.Poly5'),
        recursive=True,
    )
    for file in file_directory_list:
        reader = Poly5Reader(file)
        if algorithm == 'alternative_a_pipeline_multi':
            array = alternative_a_pipeline_multi(
                reader.samples,
                our_chosen_leads,
                picker='heart',
            )
        elif algorithm == 'alternative_b_pipeline_multi':
            array = alternative_b_pipeline_multi(
                reader.samples,
                our_chosen_leads,
                picker='heart',
            )

        else:
            array = working_pipeline_pre_ml_multi(
                reader.samples,
                our_chosen_leads,
                picker='heart',
            )
        rel_fname = os.path.relpath(file, file_directory)
        out_fname = os.path.join(processed, rel_fname)
        save_preprocessed(array, out_fname, force)


def derivative(signal, fs, window_s=None):
    """This function calculates the first derivative of a signal. If
    window_s is given, the signal is smoothed before derivative calculation.
    :param signal: signal to calculate the derivate over
    :type signal: ~numpy.ndarray
    :param fs: sample rate
    :type fs: int
    :param window_s: centralised averaging window length in samples
    :type window_s: int
    :returns: The first derivative of the signal length len(signal)-1.
    :rtype: ~numpy.ndarray
    """

    if window_s is not None:
        # Moving average over signal
        signal_series = pd.Series(signal)
        signal_moving_average = signal_series.rolling(
            window=window_s, center=True).mean().values
        dsignal_dt = (signal_moving_average[1:]
                      - signal_moving_average[:-1]) * fs
    else:
        dsignal_dt = (signal[1:] - signal[:-1]) * fs

    return dsignal_dt
