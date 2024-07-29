"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
This file contains general functions to support the functions in this
repository
"""

import os
from collections import namedtuple
import numpy as np
import pandas as pd


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
