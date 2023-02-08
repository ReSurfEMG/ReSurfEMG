"""
Copyright 2022 Netherlands eScience Center and UTwente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to work with various EMG arrays
and other types of data arrays e.g. ventilator signals.

"""

import collections
from collections import namedtuple
import math
from math import log, e
import copy
import scipy
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
import textdistance
import pandas as pd
import logging


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

        :returns: :code:`True` if this range strictly precedes another
            range
        :rtype: bool
        """
        return self.end < other.start

    def to_slice(self):
        """Converts this range to a :class:`slice`.

        :returns: A slice with its start set to this range's start and
            end set to this range's end
        :rtype: slice
        """
        return slice(*map(int, self))   # maps whole tuple set


def emg_bandpass_butter(data_emg, low_pass, high_pass):
    """The parameter taken in here is the Poly5 file. Output is
    the emg after a bandpass as made here.

    :param data_emg: Poly5 file with the samples to work over
    :type data_emg: ~TMSiSDK.file_readers.Poly5Reader
    :param low_pass: The number to cut off :code:`frequenciesabove`
    :type low_pass: int
    :param high_pass: The number to cut off :code:`frequenceisbelow`
    :type high_pass: int

    :returns: The bandpass filtered EMG sample data
    :rtype: ~numpy.ndarray
    """
    sos = signal.butter(
        3,
        [low_pass, high_pass],
        'bandpass',
        fs=data_emg.sample_rate,
        output='sos',
    )
    # sos (output parameter)is second order section  -> "stabilizes" ?
    emg_filtered = signal.sosfiltfilt(sos, data_emg.samples)
    return emg_filtered


def emg_bandpass_butter_sample(
    data_emg_samp,
    low_pass,
    high_pass,
    sample_rate,
    output='sos'
):
    """The paramemter taken in here is the Poly5 file.  Output is the
    EMG after a bandpass as made here.

    :param data_emg_samp: The array in the poly5 or other sample
    :type data_emg_samp: ~numpy.ndarray
    :param low_pass: The number to cut off :code:`frequenciesabove`
    :type low_pass: int
    :param high_pass: The number to cut off :code:`frequenceisbelow`
    :type high_pass: int
    :param sample_rate: The number of samples per second i.e. Hertz
    :type sample_rate: int
    :param output: The type of sampling stabilizor
    :type high_pass: str

    :returns: The bandpass filtered EMG sample data
    :rtype: ~numpy.ndarray
    """
    sos = signal.butter(
        3,
        [low_pass, high_pass],
        'bandpass',
        fs=sample_rate,
        output='sos',
    )
    # sos (output parameter)is second order section  -> "stabilizes" ?
    emg_filtered = signal.sosfiltfilt(sos, data_emg_samp)
    return emg_filtered


def bad_end_cutter(data_emg, percent_to_cut=7, tolerance_percent=10):
    """This algorithm takes the end off of EMGs where the end is
    radically altered, or if not radically altered cuts the last 10
    values but returns only the array, not an altered Poly5.

    :param data_emg: A poly5
    :type data_emg: ~TMSiSDK.file_readers.Poly5Reader
    :param percent_to_cut: Percentage to look at on the end
    :type percent_to_cut: int
    :param tolerance_percent: Percent variation tolerance to allow without
        cutting
    :type tolerance_percent: int

    :returns: The cut EMG sample data
    :rtype: ~numpy.ndarray
    """
    sample = data_emg.samples
    len_sample = len(data_emg.samples[0])

    last_half = data_emg.samples[:, int(len_sample/2):]
    percent = abs(int(percent_to_cut))
    cut_number_last = int(((100 - percent)/100) * len_sample)
    cut_number_last = int(((100 - percent)/100) * len_sample)
    last_part = data_emg.samples[:, cut_number_last:]
    leads = last_half.shape[0]
    percent_off_list = []
    for i in range(0, leads):
        last_half_means = last_half[i].mean()
        last_part_means = last_part[i].mean()
        difference = abs(last_half_means - last_part_means)/last_half_means
        percent_off_list.append(difference)
    tolerance_list = []
    for element in percent_off_list:
        tolerance = tolerance_percent/100

        if element >= tolerance:
            booly = True
            tolerance_list.append(booly)
        else:
            booly = False
            tolerance_list.append(booly)

    if True in tolerance_list:
        sample_cut = sample[:, :cut_number_last]
    else:
        sample_cut = sample[:, :-10]

    return sample_cut


def bad_end_cutter_for_samples(
    data_emg,
    percent_to_cut=7,
    tolerance_percent=10
):
    """This algorithm takes the end off of EMGs where the end is
    radically altered, or if not radically altered cuts the last 10
    values but returns only the array.

    :param data_emg: Array of samples
    :type data_emg: ~numpy.ndarray
    :param percent_to_cut: Percentage to look at on the end
    :type percent_to_cut: int
    :param tolerance_percent: Percent variation to allow without cutting
    :type tolerance_percent: int

    :returns: The cut EMG sample data
    :rtype: ~numpy.ndarray
    """
    sample = data_emg
    len_sample = len(data_emg[0])

    last_half = data_emg[:, int(len_sample/2):]
    percent = abs(int(percent_to_cut))
    cut_number_last = int(((100 - percent)/100) * len_sample)

    last_part = data_emg[:, cut_number_last:]
    leads = last_half.shape[0]
    percent_off_list = []
    # get rid of for loop, take advange of numpy array- next version
    for i in range(leads):
        last_half_means = last_half[i].mean()
        last_part_means = last_part[i].mean()
        difference = abs(last_half_means - last_part_means)/last_half_means
        percent_off_list.append(difference)
    tolerance = tolerance_percent / 100
    if any(elt >= tolerance for elt in percent_off_list):
        sample_cut = sample[:, :cut_number_last]
    else:
        sample_cut = sample[:, :-10]

    return sample_cut


def bad_end_cutter_better(data_emg, percent_to_cut=7, tolerance_percent=10):
    """This algorithm takes the end off of EMGs where the end is
    radically altered, or if not radically altered cuts the last 10
    values but returns only the array not an altered Poly5.

    :param data_emg: A poly5
    :type data_emg: ~TMSiSDK.file_readers.Poly5Reader
    :param percent_to_cut: Percentage to look at on the end
    :type percent_to_cut: int
    :param tolerance_percent: Percentage variation to allow without cut
    :type tolerance_percent: int

    :returns: The cut EMG sample data
    :rtype: ~numpy.ndarray
    """
    sample = data_emg.samples
    len_sample = len(data_emg.samples[0])

    last_half = data_emg.samples[:, int(len_sample/2):]
    percent = abs(int(percent_to_cut))
    cut_number_last = int(((100 - percent)/100) * len_sample)

    last_part = data_emg.samples[:, cut_number_last:]
    leads = last_half.shape[0]
    percent_off_list = []
    # get rid of for loop, take advange of numpy array- next version
    for i in range(leads):
        last_half_means = last_half[i].mean()
        last_part_means = last_part[i].mean()
        difference = abs(last_half_means - last_part_means)/last_half_means
        percent_off_list.append(difference)
    tolerance = tolerance_percent / 100
    if any(elt >= tolerance for elt in percent_off_list):
        sample_cut = sample[:, :cut_number_last]
    else:
        sample_cut = sample[:, :-10]

    return sample_cut


def notch_filter(sample, sample_frequ, freq_to_pull, quality_factor_q):
    """This is a filter designed to take out a specific frequency.  In
    the EU in some data electrical cords can interfere at around 50
    hertz.  In some other locations the interference is at 60 Hertz.
    The specificities of a local power grid may neccesitate notch
    filtering.

    :param sample: Percentage variation tolerance to allow without cutting
    :type sample: int
    :param sample_frequ: The frequency at which the sample was captured
    :type sample_frequ: int
    :param freq_to_pull: The frequency you desire to remove from teh signal
    :type freq_to_pull: int
    :param quality_factor_q: How high the quality of the removal is
    :type quality_factor_q: int

    :return: The filterered sample data
    :rtype: ~numpy.ndarray

    """
    # create notch filter
    # design a notch filter using signal.iirnotch
    b_notch, a_notch = signal.iirnotch(
        freq_to_pull,
        quality_factor_q,
        sample_frequ)

    # make the output signal
    output_signal = signal.filtfilt(b_notch, a_notch, sample)
    return output_signal


def show_my_power_spectrum(sample, sample_rate, upper_window):
    """This function plots a power spectrum of the frequencies
    comtained in an emg based on a fourier transform.  It does not
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
    N = len(sample)
    # for our emgs sample rate is usually 2048
    yf = np.abs(fft(sample))**2
    xf = fftfreq(N, 1 / sample_rate)

    idx = [i for i, v in enumerate(xf) if (0 <= v <= upper_window)]

    plt.plot(xf[idx], yf[idx])
    plt.show()
    return yf, xf


def emg_highpass_butter(data_emg, cut_above, sample_rate):
    """The paramemter taken in here is the Poly5 file's samples or
    another array.  Output is the EMG after a bandpass as made here.

    :param data_emg: Samples from the EMG
    :type data_emg: ~numpy.ndarray
    :param cut_above: The number to cut off :code:`frequenceisbelow`
    :type cut_above: int
    :param sample_rate: The number of samples per second i.e. Hertz
    :type sample_rate: int

    :returns: The bandpass filtered EMG sample data
    :rtype: ~numpy.ndarray
    """
    sos = signal.butter(3, cut_above, 'highpass', fs=sample_rate, output='sos')
    # sos (output parameter)is second order section  -> "stabilizes" ?
    emg_filtered = signal.sosfiltfilt(sos, data_emg)
    return emg_filtered


def naive_rolling_rms(x, N):
    """This function computes a root mean squared envelope over an
    array :code:`x`.  To do this it uses number of sample values
    :code:`N`.

    :param x: Samples from the EMG
    :type x: ~numpy.ndarray
    :param N: Legnth of the sample use as window for function
    :type N: int

    :returns: The root-mean-squared EMG sample data
    :rtype: ~numpy.ndarray
    """
    xc = np.cumsum(abs(x)**2)
    emg_rms = np.sqrt((xc[N:] - xc[:-N])/N)
    return emg_rms


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
    xc = np.cumsum(np.abs(x)**2)
    emg_rms = np.sqrt((xc[N:] - xc[:-N])/N)
    return emg_rms


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


def compute_ICA_two_comp(emg_samples):
    """A function that performs an independant component analysis
    (ICA) meant for EMG data that includes stacked three arrays.

    :param emg_samples: Original signal array with three layers
    :type emg_samples: ~numpy.ndarray

    :returns: Two arrays of independent components (ECG-like and EMG)
    :rtype: ~numpy.ndarray
    """
    X = np.c_[emg_samples[0], emg_samples[2]]
    ica = FastICA(n_components=2, random_state=1)
    S = ica.fit_transform(X)
    component_0 = S.T[0]
    component_1 = S.T[1]
    return component_0, component_1


def pick_more_peaks_array(components_tuple):
    """Here we have a function that takes a tuple with the two parts
    of ICA, and finds the one with more peaks and anti-peaks.  The EMG
    if without a final envelope will have more peaks

    .. note::
        Data should not have been finally filtered to envelope level

    :param components_tuple: tuple of two arrays representing different signals
    :type components_tuple: Tuple[~numpy.ndarray, ~numpy.ndarray]

    :return: Array with more peaks (should usually be the EMG as
        opposed to ECG)
    :rtype: ~numpy.ndarray
    """
    c0 = components_tuple[0]
    c1 = components_tuple[1]
    low_border_c0 = (c0.max() - c0.mean())/4
    peaks0, _0 = find_peaks(c0, height=low_border_c0, distance=10)
    antipeaks0, anti_0 = find_peaks(
        (c0*(-1)),
        height=-low_border_c0,
        distance=10)
    low_border_c1 = (c1.max() - c1.mean())/4
    peaks1, _1 = find_peaks(c1, height=low_border_c1, distance=10)
    antipeaks1, anti_1 = find_peaks(
        (c1*(-1)),
        height=-low_border_c1,
        distance=10,
    )

    sum_peaks_0 = len(peaks0) + len(antipeaks0)
    sum_peaks_1 = len(peaks1) + len(antipeaks1)

    if sum_peaks_0 > sum_peaks_1:
        emg_component = components_tuple[0]
    elif sum_peaks_1 > sum_peaks_0:
        emg_component = components_tuple[1]
    else:
        print("this is very strange data, please examine by hand")
    return emg_component


def pick_lowest_correlation_array(components_tuple, ecg_lead):
    """Here we have a function that takes a tuple with the two parts
    of ICA and the array containing the ECG recording, and finds the
    ICA component with the lowest similarity to the ECG.
    Data should not have been finally filtered to envelope level

    :param components_tuple: tuple of two arrays representing different signals
    :type components_tuple: Tuple[~numpy.ndarray, ~numpy.ndarray]

    :param ecg_lead: array containing the ECG recording
    :type ecg_lead: numpy.ndarray

    :returns: Array with the lowest correlation coefficient
     to the ECG lead (should usually be the EMG as opposed to ECG)
    :rtype: ~numpy.ndarray
    """
    c0 = components_tuple[0]
    c1 = components_tuple[1]

    # create a tuple containing the data, each row is a variable,
    # each column is an observation

    corr_tuple = np.row_stack((ecg_lead, c0, c1))

    # compute the correlation matrix
    # the absolute value is used, because the ICA decomposition might
    # produce a component with negative peaks. In this case
    # the signals will be maximally negatively correlated

    corr_matrix = abs(np.corrcoef(corr_tuple))

    # get the component with the lowest correlation to ECG
    # the matriz is symmetric, so we can check just the first row
    # the first coefficient is the autocorrelation of the ECG lead,
    # so we can check row 1 and 2

    lowest_index = np.argmin(corr_matrix[0][1:])
    emg_component = components_tuple[lowest_index]

    return emg_component


def pick_highest_correlation_array(components_tuple, ecg_lead):
    """Here we have a function that takes a tuple with the two parts
    of ICA and the array containing the ECG recording, and finds the
    ICA component with the highest similarity to the ECG.
    Data should not have been finally filtered to envelope level

    :param components_tuple: tuple of two arrays representing different signals
    :type components_tuple: Tuple[~numpy.ndarray, ~numpy.ndarray]

    :param ecg_lead: array containing the ECG recording
    :type ecg_lead: numpy.ndarray

    :returns: Array with the highest correlation coefficient
     to the ECG lead (should usually be the  ECG)
    :rtype: ~numpy.ndarray
    """
    c0 = components_tuple[0]
    c1 = components_tuple[1]
    corr_tuple = np.row_stack((ecg_lead, c0, c1))
    corr_matrix = abs(np.corrcoef(corr_tuple))

    # get the component with the highest correlation to ECG
    # the matriz is symmetric, so we can check just the first row
    # the first coefficient is the autocorrelation of the ECG lead,
    # so we can check row 1 and 2

    hi_index = np.argmax(corr_matrix[0][1:])
    ecg_component = components_tuple[hi_index]

    return ecg_component


def working_pipeline_exp(our_chosen_file):
    """This function is legacy.
    It produces a filtered respiratory EMG signal from a
    3 lead sEMG file. A better
    option is a corresponding function in multi_lead_type
    The inputs are :code:`our_chosen_file` which we
    give the function as a string of filename.  The output is the
    processed EMG signal filtered and seperated from ecg components.
    The algorithm to pick out the EMG here is by having
    more peaks.

    :param our_chosen_file: Poly5 file
    :type our_chosen_file: ~TMSiSDK.file_readers.Poly5Reader

    :returns: final_envelope_a
    :rtype: ~numpy.ndarray
    """
    cut_file_data = bad_end_cutter(
        our_chosen_file,
        percent_to_cut=3,
        tolerance_percent=5,
    )
    bd_filtered_file_data = emg_bandpass_butter_sample(
        cut_file_data,
        5,
        450,
        2048,
        output='sos',
    )
    # end-cutting again to get rid of filtering artifacts
    re_cut_file_data = bad_end_cutter_for_samples(
        bd_filtered_file_data,
        percent_to_cut=3,
        tolerance_percent=5,
    )
    # do ICA
    components = compute_ICA_two_comp(re_cut_file_data)
    #  pick components with more peaj
    emg = pick_more_peaks_array(components)
    # now process it in final steps
    abs_values = abs(emg)
    final_envelope_d = emg_highpass_butter(abs_values, 150, 2048)
    final_envelope_a = naive_rolling_rms(final_envelope_d, 300)

    return final_envelope_a


def working_pipeline_pre_ml(our_chosen_samples, picker='heart'):
    """
    This is a pipeline to pre-process
    an array of specific fixed dimenstions
    i.e. a three lead array into an EMG singal,
    the function is legacy code, and most
    processsing should be done with
    :code:`multi_lead_type.working_pipeline_pre_ml_multi`
    or :code:`multi_lead_type.working_pipeline_pre_ml_multi`

    :param our_chosen_samples: the read EMG file arrays
    :type our_chosen_samples: ~numpy.ndarray
    :param picker: the picking strategy for independant components
    :type picker: str

    :returns: final_envelope_a
    :rtype: ~numpy.ndarray
    """
    cut_file_data = bad_end_cutter_for_samples(
        our_chosen_samples,
        percent_to_cut=3,
        tolerance_percent=5
    )
    bd_filtered_file_data = emg_bandpass_butter_sample(
        cut_file_data,
        5,
        450,
        2048,
        output='sos'
    )
    # step for end-cutting again to get rid of filtering artifacts
    re_cut_file_data = bad_end_cutter_for_samples(
        bd_filtered_file_data,
        percent_to_cut=3,
        tolerance_percent=5
    )
    #  and do step for ICA
    components = compute_ICA_two_comp(re_cut_file_data)
    #     the picking step!
    if picker == 'peaks':
        emg = pick_more_peaks_array(components)
    elif picker == 'heart':
        emg = pick_lowest_correlation_array(components, re_cut_file_data[0])
    else:
        emg = pick_lowest_correlation_array(components, re_cut_file_data[0])
        print("Please choose an exising picker i.e. peaks or hearts ")
    # now process it in final steps
    abs_values = abs(emg)
    final_envelope_d = emg_highpass_butter(abs_values, 150, 2048)

    return final_envelope_d


def slices_slider(array_sample, slice_len):
    """This function produces continous sequential slices over an
    array of a certain legnth.  The inputs are the following -
    :code:`array_sample`, the signal and :code:`slice_len` - the
    window which you wish to slide with.  The function yields, does
    not return these slices.

    :param array_sample: array containing the signal
    :type array_sample: ~numpy.ndarray
    :param slice_len: the legnth of window on the array
    :type slice_len: int

    :returns: Actually yields, no return
    :rtype: ~numpy.ndarray
    """
    for i in range(len(array_sample) - slice_len + 1):
        yield array_sample[i:i + slice_len]


def slices_jump_slider(array_sample, slice_len, jump):
    """
    This function produces continous sequential slices over an
    array of a certain legnth spaced out by a 'jump'.
    The function yields, does
    not return these slices.

    :param array_sample: array containing the signal
    :type array_sample: ~numpy.ndarray
    :param slice_len: the legnth of window on the array
    :type slice_len: int
    :param jump: the amount by which the winow is moved at iteration
    :type jump: int

    :returns: Actually yields, no return
    :rtype: ~numpy.ndarray

    """
    for i in range(len(array_sample) - (slice_len)):
        yield array_sample[(jump*i):((jump*i) + slice_len)]


def entropical(sig):
    """This function computes a certain type of entropy of a series
    signal array.  Input is sig, the signal, and output is an array of
    entropy measurements. The function can be used inside a generator
    to read over slices.

    :param sig: array containin the signal
    :type sig: ~numpy.ndarray

    :returns: A number expressing the entropy using math.log w/base 2
    :rtype: float

    """
    probabilit = [n_x/len(sig) for x, n_x in collections.Counter(sig).items()]
    e_x = [-p_x*math.log(p_x, 2) for p_x in probabilit]
    return sum(e_x)


def compute_power_loss(
    original_signal,
    original_signal_sampling_frequency,
    processed_signal,
    processed_signal_sampling_frequency
):
    """This function computes the percentage of power loss after the
    processing of a signal. Inputs include the original_signal (signal
    before the processing), :code:`original_signal_sampling_frequency`
    (sampling frequency of the signal before processing),
    :code:`processed_signal` (signal after processing),
    :code:`processed_signal_sampling_frequency` (sampling frequency of
    the signal after processing).

    Output is the percentage of power loss.

    :param original_signal: Array.
    :type  original_signal: ~numpy.ndarray
    :param original_signal_sampling_frequency: Sampling freq. original signal
    :type original_signal_sampling_frequency: int
    :param processed_signal: Array.
    :type  processed_signal: ~numpy.ndarray
    :param processed_signal_sampling_frequency: Sampling frequency processed
        signal
    :type processed_signal_sampling_frequency: int

    :returns: Power loss
    :rtype: float
    """
    nperseg = 1024
    noverlap = 512

    # power spectrum density of the original and
    # processed signals using Welch method
    Pxx_den_orig = signal.welch(  # as per Lu et al. 2009
        original_signal,
        original_signal_sampling_frequency,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    Pxx_den_processed = signal.welch(
        processed_signal,
        processed_signal_sampling_frequency,
        nperseg=nperseg,
        noverlap=noverlap,)
    # compute the percentage of power loss
    power_loss = 100*(1-(np.sum(Pxx_den_processed)/np.sum(Pxx_den_orig)))

    return power_loss


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


def smooth_for_baseline(
    single_filtered_array, start=None, end=None, smooth=100
):
    """
    This is an adaptive smoothing a series that overvalues closer numbers.

    :param single_filtered_array: Array.
    :type single_filtered_array: ~numpy.ndarray
    :param start: The number on samples to work from
    :type start: int
    :param end: The number on samples to work until
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
    :param start: The number on samples to work from
    :type start: int
    :param end: The number on samples to work until
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
        print('Warning: legnth of arrays is not matched')
        longer_signal_len = np.max([len(signal1), len(signal2)])
    else:
        longer_signal_len = len(signal1)

    raw_overlap_percent = sum(
        signal1.astype(int) & signal2.astype(int)
    ) / longer_signal_len
    return raw_overlap_percent


def relative_levenshtein(signal1, signal2):
    """
    Here we take two arrays, and create an edit distance based on Levelshtien
    edit distance The distance is then normalized between 0 and one regardless
    of signal legnth

    """
    signal1_list = []
    signal2_list = []
    for element in signal1:
        signal1_list.append(element)
    for element in signal2:
        signal2_list.append(element)
    distance = textdistance.levenshtein.similarity(signal1_list, signal2_list)
    if len(signal1) != len(signal2):
        print('Warning: legnth of arrays is not matched')
    longer_signal_len = np.max([len(signal1), len(signal2)])
    normalized_distance = distance / longer_signal_len
    return normalized_distance


def full_rolling_rms(x, N):
    """This function computes a root mean squared envelope over an
    array :code:`x`.  To do this it uses number of sample values
    :code:`N`. It differs from :func:`naive_rolling_rms` by that the
    output is the same length as the input vector.

    :param x: Samples from the EMG
    :type x: ~numpy.ndarray
    :param N: Length of the sample use as window for function
    :type N: int

    :returns: The root-mean-squared EMG sample data
    :rtype: ~numpy.ndarray
    """
    x_pad = np.pad(x, (0, N-1), 'constant', constant_values=(0, 0))
    x2 = np.power(x_pad, 2)
    window = np.ones(N)/float(N)
    emg_rms = np.sqrt(np.convolve(x2, window, 'valid'))
    return emg_rms


def gating(
    src_signal,
    gate_peaks,
    gate_width=205,
    method=1,
):
    """
    Eliminate peaks (e.g. QRS) from src_signal using gates
    of width gate_width. The gate either filled by zeros or interpolation.
    The filling method for the gate is encoded as follows:
    0: Filled with zeros
    1: Interpolation samples before and after
    2: Fill with average of prior segment if exists
    otherwise fill with post segment
    3: Fill with running average of RMS (default)

    :param src_signal: Signal to process
    :type src_signalsignal: ~numpy.ndarray
    :param gate_peaks: list of individual peak index places to be gated
    :type gate_peaks: ~list
    :param gate_width: width of the gate
    :type gate_width: int
    :param method: filling method of gate
    :type method: int

    :returns: src_signal_gated, the gated result
    :rtype: ~numpy.ndarray
    """
    src_signal_gated = copy.deepcopy(src_signal)
    max_sample = src_signal_gated.shape[0]
    half_gate_width = gate_width // 2
    if method == 0:
        # Method 0: Fill with zeros
        # TODO: can rewrite with slices from numpy irange to be more efficient
        gate_samples = []
        for i, peak in enumerate(gate_peaks):
            for k in range(
                max(0, peak - half_gate_width),
                min(max_sample, peak + half_gate_width),
            ):
                gate_samples.append(k)

        src_signal_gated[gate_samples] = 0
    elif method == 1:
        # Method 1: Fill with interpolation pre- and post gate sample
        # TODO: rewrite with numpy interpolation for efficiency
        for i, peak in enumerate(gate_peaks):
            pre_ave_emg = src_signal[peak-half_gate_width-1]

            if (peak + half_gate_width + 1) < src_signal_gated.shape[0]:
                post_ave_emg = src_signal[peak+half_gate_width+1]
            else:
                post_ave_emg = 0

            k_start = max(0, peak-half_gate_width)
            k_end = min(
                peak+half_gate_width, src_signal_gated.shape[0]
            )
            for k in range(k_start, k_end):
                frac = (k - peak + half_gate_width)/gate_width
                loup = (1 - frac) * pre_ave_emg + frac * post_ave_emg
                src_signal_gated[k] = loup

    elif method == 2:
        # Method 2: Fill with window length mean over prior section
        # ..._____|_______|_______|XXXXXXX|XXXXXXX|_____...
        #         ^               ^- gate start   ^- gate end
        #         - peak - half_gate_width * 3 (replacer)

        for i, peak in enumerate(gate_peaks):
            start = peak - half_gate_width * 3
            if start < 0:
                start = peak + half_gate_width
            end = start + gate_width
            pre_ave_emg = np.nanmean(src_signal[start:end])

            k_start = max(0, peak - half_gate_width)
            k_end = min(peak + half_gate_width, src_signal_gated.shape[0])
            for k in range(k_start, k_end):
                src_signal_gated[k] = pre_ave_emg

    elif method == 3:
        # Method 3: Fill with moving average over RMS
        gate_samples = []
        for i, peak in enumerate(gate_peaks):
            for k in range(
                max([0, int(peak-gate_width/2)]),
                min([max_sample, int(peak+gate_width/2)])
            ):
                gate_samples.append(k)

        src_signal_gated_base = copy.deepcopy(src_signal_gated)
        src_signal_gated_base[gate_samples] = np.NaN
        src_signal_gated_rms = full_rolling_rms(
            src_signal_gated_base,
            gate_width,)

        for i, peak in enumerate(gate_peaks):
            k_start = max([0, int(peak-gate_width/2)])
            k_end = min([int(peak+gate_width/2), max_sample])

            for k in range(k_start, k_end):
                leftf = max([0, int(k-1.5*gate_width)])
                rightf = min([int(k+1.5*gate_width), max_sample])
                src_signal_gated[k] = np.nanmean(
                    src_signal_gated_rms[leftf:rightf]
                )

    return src_signal_gated


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


def hi_envelope(our_signal, dmax=24):
    """
    Takes a 1d signal array, and extracts 'high'envelope,
    then makes high envelope, based on connecting peaks
    dmax: int, size of chunks,

    :param our_signal: 1d signal array usually of emg
    :type our_signal: ~numpy.ndarray
    :param dmax: legnth of chunk to look for local max in
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


def compute_ICA_two_comp_multi(emg_samples):
    """A function that performs an independant component analysis
    (ICA) meant for EMG data that includes stacked arrays,
    there should be at least two arrays but there can be more.

    :param emg_samples: Original signal array with three or more layers
    :type emg_samples: ~numpy.ndarray

    :returns: Two arrays of independent components (ECG-like and EMG)
    :rtype: ~numpy.ndarray
    """
    all_component_numbers = list(range(emg_samples.shape[0]))
    list_to_c = []
    for i in all_component_numbers:
        list_to_c.append(emg_samples[i])
    X = np.column_stack(list_to_c)
    ica = FastICA(n_components=2, random_state=1)
    S = ica.fit_transform(X)
    component_0 = S.T[0]
    component_1 = S.T[1]
    return component_0, component_1


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
    reccomended to test a smoothing algorithm first, apply,
    then run the area_under the curve with none for smooth_algortihm.
    If a cutoff of the curve before it hits bottom is desired then a value
    other than zero must be in end_curve variable. This variable
    should be written from 0 to 100 for the perfentage of the max value
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
    an array of absolute values for electrophysiological signal
    will be used as the array. Te mid_savgol assumes a parabolic fit.
    It is reccomended to test a smoothing
    algorithm first, apply, then run the find peak algorithm.

    :param array: an array e.g. single lead EMG recording
    :type array: np.array
    :param start_index: which index number the breath starts on
    :type start_index: int
    :param end_index: which index number the breath ends on
    :type end_index: int
    :param smooth_algorithm: algorithm for smoothing (none or 'mid-savgol')
    :type smooth_algorithm: str

    :returns: index of max point, value at max point
    :rtype: tuple
    """
    new_array = array[start_index: (end_index+1)]
    if smooth_algorithm == 'mid_savgol':
        new_array = savgol_filter(
            new_array, int(len(new_array)),
            2,
            deriv=0,
            delta=1.0,
            axis=- 1,
            mode='interp',
            cval=0.0,
        )
        max_ind = (new_array.argmax())
        max_val = new_array[max_ind]
    else:
        max_ind = (new_array.argmax())
        max_val = new_array[max_ind]
    return (max_ind, max_val)


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
        print('Your arrays do not match in legnth, caution!')
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


def helper_lowpass(cutoff, fs, order=5):
    """
    This is a helper function inside the butter_lowpass_filter function.
    """
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def emg_lowpass_butter(array, cutoff, fs, order=5):
    """
    This is a lowpass filter of butterworth design.

    :param array: 1d signal array usually of emg
    :type array: ~numpy.ndarray
    :param cutoff: frequency above which to filter out
    :type cutoff: int
    :param fs: frequency array sampled at in Hertz
    :type fs: int
    :param order: order of the filter
    :type order: int

    :returns: signal_filtered
    :rtype: ~numpy.ndarray
    """
    b, a = helper_lowpass(cutoff, fs, order=order)
    signal_filtered = lfilter(b, a, array)
    return signal_filtered


def find_peaks_in_ecg_signal(ecg_signal, lower_border_percent=50):
    """
    This function assumes you have isolated an ecg-like signal with
    QRS peaks "higher" (or lower) than ST waves.
    In this case it can be applied to return an array of
    ecg peak locations. NB: This function assumes that the ECG
    signal has already been through a bandpass or low-pass filter
    or has little baseline drift.

    :param ecg_signal: frequency array sampled at in Hertz
    :type ecg_signal: ~numpy.ndarray
    :param low_border_percent: percentage max below which no peaks expected
    :type low_border_percent: int

    :returns: tuple first element peak locations, next a dictionary of info
    :rtype: tuple

    """
    ecg_signal = abs(ecg_signal)
    max_peak = ecg_signal.max() - ecg_signal.min()
    set_ecg_peaks = find_peaks(
        ecg_signal,
        prominence=(max_peak*lower_border_percent/100, max_peak)
    )
    return set_ecg_peaks
