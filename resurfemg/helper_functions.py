"""
This file contains functions to work with various EMG arrays
and other types of data arrays e.g. ventilator signals
"""

from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import find_peaks
import collections
import math
from math import log, e
from collections import namedtuple
import builtins


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
    """The paramemter taken in here is the Poly5 file. Output is
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
    herts.  In some other locations the interference is at 60 Hertz.
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


# def cnotch_filter(sample, sample_frequ, freq_to_pull, quality_factor_q):
#     """
#     This is a filter designed to take out a specific frequency.
#     In the EU in some data electrical cords can interfere at
#     around 50 hertzs.
#     In some other locations the interference is at 60 Hertz.
#     The specificities
#     of a local power grid may neccesitate notch filtering. It computes some
#     additional info on the results, and we may change it to return all the
#     information. Pending.


#     """
#     # create notch filter
#     samp_freq = sample_frequ # Sample frequency (Hz)
#     notch_freq = freq_to_pull # Frequency to be removed from signal (Hz)
#     quality_factor = quality_factor_q # Quality factor

#     # design a notch filter using signal.iirnotch
#     b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)

#     # compute magnitude response of the designed filter
#     freq, h = signal.freqz(b_notch, a_notch, fs=samp_freq)
#     # make the output signal
#     output_signal = signal.filtfilt(b_notch, a_notch, sample)
#     return output_signal


def show_my_power_spectrum(sample, sample_rate, upper_window):
    """This function plots a power spectrum of the frequencies
    comtained in an emg based on a fourier transform.  It does not
    return the graph, rather the values but plots the graph before it
    return.  Sample should be one single row (1-dimensional array.)

    :param sample: The sample array
    :type sample: ~numpy.ndarray
    :param sample_rate: Number of samples per second
    :type sample_rate: int
    :param upper_window: The end ofwindow over which values will be plotted
    :type upper_window: int

    :return: :code:`yf, xf` tuple of fourier transformed array and
        frequencies (the values for plotting the power spectrum)
    :rtype: Tuple[float, float]
    """
    N = len(sample)
    # for our emgs samplerate is usually 2048
    yf = fft((sample))
    xf = fftfreq(N, 1 / sample_rate)

    plt.plot(xf, np.abs(yf))
    plt.xlim(0, upper_window)
    plt.show()
    return (yf, xf)


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

    :returns: Binarized array
    :rtype: ~numpy.ndarray
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
    ica = FastICA(n_components=2)
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


def working_pipeline_exp(our_chosen_file):
    """This function produces a filtered respiratory EMG signal from a
    3 lead sEMG file.  The inputs is :code:`our_chosen_file` which we
    give the function as a string of filename.  The output is the
    processed EMG signal filtered and seperated from ecg components.
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
    # step 3 end-cutting again to get rid of filtering artifacts
    re_cut_file_data = bad_end_cutter_for_samples(
        bd_filtered_file_data,
        percent_to_cut=3,
        tolerance_percent=5,
    )
    # skip step4 and do step 5 ICA
    components = compute_ICA_two_comp(re_cut_file_data)
    #     the secret hidden step!
    emg = pick_more_peaks_array(components)
    # now process it in final steps
    abs_values = abs(emg)
    final_envelope_d = emg_highpass_butter(abs_values, 150, 2048)
    final_envelope_a = naive_rolling_rms(final_envelope_d, 300)

    return final_envelope_a


def slices_slider(array_sample, slice_len):
    """This function produces continous sequential slices over an
    array of a certain legnth.  The inputs are the following -
    :code:`array_sample`, the signal and :code:`slice_len` - the
    window which you wish to slide with.  The function yields, does
    not return these slices.
    """
    for i in range(len(array_sample) - slice_len + 1):
        yield array_sample[i:i + slice_len]


def entropical(sig):
    """This function computes a certain type of entropy of a series
    signal array.  Input is sig, the signal, and output is an array of
    entropy measurements.
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

    :return: Array
    :rtype: ~numpy.ndarray
    """
    array = single_filtered_array[start:end]
    dists = np.zeros(len(array))
    # print(len(array), array.max(), array.min())
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

    :return: Array
    :rtype: ~numpy.ndarray
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
    """
    marks = np.logical_xor(array[1:], array[:-1])
    boundaries = np.hstack(
        (np.zeros(1), np.where(marks != 0)[0], np.zeros(1) + len(array) - 1)
    )
    if not array[0]:
        boundaries = boundaries[1:]
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
