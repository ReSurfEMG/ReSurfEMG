"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to filter EMG arrays.

"""

from scipy import signal
from scipy.signal import butter, lfilter
import numpy as np


def emg_bandpass_butter(data_emg, low_pass, high_pass):
    """The parameter taken in here is the Poly5 file. Output is
    the EMG after a bandpass as made here.

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
    # sos (output parameter) is second order section  -> "stabilizes" ?
    emg_filtered = signal.sosfiltfilt(sos, data_emg.samples)
    return emg_filtered


def emg_bandpass_butter_sample(
    data_emg_samp,
    low_pass,
    high_pass,
    sample_rate,
    output='sos'
):
    """Output is the EMG after a bandpass as made here.

    :param data_emg_samp: The array in the sample
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


def emg_lowpass_butter_sample(
    data_emg_samp,
    low_pass,
    sample_rate,
    order=3,
):
    """Output is the EMG after a lowpass as made here.

    :param data_emg_samp: The array in the sample
    :type data_emg_samp: ~numpy.ndarray
    :param low_pass: The number to cut off :code:`frequenciesabove`
    :type low_pass: int
    :param low_pass: The number to cut off :code:`frequenciesabove`
    :type low_pass: int
    :param order: The filter order
    :type order: int

    :returns: The bandpass filtered EMG sample data
    :rtype: ~numpy.ndarray
    """
    sos = signal.butter(
        order,
        [low_pass],
        'lowpass',
        fs=sample_rate,
        output='sos',
    )
    emg_filtered = signal.sosfiltfilt(sos, data_emg_samp)
    return emg_filtered


def emg_highpass_butter_sample(
    data_emg_samp,
    high_pass,
    sample_rate,
    order=3,
):

    """Output is the EMG after a bandpass as made here.

    :param data_emg_samp: The array in the sample
    :type data_emg_samp: ~numpy.ndarray
    :param high_pass: The number to cut off :code:`frequenciesabove`
    :type high_pass: int
    :param order: The filter order
    :type order: int

    :returns: The bandpass filtered EMG sample data
    :rtype: ~numpy.ndarray
    """
    sos = signal.butter(
        order,
        [high_pass],
        'highpass',
        fs=sample_rate,
        output='sos',
    )
    emg_filtered = signal.sosfiltfilt(sos, data_emg_samp)
    return emg_filtered


def bad_end_cutter(data_emg, percent_to_cut=7, tolerance_percent=10):
    """This algorithm takes the end off of EMGs where the end is
    radically altered, or if not radically altered cuts the last 10
    values but returns only the array, not an altered Poly5.

    :param data_emg: A Poly5
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
        if last_half_means != 0:
            difference = abs(last_half_means - last_part_means)/last_half_means
        elif last_part_means == 0:
            difference = 0
        else:
            difference = 1

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
        if last_half_means != 0:
            difference = abs(last_half_means - last_part_means)/last_half_means
        elif last_part_means == 0:
            difference = 0
        else:
            difference = 1

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

    :param data_emg: A Poly5
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
        if last_half_means != 0:
            difference = abs(last_half_means - last_part_means)/last_half_means
        elif last_part_means == 0:
            difference = 0
        else:
            difference = 1

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
    Hertz.  In some other locations the interference is at 60 Hertz.
    The specificities of a local power grid may neccesitate notch
    filtering.

    :param sample: Percentage variation tolerance to allow without cutting
    :type sample: int
    :param sample_frequ: The frequency at which the sample was captured
    :type sample_frequ: int
    :param freq_to_pull: The frequency you desire to remove from the signal
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


def emg_highpass_butter(data_emg, cut_above, sample_rate):
    """The parameter taken in here is the Poly5 file's samples or
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


def helper_lowpass(cutoff, fs, order=5):
    """
    This is a helper function inside the butter_lowpass_filter function.
    """
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def emg_lowpass_butter(array, cutoff, fs, order=5):
    """
    This is a lowpass filter of butterworth design.

    :param array: 1d signal array usually of EMG
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
