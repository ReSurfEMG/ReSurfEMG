"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to filter EMG arrays.

"""

from scipy import signal
from scipy.signal import butter, lfilter
import numpy as np


def emg_bandpass_butter(data_emg, low_pass, high_pass, order=3):
    """The parameter taken in here is the Poly5 file. Output is
    the EMG after a bandpass as made here.

    :param data_emg: Poly5 file with the samples to work over
    :type data_emg: ~TMSiSDK.file_readers.Poly5Reader
    :param low_pass: The number to cut off :code:`frequenciesabove`
    :type low_pass: int
    :param high_pass: The number to cut off :code:`frequenceisbelow`
    :type high_pass: int
    :param order: The filter order
    :type order: int

    :returns: The bandpass filtered EMG sample data
    :rtype: ~numpy.ndarray
    """
    sos = signal.butter(
        order,
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
    fs,
    order=3,
    output='sos'
):
    """Output is the EMG after a bandpass as made here.

    :param data_emg_samp: The array in the sample
    :type data_emg_samp: ~numpy.ndarray
    :param low_pass: The number to cut off :code:`frequenciesabove`
    :type low_pass: int
    :param high_pass: The number to cut off :code:`frequenceisbelow`
    :type high_pass: int
    :param fs: The sample rate i.e. Hertz
    :type fs: int
    :param order: The filter order
    :type order: int
    :param output: The type of sampling stabilizor
    :type high_pass: str

    :returns: The bandpass filtered EMG sample data
    :rtype: ~numpy.ndarray
    """
    sos = signal.butter(
        order,
        [low_pass, high_pass],
        'bandpass',
        fs=fs,
        output='sos',
    )
    # sos (output parameter)is second order section  -> "stabilizes" ?
    emg_filtered = signal.sosfiltfilt(sos, data_emg_samp)
    return emg_filtered


def emg_lowpass_butter_sample(
    data_emg_samp,
    low_pass,
    fs,
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
        fs=fs,
        output='sos',
    )
    emg_filtered = signal.sosfiltfilt(sos, data_emg_samp)
    return emg_filtered


def emg_highpass_butter_sample(
    data_emg_samp,
    high_pass,
    fs,
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
        fs=fs,
        output='sos',
    )
    emg_filtered = signal.sosfiltfilt(sos, data_emg_samp)
    return emg_filtered


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


def emg_highpass_butter(data_emg, cut_above, fs, order=3):
    """The parameter taken in here is the Poly5 file's samples or
    another array.  Output is the EMG after a bandpass as made here.

    :param data_emg: Samples from the EMG
    :type data_emg: ~numpy.ndarray
    :param cut_above: The number to cut off :code:`frequenceisbelow`
    :type cut_above: int
    :param fs: The sample rate i.e. Hertz
    :type fs: int
    :param order: The filter order
    :type order: int

    :returns: The bandpass filtered EMG sample data
    :rtype: ~numpy.ndarray
    """
    sos = signal.butter(
        order,
        cut_above,
        'highpass',
        fs=fs,
        output='sos')
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
