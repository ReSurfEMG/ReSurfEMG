"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to determine peak and signal quality from
preprocessed EMG arrays.
"""

import numpy as np
from ..postprocessing.features import time_product, area_under_baseline


def snr_pseudo(
        src_signal,
        peaks,
        baseline,
):
    """
    Approximate the signal-to-noise ratio (SNR) of the signal based
    on the peak height relative to the baseline.

    :param signal: Signal to evaluate
    :type signal: ~numpy.ndarray
    :param peaks: list of individual peak indices
    :type gate_peaks: ~list
    :param baseline: Baseline signal to evaluate SNR to.
    :type baseline: ~numpy.ndarray
    :returns: snr_peaks, the SNR per peak
    :rtype: ~numpy.ndarray
    """

    peak_heights = np.zeros((len(peaks),))
    noise_heights = np.zeros((len(peaks),))

    for peak_nr, idx in enumerate(peaks):
        peak_heights[peak_nr] = src_signal[idx]
        start_i = max([0, idx-2048])
        end_i = min([len(src_signal), idx+2048])
        noise_heights[peak_nr] = np.median(baseline[start_i:end_i])

    snr_peaks = np.divide(peak_heights, noise_heights)
    return snr_peaks


def pocc_quality(
    p_vent_signal,
    pocc_peaks,
    pocc_ends,
    ptp_occs,
    dp_up_10_threshold=0.0,
    dp_up_90_threshold=2.0,
    dp_up_90_norm_threshold=0.8,
):
    """
    Evaluation of occlusion pressure (Pocc) quality, in accordance with Warnaar
    et al. (2024). Poccs are labelled invalid if too many negative deflections
    happen in the upslope (first decile < 0), or if the upslope is to steep
    (high absolute or relative 9th decile), indicating occlusion release before
    the patient's inspiriratory effort has ended.
    :param signal: Airway pressure signal
    :type signal: ~numpy.ndarray
    :param pocc_peaks: list of individual peak indices
    :type pocc_peaks: ~list
    :param pocc_ends: list of individual peak end indices
    :type pocc_ends: ~list
    :param dp_up_10_threshold: Minimum first decile of upslope after the
    (negative) occlusion pressure peak
    :type dp_up_10_threshold: ~float
    :param dp_up_90_threshold: Maximum 9th decile of upslope after the
    (negative) occlusion pressure peak
    :type dp_up_90_threshold: ~float
    :param dp_up_90_norm_threshold: Maximum 9th decile of upslope after the
    (negative) occlusion pressure peak
    :type dp_up_90_norm_threshold: ~float
    :returns: valid_poccs, criteria_matrix
    :rtype: list, ~numpy.ndarray
    """
    dp_up_10 = np.zeros((len(pocc_peaks),))
    dp_up_90 = np.zeros((len(pocc_peaks),))
    dp_up_90_norm = np.zeros((len(pocc_peaks),))
    for idx, pocc_peak in enumerate(pocc_peaks):
        end_i = pocc_ends[idx]
        dp = p_vent_signal[pocc_peak+1:end_i]-p_vent_signal[pocc_peak:end_i-1]
        dp_up_10[idx] = np.percentile(dp, 10)
        dp_up_90[idx] = np.percentile(dp, 90)
        dp_up_90_norm[idx] = dp_up_90[idx] / np.sqrt(ptp_occs[idx])

    criteria_matrix = np.array([
        dp_up_10,
        dp_up_90,
        dp_up_90_norm
    ])
    criteria_bool_matrix = np.array([
        dp_up_10 <= dp_up_10_threshold,
        dp_up_90 > dp_up_90_threshold,
        dp_up_90_norm > dp_up_90_norm_threshold
    ])
    valid_poccs = ~np.any(criteria_bool_matrix, axis=0)
    return valid_poccs, criteria_matrix



def interpeak_dist(ECG_peaks, EMG_peaks, threshold=1.1):
    """
    Calculate the median interpeak distances for ECG and EMG and
    check if their ratio is above the given threshold, i.e. if cardiac
    frequency is higher than respiratory frequency (TRUE)

    :param t_emg: Time points array
    :type t_emg: ~list
    :param ECG_peaks: Indices of ECG peaks
    :type ECG_peaks: ~numpy.ndarray
    :param EMG_peaks: Indices of EMG peaks
    :type EMG_peaks: ~numpy.ndarray
    :param threshold: The threshold value to compare against. Default is 1.1
    :type threshold: ~float
    :returns: valid_interpeak
    :rtype: bool
    """

    # Calculate median interpeak distance for ECG
    t_delta_ecg_med = np.median(np.array(ECG_peaks[1:])
                                - np.array(ECG_peaks[:-1]))
    # # Calculate median interpeak distance for EMG
    t_delta_emg_med = np.median(np.array(EMG_peaks[1:])
                                - np.array(EMG_peaks[:-1]))
    # Check if each median interpeak distance is above the threshold
    t_delta_relative = t_delta_emg_med / t_delta_ecg_med

    valid_interpeak = t_delta_relative > threshold

    return valid_interpeak


def percentage_under_baseline(
    signal,
    fs,
    peaks_s,
    starts_s,
    ends_s,
    baseline,
    aub_window_s=None,
    ref_signal=None,
    aub_threshold=40,
):
    """
    Calculate the percentage area under the baseline, in accordance with
    Warnaar et al. (2024).
    :param signal: signal in which the peaks are detected
    :type signal: ~numpy.ndarray
    :param fs: sampling frequency
    :type fs: ~int
    :param peaks_s: list of individual peak indices
    :type peaks_s: ~list
    :param starts_s: list of individual peak start indices
    :type starts_s: ~list
    :param ends_s: list of individual peak end indices
    :type ends_s: ~list
    :param baseline: running baseline of the signal
    :type baseline: ~numpy.ndarray
    :param aub_window_s: number of samples before and after peaks_s to look for
    the nadir
    :type aub_window_s: ~int
    :param ref_signal: signal in which the nadir is searched
    :type ref_signal: ~numpy.ndarray
    :returns: valid_timeproducts, percentages_aub
    :rtype: list, list
    """
    if aub_window_s is None:
        aub_window_s = 5*fs

    if ref_signal is None:
        ref_signal = signal

    time_products = time_product(
        signal,
        fs,
        starts_s,
        ends_s,
        baseline,
    )
    aubs = area_under_baseline(
        signal,
        fs,
        peaks_s,
        starts_s,
        ends_s,
        aub_window_s,
        baseline,
        ref_signal=signal,
    )

    percentages_aub = aubs / (time_products + aubs) * 100
    valid_timeproducts = percentages_aub < aub_threshold

    return valid_timeproducts, percentages_aub

