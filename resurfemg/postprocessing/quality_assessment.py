"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to determine peak and signal quality from
preprocessed EMG arrays.
"""

import numpy as np
from scipy.optimize import curve_fit
import resurfemg.helper_functions.helper_functions as hf
import resurfemg.postprocessing.features as feat


def snr_pseudo(
    src_signal,
    peaks,
    baseline,
    fs,
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
    :param fs: sampling rate
    :type fs: int
    :returns: snr_peaks, the SNR per peak
    :rtype: ~numpy.ndarray
    """

    peak_heights = np.zeros((len(peaks),))
    noise_heights = np.zeros((len(peaks),))

    for peak_nr, idx in enumerate(peaks):
        peak_heights[peak_nr] = src_signal[idx]
        start_i = max([0, idx - fs])
        end_i = min([len(src_signal), idx + fs])
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

    valid_interpeak = t_delta_relative >= threshold

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
    :param aub_threshold: maximum aub error percentage for a peak
    :type aub_threshold: ~float
    :returns: valid_timeproducts, percentages_aub
    :rtype: list, list
    """
    if aub_window_s is None:
        aub_window_s = 5*fs

    if ref_signal is None:
        ref_signal = signal

    time_products = feat.time_product(
        signal,
        fs,
        starts_s,
        ends_s,
        baseline,
    )
    aubs = feat.area_under_baseline(
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
    valid_timeproducts = percentages_aub <= aub_threshold

    return valid_timeproducts, percentages_aub


def detect_non_consecutive_manoeuvres(
    ventilator_breath_idxs,
    manoeuvres_idxs
):

    """
    Detect manoeuvres (for example Pocc) with no supported breaths
    in between. Input are the ventilator breaths, to be detected with the
    function event_detecton.detect_supported_breaths(...)
    If no supported breaths are detected in between two manoeuvres,
    valid_manoeuvres is 'true'
    Note: fs of both signals should be equal.

    :param ventilator_breath_idxs: list of supported breath indices
    :type ventilator_breath_idxs: ~list
    :param manoeuvres_idxs : list of manoeuvres indices
    :type manoeuvres_idxs: ~list

    :returns: valid_manoeuvres
    :return type: list
    """

    consecutive_manoeuvres = np.zeros(len(manoeuvres_idxs), dtype=bool)
    for idx, _ in enumerate(manoeuvres_idxs):
        if idx > 0:
            # Check for supported breaths in between two Poccs
            intermediate_breaths = np.equal(
                (manoeuvres_idxs[idx-1] < ventilator_breath_idxs),
                (ventilator_breath_idxs < manoeuvres_idxs[idx]))

            # If no supported breaths are detected in between, a
            # 'double dip' is detected
            intermediate_breath_count = np.sum(intermediate_breaths)
            if intermediate_breath_count > 0:
                consecutive_manoeuvres[idx] = False
            else:
                consecutive_manoeuvres[idx] = True
        else:
            consecutive_manoeuvres[idx] = False

    valid_manoeuvres = np.logical_not(consecutive_manoeuvres)

    return valid_manoeuvres


def evaluate_bell_curve_error(
    peaks_s,
    starts_s,
    ends_s,
    signal,
    fs,
    time_products,
    bell_window_s=None,
    bell_threshold=40,
):

    """This function calculates the bell-curve error of signal peaks, and pro

    :param signal: filtered signal
    :type signal: ~numpy.ndarray
    :param peaks_s: list of peak indices
    :type peaks_s: ~numpy.ndarray
    :param starts_s: list of onsets indices
    :type starts_s: ~numpy.ndarray
    :param ends_s: list of offsets indices
    :type ends_s: ~numpy.ndarray
    :param fs: sample rate
    :type fs: int
    :param time_products: list of area under the curves per peak
    :type time_products: ~numpy.ndarray
    :param bell_window_s: number of samples before and after peaks_s to look
    for the nadir
    :type bell_window_s: ~int
    :param bell_threshold: maximum bell error percentage for a valid peak
    :type bell_threshold: ~float
    :returns: bell_error
    :rtype: ~numpy.ndarray
    """
    if bell_window_s is None:
        bell_window_s = fs * 5
    t = np.array([i / fs for i in range(len(signal))])

    bell_error = np.zeros((len(peaks_s),))
    percentage_bell_error = np.zeros((len(peaks_s),))
    fitted_parameters = np.zeros((len(peaks_s), 3))
    y_min = np.zeros((len(peaks_s),))
    for idx, (peak_s, start_i, end_i, tp) in enumerate(
            zip(peaks_s, starts_s, ends_s, time_products)):
        baseline_start_i = max(0, peak_s - bell_window_s)
        baseline_end_i = min(len(signal) - 1, peak_s + bell_window_s)
        y_min[idx] = np.min(signal[baseline_start_i:baseline_end_i])

        if end_i - start_i < 3:
            plus_idx = 3 - (end_i - start_i)
        else:
            plus_idx = 0

        x_data = t[start_i:end_i + 1 + plus_idx]
        y_data = signal[start_i:end_i + 1 + plus_idx] - y_min[idx]

        if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)) or np.any(
                np.isinf(x_data)) or np.any(np.isinf(y_data)):
            print(f"NaNs or Infs detected in data for peak index {idx}. "
                  + "Skipping this peak.")
            bell_error[idx] = np.nan
            continue

        try:
            popt, *_ = curve_fit(
                hf.bell_curve, x_data, y_data,
                bounds=([0., t[peak_s] - 0.5, 0.],
                        [60., t[peak_s] + 0.5, 0.5])
            )
        except RuntimeError as e:
            print(f"Curve fitting failed for peak index {idx} with error: {e}")
            bell_error[idx] = np.nan
            popt = np.array([np.nan, np.nan, np.nan])
            continue

        bell_error[idx] = np.trapz(
            np.abs((signal[start_i:end_i + 1] - (
                hf.bell_curve(t[start_i:end_i + 1], *popt) + y_min[idx]))),
            dx=1 / fs
        )
        percentage_bell_error[idx] = bell_error[idx] / tp * 100
        fitted_parameters[idx, :] = (popt)

    valid_peak = percentage_bell_error <= bell_threshold

    return (valid_peak, bell_error, percentage_bell_error, y_min,
            fitted_parameters)
