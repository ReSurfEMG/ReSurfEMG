"""
Copyright 2024 University of Twente Licensed under the Apache License, version
 2.0. See LICENSE for details.

This file contains functions to calculate moving baselines from a filtered
 EMG envelope. First the moving baseline (from Graßhoff et al. (2021)) and
 then the augmented version of a slope sum baseline
"""
import numpy as np
import pandas as pd
import scipy


def moving_baseline(
       emg_env,
       window_s,
       start_s,
       end_s,
       emg_sample_rate,
       set_percentile=33,
):
    """This function calculates a moving baseline from a filtered EMG
        envelope in accordance with Graßhoff et al. (2021)
        :param emg_env: filtered envelope signal of EMG data
        :type emg_env: ~numpy.ndarray
        :param window_s: window length in samples
        :type window_s: int
        :param start_s: start sample of selected window
        :type start_s: int
        :param end_s: end sample of selected window
        :type end_s: int
        :param emg_sample_rate: sample rate from recording
        :type emg_sample_rate: int
        :param: set_percentile
        :type: numpy percentile

        :returns: The rolling baseline for the filtered EMG data
        :rtype: ~numpy.ndarray
        """

    baseline_w_emg = int(window_s*emg_sample_rate)  # window length

    rolling_baseline = np.zeros(
        (len(emg_env[int(start_s):int(end_s)]), ))

    for idx in range(0, int(end_s)-int(start_s), int(emg_sample_rate/5)):
        start_i = max([int(start_s), int(start_s)+idx-int(baseline_w_emg/2)])
        end_i = min([int(end_s), int(start_s)+idx+int(baseline_w_emg/2)])
        baseline_value_emg_di = np.percentile(
            emg_env[start_i:end_i], set_percentile)

        for i in range(idx, min([idx+int(emg_sample_rate/5),
                                 int(end_s) - int(start_s)])):
            rolling_baseline[i] = baseline_value_emg_di
    return rolling_baseline


def slopesum_baseline(
       emg_env,
       window_s,
       start_s,
       end_s,
       emg_sample_rate,
       set_percentile=33,
       augm_percentile=25
):
    """This function calculates the augmented version of the moving baseline
        from a filtered EMG, using a slope sum

        :param emg_env: filtered envelope signal of EMG data
        :type emg_env: ~numpy.ndarray
        :param window_s: window length in seconds
        :type window_s: int
        :param start_s: start sample of selected window
        :type start_s: int
        :param end_s: end sample of selected window
        :type end_s: int
        :param emg_sample_rate: sample rate from recording
        :type emg_sample_rate: int
        :param: set_percentile
        :type: numpy percentile

        :returns: The slopesum baseline for the filtered EMG data
        :rtype: ~numpy.ndarray
        """
    # 1. call the Graßhoff version function for moving baseline
    rolling_baseline = moving_baseline(
        emg_env, window_s, start_s, end_s, emg_sample_rate, set_percentile)

    # 2. Calculate the augmented moving baseline for the sEAdi data
    # 2.a. Rolling standard deviation and mean over provided window length
    baseline_w_emg = int(window_s * emg_sample_rate)  # window length

    di_baseline_series = pd.Series(rolling_baseline)
    di_baseline_std = di_baseline_series.rolling(baseline_w_emg,
                                                 min_periods=1,
                                                 center=True).std().values
    di_baseline_mean = di_baseline_series.rolling(baseline_w_emg,
                                                  min_periods=1,
                                                  center=True).mean().values

    # 2.b. Augmented signal: EMG + abs([dEMG/dt]_smoothed)
    ma_window = emg_sample_rate//2
    perc_window = emg_sample_rate

    y_di_rms = emg_env[int(start_s):int(end_s)]
    s_di = pd.Series(y_di_rms - rolling_baseline)
    seadi_MA = s_di.rolling(window=ma_window, center=True).mean().values
    dseadi_dt = (seadi_MA[1:] - seadi_MA[:-1]) * emg_sample_rate
    seadi_aug = y_di_rms[:-1] + np.abs(dseadi_dt)

    # 2.c. Run the moving median filter over the augmented signal to obtain
    #       the baseline
    slopesum_baseline = np.zeros(
        (len(emg_env[int(start_s):int(end_s)]), ))

    for idx in range(0, int(end_s-1)-int(start_s), perc_window):
        start_i = max([0, idx-int(baseline_w_emg)])
        end_i = min([int(end_s-start_s-1), idx+int(baseline_w_emg)])

        baseline_value_emg_di = np.nanpercentile(
            seadi_aug[start_i:end_i], augm_percentile)
        for i in range(idx, min([idx+int(perc_window),
                                 int(end_s-1)-int(start_s)])):
            slopesum_baseline[i] = 1.2 * baseline_value_emg_di

    return (slopesum_baseline, di_baseline_mean,
            di_baseline_std, di_baseline_series)


def onoffpeak_baseline(
        emg_env,
        slopesum_baseline,
        start_s,
        end_s,
        emg_sample_rate,
        prominence_f=0.5
):
    """This function calculates the peaks of each breath using the
    slopesum baseline from a filtered EMG

    :param emg_env: filtered envelope signal of EMG data
    :type emg_env: ~numpy.ndarray
    :param slopesum_baseline: baseline signal of EMG data
    :type slopesum_baseline: ~numpy.ndarray
    :param start_sec: start sample of selected window
    :type start_sec: int
    :param end_sec: end sample of selected window
    :type end_sec: int
    :param emg_sample_rate: sample rate from recording
    :type emg_sample_rate: int
    :param: prominence_default
    :type: int

        :returns: eadi_peak, eadi_start, eadi_end
        :rtype: list
        """
    # EMG peak detection parameters:
    # sEAdi_prominence_factor_default = 0.5
    # Threshold peak height as fraction of max peak height
    # prominence_factor = 0.5 (default value)
    # prominence_factor = prominence_default
    emg_peak_width = 0.2

    # Find diaphragm sEAdi peaks and baseline crossings using the new baseline

    y_di_RMS = emg_env[int(start_s):(int(end_s))]
    slopesum_baseline = slopesum_baseline[int(start_s):(int(end_s))]
    treshold = 0
    width = int(emg_peak_width * emg_sample_rate)
    prominence = prominence_f * \
        (np.nanpercentile(y_di_RMS - slopesum_baseline, 75)
            + np.nanpercentile(y_di_RMS - slopesum_baseline, 50))
    EMG_peaks_di, properties = scipy.signal.find_peaks(
        y_di_RMS, height=treshold, prominence=prominence, width=width)

    # Detect the sEAdi on- and offsets
    baseline_crossings_idx = np.argwhere(
        np.diff(np.sign(y_di_RMS - slopesum_baseline)) != 0)

    eadi_start = []
    eadi_end = []
    for idx in range(len(baseline_crossings_idx)-1):
        if emg_env[(baseline_crossings_idx[idx])+1
                   ] > emg_env[baseline_crossings_idx[idx]]:
            eadi_start.append(int(baseline_crossings_idx[idx]))
        else:
            eadi_end.append(int(baseline_crossings_idx[idx]))

    eadi_peak = []

    if eadi_end[0] < eadi_start[0]:
        del eadi_end[0]

    for start_index, end_index in zip(eadi_start, eadi_end):
        # Extract section between start and end indices
        section = emg_env[start_index:end_index + 1]
        # Find the index of the peak value within the section
        peak_index_relative = np.argmax(section)
        # Calculate the absolute index of the peak value
        peak_index_absolute = start_index + peak_index_relative
        # Append the peak index to EAdi_peak list
        eadi_peak.append(peak_index_absolute)
    return (eadi_peak, eadi_start, eadi_end)
