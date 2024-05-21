"""
Copyright 2024 University of Twente Licensed under the Apache License, version
 2.0. See LICENSE for details.

This file contains functions to calculate moving baselines from a filtered
 EMG envelope. First the moving baseline (from Graßhoff et al. (2021)) and
 then the augmented version of a slope sum baseline
"""
import numpy as np
import pandas as pd


def moving_baseline(
    emg_env,
    window_s,
    step_s,
    set_percentile=33,
):
    """This function calculates a moving baseline from a filtered EMG
        envelope in accordance with Graßhoff et al. (2021)
        :param emg_env: filtered envelope signal of EMG data
        :type emg_env: ~numpy.ndarray
        :param window_s: window length in samples
        :type window_s: int
        :param step_s: number of consecutive samples with the same baseline
        value
        :type step_s: int
        :param: set_percentile
        :type: numpy percentile
        :returns: The rolling baseline for the filtered EMG data
        :rtype: ~numpy.ndarray
        """

    rolling_baseline = np.zeros((len(emg_env), ))

    for idx in range(0, len(emg_env), step_s):
        start_i = max([0, idx-int(window_s/2)])
        end_i = min([len(emg_env), idx + int(window_s/2)])
        baseline_value_y = np.percentile(
            emg_env[start_i:end_i], set_percentile)

        for i in range(idx, min([idx + step_s, len(emg_env)])):
            rolling_baseline[i] = baseline_value_y
    return rolling_baseline


def slopesum_baseline(
    emg_env,
    window_s,
    step_s,
    emg_sample_rate,
    set_percentile=33,
    augm_percentile=25,
    ma_window=None,
    perc_window=None,
):
    """This function calculates the augmented version of the moving baseline
        from a filtered EMG, using a slope sum

        :param emg_env: filtered envelope signal of EMG data
        :type emg_env: ~numpy.ndarray
        :param window_s: window length in seconds
        :type window_s: int
        :param step_s: number of consecutive samples with the same baseline
        value
        :type step_s: int
        :param emg_sample_rate: sample rate from recording
        :type emg_sample_rate: int
        :param set_percentile
        :type set_percentile: float (0-100)
        :param ma_window: moving average window in samples for average dy/dt
        :type ma_window: int
        :param perc_window: number of consecutive samples with the same
        baseline value
        :type perc_window: int
        :returns: The slopesum baseline for the filtered EMG data
        :rtype: ~numpy.ndarray
        """

    if ma_window is None:
        ma_window = emg_sample_rate//2
        print(ma_window)

    if perc_window is None:
        perc_window = emg_sample_rate
        print(perc_window)

    # 1. call the Graßhoff version function for moving baseline
    rolling_baseline = moving_baseline(
        emg_env,
        window_s,
        step_s,
        set_percentile)

    # 2. Calculate the augmented moving baseline for the sEAdi data
    # 2.a. Rolling standard deviation and mean over provided window length
    # baseline_w_emg = int(window_s * emg_sample_rate)  # window length

    y_baseline_series = pd.Series(rolling_baseline)
    y_baseline_std = y_baseline_series.rolling(window_s,
                                               min_periods=1,
                                               center=True).std().values
    y_baseline_mean = y_baseline_series.rolling(window_s,
                                                min_periods=1,
                                                center=True).mean().values

    # 2.b. Augmented signal: EMG + abs([dEMG/dt]_smoothed)
    s_di = pd.Series(emg_env - rolling_baseline)
    y_ma = s_di.rolling(window=ma_window, center=True).mean().values
    dy_dt = (y_ma[1:] - y_ma[:-1]) * emg_sample_rate
    y_aug = emg_env[:-1] + np.abs(dy_dt)

    # 2.c. Run the moving median filter over the augmented signal to obtain
    #       the baseline
    _slopesum_baseline = np.zeros((len(emg_env), ))

    for idx in range(0, len(emg_env), perc_window):
        start_i = max([0, idx-int(window_s)])
        end_i = min([len(emg_env)-1, idx + int(window_s)])

        baseline_value_y = np.nanpercentile(
            y_aug[start_i:end_i], augm_percentile)
        for i in range(idx, min([idx+int(perc_window), len(emg_env)-1])):
            _slopesum_baseline[i] = 1.2 * baseline_value_y

    _slopesum_baseline[i+1] = _slopesum_baseline[i]
    return (_slopesum_baseline, y_baseline_mean,
            y_baseline_std, y_baseline_series)


def onoffpeak_baseline(
    emg_env,
    baseline,
    peak_idxs
):
    """This function calculates the peaks of each breath using the
    slopesum baseline from a filtered EMG

    :param emg_env: filtered envelope signal of EMG data
    :type emg_env: ~numpy.ndarray
    :param baseline: baseline signal of EMG data for baseline detection
    :type baseline: ~numpy.ndarray
    :param peak_idxs: list of peak indices for which to find on- and offset
    :type peak_idxs: ~numpy.ndarray
    :returns: peak_idxs, peak_start_idxs, peak_end_idxs
    :rtype: list
    """

    # Detect the sEAdi on- and offsets
    baseline_crossings_idx = np.nonzero(
        np.diff(np.sign(emg_env - baseline)) != 0)[0]

    peak_start_idxs = np.zeros((len(peak_idxs),), dtype=int)
    peak_end_idxs = np.zeros((len(peak_idxs),), dtype=int)
    for peak_nr, peak_idx in enumerate(peak_idxs):
        delta_samples = peak_idx - baseline_crossings_idx[
            baseline_crossings_idx < peak_idx]
        if len(delta_samples) < 1:
            peak_start_idxs[peak_nr] = 0
            peak_end_idxs[peak_nr] = baseline_crossings_idx[
                baseline_crossings_idx > peak_idx][0]
        else:
            a = np.argmin(delta_samples)

            peak_start_idxs[peak_nr] = int(baseline_crossings_idx[a])
            if a < len(baseline_crossings_idx) - 1:
                peak_end_idxs[peak_nr] = int(baseline_crossings_idx[a+1])
            else:
                peak_end_idxs[peak_nr] = len(emg_env) - 1

    return (peak_idxs, peak_start_idxs, peak_end_idxs)
