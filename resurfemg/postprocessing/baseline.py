"""
Copyright 2024 University of Twente Licensed under the Apache License, version
 2.0. See LICENSE for details.

This file contains functions to calculate moving baselines from a filtered
 EMG envelope. First the moving baseline (from Graßhoff et al. (2021)) and
 then the augmented version of a slope sum baseline
"""
import numpy as np
import pandas as pd
from ..helper_functions.helper_functions import derivative


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
    fs,
    set_percentile=33,
    augm_percentile=25,
    ma_window=None,
    perc_window=None,
):
    """
    This function calculates the augmented version of the moving baseline from
    a filtered EMG, using a slope sum.
    :param emg_env: filtered envelope signal of EMG data
    :type emg_env: ~numpy.ndarray
    :param window_s: window length in seconds
    :type window_s: int
    :param step_s: number of consecutive samples with the same baseline value
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

        ma_window = fs//2

    if perc_window is None:
        perc_window = fs

    # 1. call the Graßhoff version function for moving baseline
    rolling_baseline = moving_baseline(
        emg_env,
        window_s,
        step_s,
        set_percentile)

    # 2. Calculate the augmented moving baseline for the sEAdi data
    # 2.a. Rolling standard deviation and mean over provided window length
    y_baseline_series = pd.Series(rolling_baseline)
    y_baseline_std = y_baseline_series.rolling(window_s,
                                               min_periods=1,
                                               center=True).std().values
    y_baseline_mean = y_baseline_series.rolling(window_s,
                                                min_periods=1,
                                                center=True).mean().values

    # 2.b. Augmented signal: EMG + abs([dEMG/dt]_smoothed)
    dy_dt = derivative(emg_env - rolling_baseline, fs, ma_window)
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
