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
       set_percentile=33
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
    rolling_baseline = moving_baseline(emg_env, window_s, 0*emg_sample_rate,
                                       50*emg_sample_rate, emg_sample_rate)

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
    augmented_perc = 25
    perc_window = emg_sample_rate

    y_di_rms = emg_env[int(start_s):int(end_s)]
    s_di = pd.Series(y_di_rms - rolling_baseline)
    seadi_MA = s_di.rolling(window=ma_window, center=True).mean().values
    dseadi_dt = (seadi_MA[1:] - seadi_MA[:-1]) * emg_sample_rate
    seadi_aug = y_di_rms[:-1] + np.abs(dseadi_dt)

    # 2.c. Run the moving median filter over the augmented signal to obtain
    #       the baseline
    slopesum_baseline = np.zeros(
        (len(emg_env[int(start_s):int(end_s)-1]), ))

    for idx in range(0, int(end_s-1)-int(start_s), perc_window):
        start_i = max([0, idx-int(baseline_w_emg)])
        end_i = min([int(end_s-start_s-1), idx+int(baseline_w_emg)])

        baseline_value_emg_di = np.nanpercentile(
            seadi_aug[start_i:end_i], augmented_perc)
        for i in range(idx, min([idx+int(perc_window),
                                 int(end_s-1)-int(start_s)])):
            slopesum_baseline[i] = 1.2 * baseline_value_emg_di

    return (slopesum_baseline, di_baseline_mean,
            di_baseline_std, di_baseline_series)
