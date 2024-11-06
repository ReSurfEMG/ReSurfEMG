"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions extract the envelopes from EMG arrays.
"""

import numpy as np
import pandas as pd


def full_rolling_rms(emg_clean, window_length):
    """This function computes a root mean squared envelope over an
    array cleaned (filtered and ECG eliminated) EMG.
    ---------------------------------------------------------------------------
    :param emg_clean: Samples from the EMG
    :type emg_clean: ~numpy.ndarray
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns emg_rms: The root-mean-squared EMG sample data
    :rtype emg_rms: ~numpy.ndarray[float]
    """
    emg_clean_sqr = pd.Series(np.power(emg_clean, 2))
    emg_rms = np.sqrt(emg_clean_sqr.rolling(
        window=window_length,
        min_periods=1,
        center=True).mean()).values

    return emg_rms


def naive_rolling_rms(emg_clean, window_length):
    """This function computes a root mean squared envelope over an
    array `emg_clean`.
    ---------------------------------------------------------------------------
    :param emg_clean: Samples from the EMG
    :type emg_clean: ~numpy.ndarray
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns emg_rms: : The root-mean-squared EMG sample data
    :rtype emg_rms: ~numpy.ndarray[float]
    """
    x_c = np.cumsum(abs(emg_clean)**2)
    emg_rms = np.sqrt((x_c[window_length:] - x_c[:-window_length])
                      / window_length)
    return emg_rms


def full_rolling_arv(emg_clean, window_length):
    """This function computes an average rectified value envelope over an
    array `emg_clean`.
    ---------------------------------------------------------------------------
    :param emg_clean: Samples from the EMG
    :type emg_clean: ~numpy.ndarray
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns emg_arv: The average rectified value EMG sample data
    :rtype emg_arv: ~numpy.ndarray[float]
    """
    emg_clean_abs = pd.Series(np.abs(emg_clean))
    emg_arv = emg_clean_abs.rolling(
        window=window_length,
        min_periods=1,
        center=True).mean().values

    return emg_arv
