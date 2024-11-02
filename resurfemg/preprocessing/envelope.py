"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions extract the envelopes from EMG arrays.
"""

import numpy as np
import pandas as pd


<<<<<<< HEAD
def full_rolling_rms(emg_clean, window_length):
    """This function computes a root mean squared envelope over an
    array cleaned (filtered and ECG eliminated) EMG.
    ---------------------------------------------------------------------------
    :param emg_clean: Samples from the EMG
    :type emg_clean: ~numpy.ndarray
=======
def full_rolling_rms(emg_raw, window_length):
    """This function computes a root mean squared envelope over an
    array :code:`emg_raw`.  To do this it uses number of sample values
    :code:`window_length`.

    :param emg_raw: Samples from the EMG
    :type emg_raw: ~numpy.ndarray
>>>>>>> 34c784f (Release 2 0 0/wavelet denoising (#336))
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns emg_rms: The root-mean-squared EMG sample data
    :rtype emg_rms: ~numpy.ndarray[float]
    """
<<<<<<< HEAD
    emg_clean_sqr = pd.Series(np.power(emg_clean, 2))
    emg_rms = np.sqrt(emg_clean_sqr.rolling(
=======
    emg_raw_sqr = pd.Series(np.power(emg_raw, 2))
    emg_rms = np.sqrt(emg_raw_sqr.rolling(
>>>>>>> 34c784f (Release 2 0 0/wavelet denoising (#336))
        window=window_length,
        min_periods=1,
        center=True).mean()).values

    return emg_rms


<<<<<<< HEAD
def naive_rolling_rms(emg_clean, window_length):
    """This function computes a root mean squared envelope over an
    array `emg_clean`.
    ---------------------------------------------------------------------------
    :param emg_clean: Samples from the EMG
    :type emg_clean: ~numpy.ndarray
=======
def hi_envelope(emg_raw, dmax=24):
    """
    Takes a 1d signal array, and extracts 'high' based on connecting peaks
    dmax: int, size of chunks,

    :param our_signal: 1d signal array usually of emg
    :type our_signal: ~numpy.ndarray
    :param dmax: length of chunk to look for local max in
    :type dmax: int

    :returns: src_signal_gated, the gated result
    :rtype: ~numpy.ndarray
    """
    # local maximum: lmax
    lmax = (np.diff(np.sign(np.diff(emg_raw))) < 0).nonzero()[0] + 1
    lmax = lmax[[
        i+np.argmax(emg_raw[lmax[i:i+dmax]]) for i in range(0, len(lmax), dmax)
    ]]
    smoothed = savgol_filter(emg_raw[lmax], int(0.8 * (len(lmax))), 3)
    emg_high = signal.resample(smoothed, len(emg_raw))

    return emg_high


def naive_rolling_rms(emg_raw, window_length):
    """This function computes a root mean squared envelope over an
    array :code:`emg_raw`. To do this it uses number of sample values
    :code:`window_length`.

    :param emg_raw: Samples from the EMG
    :type emg_raw: ~numpy.ndarray
>>>>>>> 34c784f (Release 2 0 0/wavelet denoising (#336))
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns emg_rms: : The root-mean-squared EMG sample data
    :rtype emg_rms: ~numpy.ndarray[float]
    """
<<<<<<< HEAD
    x_c = np.cumsum(abs(emg_clean)**2)
=======
    x_c = np.cumsum(abs(emg_raw)**2)
>>>>>>> 34c784f (Release 2 0 0/wavelet denoising (#336))
    emg_rms = np.sqrt((x_c[window_length:] - x_c[:-window_length])
                      / window_length)
    return emg_rms


<<<<<<< HEAD
def full_rolling_arv(emg_clean, window_length):
    """This function computes an average rectified value envelope over an
    array `emg_clean`.
    ---------------------------------------------------------------------------
    :param emg_clean: Samples from the EMG
    :type emg_clean: ~numpy.ndarray
=======
def full_rolling_arv(emg_raw, window_length):
    """This function computes an average rectified value envelope over an
    array :code:`emg_raw`.  To do this it uses number of sample values
    :code:`window_length`.

    :param emg_raw: Samples from the EMG
    :type emg_raw: ~numpy.ndarray
>>>>>>> 34c784f (Release 2 0 0/wavelet denoising (#336))
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns emg_arv: The average rectified value EMG sample data
    :rtype emg_arv: ~numpy.ndarray[float]
    """
<<<<<<< HEAD
    emg_clean_abs = pd.Series(np.abs(emg_clean))
    emg_arv = emg_clean_abs.rolling(
=======
    emg_raw_abs = pd.Series(np.abs(emg_raw))
    emg_arv = emg_raw_abs.rolling(
>>>>>>> 34c784f (Release 2 0 0/wavelet denoising (#336))
        window=window_length,
        min_periods=1,
        center=True).mean().values

    return emg_arv
