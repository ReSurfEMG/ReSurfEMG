"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to extract detect peak, on- and offset samples.
"""
import numpy as np
import pandas as pd


def onoffpeak_baseline_crossing(
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


