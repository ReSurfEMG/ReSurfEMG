"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to determine peak and signal quality from
preprocessed EMG arrays.
"""

import numpy as np


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
