
"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to perform default procedures.
"""
import resurfemg.preprocessing.ecg_removal as ecg_rm


def ecg_removal_gating(
    emg_raw,
    ecg_peaks_idxs,
    gate_width_samples,
    method=3,
    ecg_shift=None,
):
    """
    Eliminate the ECG peaks from the emg_raw signal.
    :param emg_raw: 1 dimensional emg signal to gate
    :type emg_raw: ~numpy.ndarray
    :param ecg_peaks_idxs: List of ECG peak sample numbers to gate.
    :type ecg_peaks_idxs: ~numpy.ndarray
    :param gate_width_samples: Number of samples to gate
    :type gate_width_samples: int
    :param fs: Sampling rate of emg_raw
    :type fs: int
    :param method: gating method. See the ecg_removal.gating function.
    :type method: int
    :param ecg_shift: Shift gate windows relative to detected peaks in samples.
    :type ecg_shift: int

    :returns: emg_gated
    :rtype: ~numpy.ndarray
    """
    if len(emg_raw.shape) > 1:
        raise ValueError('emg_raw should be a 1-D array')

    if ecg_shift is None:
        ecg_shift = 0

    gate_peaks_idxs = ecg_peaks_idxs + ecg_shift

    # Gate ECG and EMG signal
    # Fill methods: 0: Zeros, 1: Interpolate start-end, 2: Average prior data
    # 3: Moving average
    emg_gated = ecg_rm.gating(
        emg_raw,
        gate_peaks_idxs,
        gate_width=gate_width_samples,
        method=method)

    return emg_gated
