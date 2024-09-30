"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to eliminate ECG artifacts from
 with various EMG arrays.
"""

import copy
import numpy as np
import scipy
import pywt
import pandas as pd
from sklearn.decomposition import FastICA
from scipy.signal import find_peaks

from resurfemg.preprocessing import envelope as evl
import resurfemg.preprocessing.filtering as filt


def compute_ica_two_comp(emg_raw):
    """A function that performs an independent component analysis
    (ICA) meant for EMG data that includes three stacked arrays.

    :param emg_raw: Original signal array with three layers
    :type emg_raw: ~numpy.ndarray

    :returns: Two arrays of independent components (ECG-like and EMG)
    :rtype: ~numpy.ndarray
    """
    X = np.c_[emg_raw[0], emg_raw[2]]
    ica = FastICA(n_components=2, random_state=1)
    S = ica.fit_transform(X)
    component_0 = S.T[0]
    component_1 = S.T[1]
    return component_0, component_1


def compute_ica_two_comp_multi(emg_raw):
    """A function that performs an independant component analysis
    (ICA) meant for EMG data that includes stacked arrays,
    there should be at least two arrays but there can be more.

    :param emg_raw: Original signal array with three or more layers
    :type emg_raw: ~numpy.ndarray

    :returns: Two arrays of independent components (ECG-like and EMG)
    :rtype: ~numpy.ndarray
    """
    all_component_numbers = list(range(emg_raw.shape[0]))
    list_to_c = []
    for i in all_component_numbers:
        list_to_c.append(emg_raw[i])
    X = np.column_stack(list_to_c)
    ica = FastICA(n_components=2, random_state=1)
    S = ica.fit_transform(X)
    component_0 = S.T[0]
    component_1 = S.T[1]
    return component_0, component_1


def compute_ICA_two_comp_selective(
    emg_raw,
    use_all_leads=True,
    desired_leads=(0, 2),
):
    """A function that performs an independant component analysis
    (ICA) meant for EMG data that includes stacked arrays,
    there should be at least two arrays but there can be more.

    :param emg_raw: Original signal array with three or more layers
    :type emg_raw: ~numpy.ndarray
    :param use_all_leads: True if all leads used, otherwise specify leads
    :type use_all_leads: bool
    :param desired_leads: tuple of leads to use starting from 0
    :type desired_leads: tuple

    :returns: Two arrays of independent components (ECG-like and EMG)
    :rtype: ~numpy.ndarray
    """
    if use_all_leads:
        all_component_numbers = list(range(emg_raw.shape[0]))
    else:
        all_component_numbers = desired_leads
        diff = set(all_component_numbers) - set(range(emg_raw.shape[0]))
        if diff:
            raise IndexError(
                "You picked nonexistant leads {}, "
                "please see documentation".format(diff)
            )
    list_to_c = []
    # TODO (makeda): change to list comprehension on refactoring
    for i in all_component_numbers:
        list_to_c.append(emg_raw[i])
    X = np.column_stack(list_to_c)
    ica = FastICA(n_components=2, random_state=1)
    S = ica.fit_transform(X)
    component_0 = S.T[0]
    component_1 = S.T[1]
    return component_0, component_1


def compute_ICA_n_comp(
    emg_raw,
    use_all_leads=True,
    desired_leads=(0, 2),
):
    """A function that performs an independant component analysis
    (ICA) meant for EMG data that includes stacked arrays,
    there should be at least two arrays but there can be more.
    This differs from helper_functions.compute_ICA_two_comp_multi
    because you can get n leads back instead of only two.

    :param emg_raw: Original signal array with three or more layers
    :type emg_raw: ~numpy.ndarray
    :param use_all_leads: True if all leads used, otherwise specify leads
    :type use_all_leads: bool
    :param desired_leads: tuple of leads to use starting from 0
    :type desired_leads: tuple

    :returns: Arrays of independent components (ECG-like and EMG)
    :rtype: ~numpy.ndarray
    """
    if use_all_leads:
        all_component_numbers = list(range(emg_raw.shape[0]))
        n_components = len(all_component_numbers)
    else:
        all_component_numbers = desired_leads
        n_components = len(all_component_numbers)
        diff = set(all_component_numbers) - set(range(emg_raw.shape[0]))
        if diff:
            raise IndexError(
                "You picked nonexistant leads {}, "
                "please see documentation".format(diff)
            )
    list_to_c = []
    # TODO (makeda): change to list comprehension on refactoring
    for i in all_component_numbers:
        list_to_c.append(emg_raw[i])
    X = np.column_stack(list_to_c)
    ica = FastICA(n_components, random_state=1)
    S = ica.fit_transform(X)
    answer = S.T
    return answer


def pick_more_peaks_array(components_tuple):
    """Here we have a function that takes a tuple with the two parts
    of ICA, and finds the one with more peaks and anti-peaks.  The EMG
    if without a final envelope will have more peaks

    .. note::
        Data should not have been finally filtered to envelope level

    :param components_tuple: tuple of two arrays representing different signals
    :type components_tuple: Tuple[~numpy.ndarray, ~numpy.ndarray]

    :return: Array with more peaks (should usually be the EMG as
        opposed to ECG)
    :rtype: ~numpy.ndarray
    """
    component_0 = components_tuple[0]
    component_1 = components_tuple[1]
    low_border_c0 = (component_0.max() - component_0.mean())/4
    peaks0, _0 = find_peaks(component_0, height=low_border_c0, distance=10)
    antipeaks0, _ = find_peaks(
        (component_0*(-1)),
        height=-low_border_c0,
        distance=10)
    low_border_c1 = (component_1.max() - component_1.mean())/4
    peaks1, _1 = find_peaks(component_1, height=low_border_c1, distance=10)
    antipeaks1, _ = find_peaks(
        (component_1*(-1)),
        height=-low_border_c1,
        distance=10,
    )

    sum_peaks_0 = len(peaks0) + len(antipeaks0)
    sum_peaks_1 = len(peaks1) + len(antipeaks1)

    if sum_peaks_0 > sum_peaks_1:
        emg_component = components_tuple[0]
    elif sum_peaks_1 > sum_peaks_0:
        emg_component = components_tuple[1]
    else:
        print("this is very strange data, please examine by hand")
    return emg_component


def pick_highest_correlation_array_multi(components, ecg_lead):
    """Here we have a function that takes a tuple with n parts
    of ICA and the array defined by the user as the ECG recording,
    and finds the ICA component with the highest similarity to the ECG.
    Data should not have been finally filtered to envelope level

    :param components: n-dimensional array representing different components.
        Each row is a component.
    :type components: ~numpy.ndarray
    :param ecg_lead: array containing the ECG recording
    :type ecg_lead: numpy.ndarray

    :returns: Index of the array with the highest correlation coefficient
     to the ECG lead (should usually be the  ECG)
    :rtype: int
    """

    corr_tuple = np.vstack((ecg_lead, components))
    corr_matrix = abs(np.corrcoef(corr_tuple))

    # get the component with the highest correlation to ECG
    # the matriz is symmetric, so we can check just the first row
    # the first coefficient is the autocorrelation of the ECG lead,
    # so we can check the other rows

    hi_index = np.argmax(corr_matrix[0][1:])
    return hi_index


def compute_ICA_n_comp_selective_zeroing(
    emg_raw,
    ecg_lead_to_remove,
    use_all_leads=True,
    desired_leads=(0, 2),
):
    """A function that performs an independant component analysis
    (ICA) meant for EMG data that includes stacked arrays,
    there should be at least two arrays but there can be more.
    In this ICA one lead is put to zero before reconstruction.
    This should probably be the ECG lead.

    :param emg_raw: Original signal array with three or more layers
    :type emg_raw: ~numpy.ndarray
    :param ecg_lead_to_remove: Lead number counting from zero to get rid of
    :type ecg_lead_to_remove: int
    :param use_all_leads: True if all leads used, otherwise specify leads
    :type use_all_leads: bool
    :param desired_leads: tuple of leads to use starting from 0
    :type desired_leads: tuple

    :returns: Arrays of independent components (ECG-like and EMG)
    :rtype: ~numpy.ndarray
    """
    if use_all_leads:
        all_component_numbers = list(range(emg_raw.shape[0]))
        n_components = len(all_component_numbers)
    else:
        all_component_numbers = desired_leads
        n_components = len(all_component_numbers)
        diff = set(all_component_numbers) - set(range(emg_raw.shape[0]))
        if diff:
            raise IndexError(
                "You picked nonexistant leads {}, "
                "please see documentation".format(diff)
            )
    list_to_c = []
    # TODO (makeda): change to list comprehension on refactoring
    for i in all_component_numbers:
        list_to_c.append(emg_raw[i])

    X = np.column_stack(list_to_c)
    ica = FastICA(n_components, random_state=1)
    S = ica.fit_transform(X)
    S_copy = copy(S)

    hi_index = pick_highest_correlation_array_multi(
        S_copy.transpose(),
        emg_raw[ecg_lead_to_remove])

    S_copy.T[hi_index] = np.zeros(len(S_copy.T[hi_index]))

    reconstructed = ica.inverse_transform(S_copy)
    reconstructed = reconstructed.T

    return reconstructed


def pick_lowest_correlation_array(components_tuple, ecg_lead):
    """Here we have a function that takes a tuple with the two parts
    of ICA and the array containing the ECG recording, and finds the
    ICA component with the lowest similarity to the ECG.
    Data should not have been finally filtered to envelope level

    :param components_tuple: tuple of two arrays representing different signals
    :type components_tuple: Tuple[~numpy.ndarray, ~numpy.ndarray]

    :param ecg_lead: array containing the ECG recording
    :type ecg_lead: numpy.ndarray

    :returns: Array with the lowest correlation coefficient
     to the ECG lead (should usually be the EMG as opposed to ECG)
    :rtype: ~numpy.ndarray
    """
    component_0 = components_tuple[0]
    component_1 = components_tuple[1]

    # create a tuple containing the data, each row is a variable,
    # each column is an observation

    corr_tuple = np.vstack((ecg_lead, component_0, component_1))

    # compute the correlation matrix
    # the absolute value is used, because the ICA decomposition might
    # produce a component with negative peaks. In this case
    # the signals will be maximally negatively correlated

    corr_matrix = abs(np.corrcoef(corr_tuple))

    # get the component with the lowest correlation to ECG
    # the matrix is symmetric, so we can check just the first row
    # the first coefficient is the autocorrelation of the ECG lead,
    # so we can check row 1 and 2

    lowest_index = np.argmin(corr_matrix[0][1:])
    emg_component = components_tuple[lowest_index]

    return emg_component


def pick_highest_correlation_array(components_tuple, ecg_lead):
    """Here we have a function that takes a tuple with the two parts
    of ICA and the array containing the ECG recording, and finds the
    ICA component with the highest similarity to the ECG.
    Data should not have been finally filtered to envelope level

    :param components_tuple: tuple of two arrays representing different signals
    :type components_tuple: Tuple[~numpy.ndarray, ~numpy.ndarray]

    :param ecg_lead: array containing the ECG recording
    :type ecg_lead: numpy.ndarray

    :returns: Array with the highest correlation coefficient
     to the ECG lead (should usually be the  ECG)
    :rtype: ~numpy.ndarray
    """
    component_0 = components_tuple[0]
    component_1 = components_tuple[1]
    corr_tuple = np.vstack((ecg_lead, component_0, component_1))
    corr_matrix = abs(np.corrcoef(corr_tuple))

    # get the component with the highest correlation to ECG
    # the matrix is symmetric, so we can check just the first row
    # the first coefficient is the autocorrelation of the ECG lead,
    # so we can check row 1 and 2

    hi_index = np.argmax(corr_matrix[0][1:])
    ecg_component = components_tuple[hi_index]

    return ecg_component


def detect_ecg_peaks(
    ecg_raw,
    fs,
    peak_fraction=0.3,
    peak_width_s=None,
    peak_distance=None,
    bp_filter=True,
):
    """
    Detect ECG peaks in EMG signal.
    :param ecg_raw: ecg signals to detect the ECG peaks in.
    :type ecg_raw: ~numpy.ndarray
    :param emg_raw: emg signals to gate
    :type emg_raw: ~numpy.ndarray
    :param fs: Sampling rate of the emg signals.
    :type fs: int
    :param peak_fraction: ECG peaks amplitude threshold relative to the
    specified fraction of the min-max values in the ECG signal
    :type peak_fraction: float
    :param peak_width_s: ECG peaks width threshold in samples.
    :type peak_width_s: int
    :param peak_distance: Minimum time between ECG peaks in samples.
    :type peak_distance: int
    :param filter: Bandpass filter the ecg_raw between 1-500 Hz before peak
    detection.
    :type filter: bool

    :returns: ecg_peak_idxs
    :rtype: ~numpy.ndarray
    """

    if peak_width_s is None:
        peak_width_s = fs // 1000

    if peak_distance is None:
        peak_distance = fs // 3

    if bp_filter:
        lp_cf = min([500, fs//2])
        ecg_filt = filt.emg_bandpass_butter_sample(
            ecg_raw, 1, lp_cf, fs, output='sos')
        ecg_rms = evl.full_rolling_rms(ecg_filt, fs // 200)
    else:
        ecg_rms = evl.full_rolling_rms(ecg_raw, fs // 200)
    max_ecg_rms = max(ecg_rms)
    min_ecg_rms = min(ecg_rms)
    peak_height = peak_fraction * (max_ecg_rms - min_ecg_rms)

    ecg_peak_idxs, _ = scipy.signal.find_peaks(
        ecg_rms,
        height=peak_height,
        width=peak_width_s,
        distance=peak_distance
    )

    return ecg_peak_idxs


def gating(
    emg_raw,
    peak_idxs,
    gate_width=205,
    method=1,
):
    """
    Eliminate peaks (e.g. QRS) from emg_raw using gates
    of width gate_width. The gate either filled by zeros or interpolation.
    The filling method for the gate is encoded as follows:
    0: Filled with zeros
    1: Interpolation samples before and after
    2: Fill with average of prior segment if exists
    otherwise fill with post segment
    3: Fill with running average of RMS (default)

    :param emg_raw: Signal to process
    :type emg_raw: ~numpy.ndarray
    :param peak_idxs: list of individual peak index places to be gated
    :type peak_idxs: ~list
    :param gate_width: width of the gate
    :type gate_width: int
    :param method: filling method of gate
    :type method: int

    :returns: emg_raw_gated, the gated result
    :rtype: ~numpy.ndarray
    """
    emg_raw_gated = copy.deepcopy(emg_raw)
    max_sample = emg_raw_gated.shape[0]
    half_gate_width = gate_width // 2
    if method == 0:
        # Method 0: Fill with zeros
        # TODO: can rewrite with slices from numpy irange to be more efficient
        gate_samples = []
        for _, peak in enumerate(peak_idxs):
            for k in range(
                max(0, peak - half_gate_width),
                min(max_sample, peak + half_gate_width),
            ):
                gate_samples.append(k)

        emg_raw_gated[gate_samples] = 0
    elif method == 1:
        # Method 1: Fill with interpolation pre- and post gate sample
        # TODO: rewrite with numpy interpolation for efficiency
        for _, peak in enumerate(peak_idxs):
            pre_ave_emg = emg_raw[peak-half_gate_width-1]

            if (peak + half_gate_width + 1) < emg_raw_gated.shape[0]:
                post_ave_emg = emg_raw[peak+half_gate_width+1]
            else:
                post_ave_emg = 0

            k_start = max(0, peak-half_gate_width)
            k_end = min(
                peak+half_gate_width, emg_raw_gated.shape[0]
            )
            for k in range(k_start, k_end):
                frac = (k - peak + half_gate_width)/gate_width
                loup = (1 - frac) * pre_ave_emg + frac * post_ave_emg
                emg_raw_gated[k] = loup

    elif method == 2:
        # Method 2: Fill with window length mean over prior section
        # ..._____|_______|_______|XXXXXXX|XXXXXXX|_____...
        #         ^               ^- gate start   ^- gate end
        #         - peak - half_gate_width * 3 (replacer)

        for _, peak in enumerate(peak_idxs):
            start = peak - half_gate_width * 3
            if start < 0:
                start = peak + half_gate_width
            end = start + gate_width
            pre_ave_emg = np.nanmean(emg_raw[start:end])

            k_start = max(0, peak - half_gate_width)
            k_end = min(peak + half_gate_width, emg_raw_gated.shape[0])
            for k in range(k_start, k_end):
                emg_raw_gated[k] = pre_ave_emg

    elif method == 3:
        # Method 3: Fill with moving average over RMS
        gate_samples = []
        for _, peak in enumerate(peak_idxs):
            for k in range(
                max([0, int(peak-gate_width/2)]),
                min([max_sample, int(peak+gate_width/2)])
            ):
                gate_samples.append(k)

        emg_raw_gated_base = copy.deepcopy(emg_raw_gated)
        emg_raw_gated_base[gate_samples] = np.nan
        emg_raw_gated_rms = evl.full_rolling_rms(
            emg_raw_gated_base,
            gate_width,)

        interpolate_samples = list()
        for _, peak in enumerate(peak_idxs):
            k_start = max([0, int(peak-gate_width/2)])
            k_end = min([int(peak+gate_width/2), max_sample])

            for k in range(k_start, k_end):
                leftf = max([0, int(k-1.5*gate_width)])
                rightf = min([int(k+1.5*gate_width), max_sample])
                if any(np.logical_not(np.isnan(
                        emg_raw_gated_rms[leftf:rightf]))):
                    emg_raw_gated[k] = np.nanmean(
                        emg_raw_gated_rms[leftf:rightf]
                    )
                else:
                    interpolate_samples.append(k)

        if len(interpolate_samples) > 0:
            interpolate_samples = np.array(interpolate_samples)
            if 0 in interpolate_samples:
                emg_raw_gated[0] = 0

            if len(emg_raw_gated)-1 in interpolate_samples:
                emg_raw_gated[-1] = 0

            x_samp = np.array([x_i for x_i in range(len(emg_raw_gated))])
            other_samples = x_samp[~np.isin(x_samp, interpolate_samples)]
            emg_raw_gated_interp = np.interp(
                x_samp[interpolate_samples],
                x_samp[other_samples],
                emg_raw_gated[other_samples])
            emg_raw_gated[interpolate_samples] = emg_raw_gated_interp

    return emg_raw_gated


def find_peaks_in_ecg(ecg_raw, lower_border_percent=50):
    """
    This function assumes you have isolated an ecg-like signal with
    QRS peaks "higher" (or lower) than ST waves.
    In this case it can be applied to return an array of
    ECG peak locations. NB: This function assumes that the ECG
    signal has already been through a bandpass or low-pass filter
    or has little baseline drift.

    :param ecg_raw: frequency array sampled at in Hertz
    :type ecg_raw: ~numpy.ndarray
    :param low_border_percent: percentage max below which no peaks expected
    :type low_border_percent: int

    :returns: tuple first element peak locations, next a dictionary of info
    :rtype: tuple

    """
    # TODO: Eliminate. Not robust to +/- deflections.
    ecg_raw = abs(ecg_raw)
    max_peak = ecg_raw.max() - ecg_raw.min()
    set_ecg_peaks = find_peaks(
        ecg_raw,
        prominence=(max_peak*lower_border_percent/100, max_peak)
    )
    return set_ecg_peaks


def wavelet_denoising(
    emg_raw,
    ecg_peak_idxs,
    fs,
    hard_thresholding=True,
    n=4,
    wavelet_type='db2',
    fixed_threshold=4.5
):
    """
    Shrinkage Denoising using a-trous wavelet decomposition (SWT). NB: This
    function assumes that the emg_raw has already been preprocessed for
    removal of baseline, powerline, and aliasing. N.B. This is a Python
    implementation of the SWT, as previously implemented in MATLAB by Jan
    Graßhoff. See Copyright notice below.

    :param emg_raw: 1D raw EMG data
    :type emg_raw: numpy.ndarray
    :param ecg_peak_idxs: list of R-peaks indices
    :type ecg_peak_idxs: numpy.ndarray
    :param fs: Sampling rate of emg_raw
    :type fs: int
    :param hard_thresholding: True: hard (default), False: soft
    :type hard_thresholding: bool
    :param n: True: decomposition level (default: 4)
    :type n: int
    :param wavelet_type: wavelet type (default: 'db2', see pywt.swt help)
    :type wavelet_type: str

    :returns: (cleansed EMG, wavelet decomposition, thresholds, gate_windows)
    :rtype: tuple(numpy.ndarray, list(tuples), numpy.ndarray, numpy.ndarray)

    --------------------------------------------------------------------------
    Copyright 2019 Institute for Electrical Engineering in Medicine,
    University of Luebeck
    Jan Graßhoff

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    """
    def estimate_noise(signal, window_length):
        """
        Estimate noise level
        :param signal: wavelet-decomposed signal
        :type signal: numpy.ndarray
        :param window_length: window length for noise estimation
        :type window_length: int

        :returns: estimated noise level
        :rtype: numpy.ndarray(~float)
        """
        nb_level = signal.shape[0]
        std_estimated = np.zeros(signal.shape)

        for k in range(nb_level):
            # Estimate std from MAD: std ~ MAD/0.6745
            std_estimated[k, :] = pd.Series(np.abs(signal[k, :])).rolling(
                window=window_length,
                min_periods=1,
                center=True).median().values / 0.6745

            # Correct on- and offset effects
            std_estimated[k, :window_length // 2] = std_estimated[
                k, window_length // 2]
            std_estimated[k, -window_length // 2:] = std_estimated[
                k, -window_length // 2]
        return std_estimated

    def get_gate_windows(rpeak_bool_vec, window_length):
        """
        Generate gate windows for the peaks
        :param rpeak_bool_vec: 1D raw, where R-peak location == 1
        :type rpeak_bool_vec: numpy.ndarray
        :param window_length: number of samples to gate around peaks
        :type window_length: int

        :returns: gated signal based on R-peaks, where gate == 1
        :rtype: numpy.ndarray(int)
        """
        window_length = int(np.floor(window_length / 2) * 2)
        rpeak_idxs = np.where(rpeak_bool_vec == 1)[0]

        gate_windows = np.zeros_like(rpeak_bool_vec)
        for _, rpeak_idx in enumerate(rpeak_idxs):
            gate_windows[
                max(rpeak_idx - window_length // 2, 0):
                min(rpeak_idx + window_length // 2, len(rpeak_bool_vec))
            ] = 1

        return gate_windows

    def threshold_wavelets(data, hard_thresholding, threshold):
        """
        Apply thresholding to data based on 'soft' or 'hard' option
        :param data: input data
        :type data: numpy.ndarray
        :param hard_thresholding: True: hard (default), False: soft
        :type hard_thresholding: bool
        :param threshold: threshold value
        :type threshold: ~float

        :returns: thresholded data
        :rtype: numpy.ndarray
        """
        if hard_thresholding is True:
            # Hard thresholding
            data[np.abs(data) < threshold] = 0
        elif hard_thresholding is False:
            # Soft thresholding
            data = np.sign(data) * np.maximum(np.abs(data) - threshold, 0)
        return data

    # Calculate gate windows
    r_peak_bool = np.zeros(emg_raw.shape)
    r_peak_bool[ecg_peak_idxs] = 1
    gate_bool_array = get_gate_windows(r_peak_bool, fs//10)

    # Signal Extension by zero padding
    pow_2_n = 2 ** n
    n_samp = len(emg_raw)
    n_samp_extended = int(np.ceil(n_samp / pow_2_n) * pow_2_n)
    zero_padding = np.zeros(n_samp_extended - n_samp)
    emg_raw_zero_padded = np.concatenate((emg_raw, zero_padding))
    gate_bool_array = np.concatenate((gate_bool_array, zero_padding))

    # Wavelet decomposition of emg_raw using Stationary Wavelet Transform (SWT)
    wav_dec = pywt.swt(emg_raw_zero_padded, wavelet_type, level=n)
    wav_dec_unpacked = np.array(
        [[subband[0], subband[1]] for subband in wav_dec])
    swc = np.vstack((wav_dec_unpacked[:, 1, :], wav_dec_unpacked[n-1, 0, :]))

    # Gate out R-peaks in wavelet subbands
    wav_dec_gated = np.array(swc)
    wav_dec_gated[:, gate_bool_array == 1] = np.nan

    # Custom threshold coefficients
    window_length = 15 * fs
    # win_len = 15000
    s = estimate_noise(wav_dec_gated[:-1], window_length)

    thresholds = np.zeros_like(swc)
    wxd = np.array(wav_dec_unpacked)

    for k in range(n):
        threshold = fixed_threshold * s[k, :]
        thresholds[k, :] = threshold
        wxd[k, 1, :] = threshold_wavelets(
            wav_dec_unpacked[k, 1, :], hard_thresholding, threshold)

    # # Wavelet reconstruction
    ecg_reconstructd = pywt.iswt(
        [tuple(subband) for subband in wxd],
        wavelet_type
    )

    # Return results
    wav_dec = np.array(swc)
    ecg_reconstructd = ecg_reconstructd[:n_samp]
    thresholds = thresholds[:, :n_samp]
    gate_bool_array = gate_bool_array[:n_samp]
    emg_clean = emg_raw - ecg_reconstructd

    return emg_clean, wav_dec, thresholds, gate_bool_array
