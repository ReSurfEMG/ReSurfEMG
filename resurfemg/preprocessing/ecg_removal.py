"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to eliminate ECG artifacts from
 with various EMG arrays.
"""

import copy
import numpy as np
import scipy
from sklearn.decomposition import FastICA
from scipy.signal import find_peaks

from resurfemg.preprocessing import envelope as evl
import resurfemg.preprocessing.filtering as filt


def compute_ica_two_comp(emg_samples):
    """A function that performs an independent component analysis
    (ICA) meant for EMG data that includes three stacked arrays.

    :param emg_samples: Original signal array with three layers
    :type emg_samples: ~numpy.ndarray

    :returns: Two arrays of independent components (ECG-like and EMG)
    :rtype: ~numpy.ndarray
    """
    X = np.c_[emg_samples[0], emg_samples[2]]
    ica = FastICA(n_components=2, random_state=1)
    S = ica.fit_transform(X)
    component_0 = S.T[0]
    component_1 = S.T[1]
    return component_0, component_1


def compute_ica_two_comp_multi(emg_samples):
    """A function that performs an independant component analysis
    (ICA) meant for EMG data that includes stacked arrays,
    there should be at least two arrays but there can be more.

    :param emg_samples: Original signal array with three or more layers
    :type emg_samples: ~numpy.ndarray

    :returns: Two arrays of independent components (ECG-like and EMG)
    :rtype: ~numpy.ndarray
    """
    all_component_numbers = list(range(emg_samples.shape[0]))
    list_to_c = []
    for i in all_component_numbers:
        list_to_c.append(emg_samples[i])
    X = np.column_stack(list_to_c)
    ica = FastICA(n_components=2, random_state=1)
    S = ica.fit_transform(X)
    component_0 = S.T[0]
    component_1 = S.T[1]
    return component_0, component_1


def compute_ICA_two_comp_selective(
    emg_samples,
    use_all_leads=True,
    desired_leads=(0, 2),
):
    """A function that performs an independant component analysis
    (ICA) meant for EMG data that includes stacked arrays,
    there should be at least two arrays but there can be more.

    :param emg_samples: Original signal array with three or more layers
    :type emg_samples: ~numpy.ndarray
    :param use_all_leads: True if all leads used, otherwise specify leads
    :type use_all_leads: bool
    :param desired_leads: tuple of leads to use starting from 0
    :type desired_leads: tuple

    :returns: Two arrays of independent components (ECG-like and EMG)
    :rtype: ~numpy.ndarray
    """
    if use_all_leads:
        all_component_numbers = list(range(emg_samples.shape[0]))
    else:
        all_component_numbers = desired_leads
        diff = set(all_component_numbers) - set(range(emg_samples.shape[0]))
        if diff:
            raise IndexError(
                "You picked nonexistant leads {}, "
                "please see documentation".format(diff)
            )
    list_to_c = []
    # TODO (makeda): change to list comprehension on refactoring
    for i in all_component_numbers:
        list_to_c.append(emg_samples[i])
    X = np.column_stack(list_to_c)
    ica = FastICA(n_components=2, random_state=1)
    S = ica.fit_transform(X)
    component_0 = S.T[0]
    component_1 = S.T[1]
    return component_0, component_1


def compute_ICA_n_comp(
    emg_samples,
    use_all_leads=True,
    desired_leads=(0, 2),
):
    """A function that performs an independant component analysis
    (ICA) meant for EMG data that includes stacked arrays,
    there should be at least two arrays but there can be more.
    This differs from helper_functions.compute_ICA_two_comp_multi
    because you can get n leads back instead of only two.

    :param emg_samples: Original signal array with three or more layers
    :type emg_samples: ~numpy.ndarray
    :param use_all_leads: True if all leads used, otherwise specify leads
    :type use_all_leads: bool
    :param desired_leads: tuple of leads to use starting from 0
    :type desired_leads: tuple

    :returns: Arrays of independent components (ECG-like and EMG)
    :rtype: ~numpy.ndarray
    """
    if use_all_leads:
        all_component_numbers = list(range(emg_samples.shape[0]))
        n_components = len(all_component_numbers)
    else:
        all_component_numbers = desired_leads
        n_components = len(all_component_numbers)
        diff = set(all_component_numbers) - set(range(emg_samples.shape[0]))
        if diff:
            raise IndexError(
                "You picked nonexistant leads {}, "
                "please see documentation".format(diff)
            )
    list_to_c = []
    # TODO (makeda): change to list comprehension on refactoring
    for i in all_component_numbers:
        list_to_c.append(emg_samples[i])
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
    emg_samples,
    ecg_lead_to_remove,
    use_all_leads=True,
    desired_leads=(0, 2),
):
    """A function that performs an independant component analysis
    (ICA) meant for EMG data that includes stacked arrays,
    there should be at least two arrays but there can be more.
    In this ICA one lead is put to zero before reconstruction.
    This should probably be the ECG lead.

    :param emg_samples: Original signal array with three or more layers
    :type emg_samples: ~numpy.ndarray
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
        all_component_numbers = list(range(emg_samples.shape[0]))
        n_components = len(all_component_numbers)
    else:
        all_component_numbers = desired_leads
        n_components = len(all_component_numbers)
        diff = set(all_component_numbers) - set(range(emg_samples.shape[0]))
        if diff:
            raise IndexError(
                "You picked nonexistant leads {}, "
                "please see documentation".format(diff)
            )
    list_to_c = []
    # TODO (makeda): change to list comprehension on refactoring
    for i in all_component_numbers:
        list_to_c.append(emg_samples[i])

    X = np.column_stack(list_to_c)
    ica = FastICA(n_components, random_state=1)
    S = ica.fit_transform(X)
    S_copy = copy(S)

    hi_index = pick_highest_correlation_array_multi(
        S_copy.transpose(),
        emg_samples[ecg_lead_to_remove])

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

    :returns: emg_gated
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

    ecg_peaks_s, _ = scipy.signal.find_peaks(
        ecg_rms,
        height=peak_height,
        width=peak_width_s,
        distance=peak_distance
    )

    return ecg_peaks_s


def gating(
    src_signal,
    gate_peaks,
    gate_width=205,
    method=1,
):
    """
    Eliminate peaks (e.g. QRS) from src_signal using gates
    of width gate_width. The gate either filled by zeros or interpolation.
    The filling method for the gate is encoded as follows:
    0: Filled with zeros
    1: Interpolation samples before and after
    2: Fill with average of prior segment if exists
    otherwise fill with post segment
    3: Fill with running average of RMS (default)

    :param src_signal: Signal to process
    :type src_signalsignal: ~numpy.ndarray
    :param gate_peaks: list of individual peak index places to be gated
    :type gate_peaks: ~list
    :param gate_width: width of the gate
    :type gate_width: int
    :param method: filling method of gate
    :type method: int

    :returns: src_signal_gated, the gated result
    :rtype: ~numpy.ndarray
    """
    src_signal_gated = copy.deepcopy(src_signal)
    max_sample = src_signal_gated.shape[0]
    half_gate_width = gate_width // 2
    if method == 0:
        # Method 0: Fill with zeros
        # TODO: can rewrite with slices from numpy irange to be more efficient
        gate_samples = []
        for _, peak in enumerate(gate_peaks):
            for k in range(
                max(0, peak - half_gate_width),
                min(max_sample, peak + half_gate_width),
            ):
                gate_samples.append(k)

        src_signal_gated[gate_samples] = 0
    elif method == 1:
        # Method 1: Fill with interpolation pre- and post gate sample
        # TODO: rewrite with numpy interpolation for efficiency
        for _, peak in enumerate(gate_peaks):
            pre_ave_emg = src_signal[peak-half_gate_width-1]

            if (peak + half_gate_width + 1) < src_signal_gated.shape[0]:
                post_ave_emg = src_signal[peak+half_gate_width+1]
            else:
                post_ave_emg = 0

            k_start = max(0, peak-half_gate_width)
            k_end = min(
                peak+half_gate_width, src_signal_gated.shape[0]
            )
            for k in range(k_start, k_end):
                frac = (k - peak + half_gate_width)/gate_width
                loup = (1 - frac) * pre_ave_emg + frac * post_ave_emg
                src_signal_gated[k] = loup

    elif method == 2:
        # Method 2: Fill with window length mean over prior section
        # ..._____|_______|_______|XXXXXXX|XXXXXXX|_____...
        #         ^               ^- gate start   ^- gate end
        #         - peak - half_gate_width * 3 (replacer)

        for _, peak in enumerate(gate_peaks):
            start = peak - half_gate_width * 3
            if start < 0:
                start = peak + half_gate_width
            end = start + gate_width
            pre_ave_emg = np.nanmean(src_signal[start:end])

            k_start = max(0, peak - half_gate_width)
            k_end = min(peak + half_gate_width, src_signal_gated.shape[0])
            for k in range(k_start, k_end):
                src_signal_gated[k] = pre_ave_emg

    elif method == 3:
        # Method 3: Fill with moving average over RMS
        gate_samples = []
        for _, peak in enumerate(gate_peaks):
            for k in range(
                max([0, int(peak-gate_width/2)]),
                min([max_sample, int(peak+gate_width/2)])
            ):
                gate_samples.append(k)

        src_signal_gated_base = copy.deepcopy(src_signal_gated)
        src_signal_gated_base[gate_samples] = np.nan
        src_signal_gated_rms = evl.full_rolling_rms(
            src_signal_gated_base,
            gate_width,)

        interpolate_samples = list()
        for _, peak in enumerate(gate_peaks):
            k_start = max([0, int(peak-gate_width/2)])
            k_end = min([int(peak+gate_width/2), max_sample])

            for k in range(k_start, k_end):
                leftf = max([0, int(k-1.5*gate_width)])
                rightf = min([int(k+1.5*gate_width), max_sample])
                if any(np.logical_not(np.isnan(
                        src_signal_gated_rms[leftf:rightf]))):
                    src_signal_gated[k] = np.nanmean(
                        src_signal_gated_rms[leftf:rightf]
                    )
                else:
                    interpolate_samples.append(k)

        if len(interpolate_samples) > 0:
            interpolate_samples = np.array(interpolate_samples)
            if 0 in interpolate_samples:
                src_signal_gated[0] = 0

            if len(src_signal_gated)-1 in interpolate_samples:
                src_signal_gated[-1] = 0

            x_samp = np.array([x_i for x_i in range(len(src_signal_gated))])
            other_samples = x_samp[~np.isin(x_samp, interpolate_samples)]
            src_signal_gated_interp = np.interp(
                x_samp[interpolate_samples],
                x_samp[other_samples],
                src_signal_gated[other_samples])
            src_signal_gated[interpolate_samples] = src_signal_gated_interp

    return src_signal_gated


def find_peaks_in_ecg_signal(ecg_signal, lower_border_percent=50):
    """
    This function assumes you have isolated an ecg-like signal with
    QRS peaks "higher" (or lower) than ST waves.
    In this case it can be applied to return an array of
    ECG peak locations. NB: This function assumes that the ECG
    signal has already been through a bandpass or low-pass filter
    or has little baseline drift.

    :param ecg_signal: frequency array sampled at in Hertz
    :type ecg_signal: ~numpy.ndarray
    :param low_border_percent: percentage max below which no peaks expected
    :type low_border_percent: int

    :returns: tuple first element peak locations, next a dictionary of info
    :rtype: tuple

    """
    ecg_signal = abs(ecg_signal)
    max_peak = ecg_signal.max() - ecg_signal.min()
    set_ecg_peaks = find_peaks(
        ecg_signal,
        prominence=(max_peak*lower_border_percent/100, max_peak)
    )
    return set_ecg_peaks
