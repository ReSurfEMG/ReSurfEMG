"""
Copyright 2022 Netherlands eScience Center and U. Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to work with various EMG arrays
and other types of data arrays e.g. ventilator signals
when EMG leads represent something other than inspiratory muscles
and/or diaphragm in some cases.
"""

import collections
from collections import namedtuple
import math
from math import log, e
import builtins
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from resurfemg.helper_functions import bad_end_cutter_for_samples
from resurfemg.helper_functions import emg_bandpass_butter_sample
from resurfemg.helper_functions import pick_lowest_correlation_array
from resurfemg.helper_functions import pick_more_peaks_array
from resurfemg.helper_functions import emg_highpass_butter


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
    ica = FastICA(n_components=2)
    S = ica.fit_transform(X)
    component_0 = S.T[0]
    component_1 = S.T[1]
    return component_0, component_1


def working_pipe_multi(our_chosen_samples, picker='heart', selected=(0, 2)):
    """
    This is a pipeline to pre-process
    an array of any dimenstions (number of leads)
    into an EMG singal, you need to pick the leads

    :param our_chosen_samples: the read EMG file arrays
    :type our_chosen_samples: ~numpy.ndarray
    :param picker: the picking strategy for independant components
    :type picker: str
    :param selected: the leads selected for the pipeline to run over
    :type selected: tuple

    :returns: final_envelope_a
    :rtype: ~numpy.ndarray
    """
    cut_file_data = bad_end_cutter_for_samples(
        our_chosen_samples,
        percent_to_cut=3,
        tolerance_percent=5
    )
    bd_filtered_file_data = emg_bandpass_butter_sample(
        cut_file_data,
        5,
        450,
        2048,
        output='sos'
    )
    # step for end-cutting again to get rid of filtering artifacts
    re_cut_file_data = bad_end_cutter_for_samples(
        bd_filtered_file_data,
        percent_to_cut=3,
        tolerance_percent=5
    )
    #  and do step for ICA
    components = compute_ICA_two_comp_selective(
        re_cut_file_data,
        False,
        selected
    )
    #     the picking step!
    if picker == 'peaks':
        emg = pick_more_peaks_array(components)
    elif picker == 'heart':
        emg = pick_lowest_correlation_array(components, re_cut_file_data[0])
    else:
        emg = pick_lowest_correlation_array(components, re_cut_file_data[0])
        print("Please choose an exising picker i.e. peaks or hearts ")
    # now process it in final steps
    abs_values = abs(emg)
    final_envelope_d = emg_highpass_butter(abs_values, 150, 2048)

    return final_envelope_d


def working_pipeline_pre_ml_multi(
    our_chosen_samples,
    our_chosen_leads,
    picker='heart',
):
    """
    This is a pipeline to pre-process
    an array of non-specific dimensions
    e.g. a five lead array into an EMG singal,
    of which we want leads 0 to 2 included

    :param our_chosen_samples: the read EMG file arrays
    :type our_chosen_samples: ~numpy.ndarray
    :param our_chosen_leads: the read EMG file arrays that should be included
    :type our_chosen_leads: tuple
    :param picker: the picking strategy for independant components
    :type picker: str

    :returns: final_envelope_a
    :rtype: ~numpy.ndarray
    """
    cut_file_data = bad_end_cutter_for_samples(
        our_chosen_samples,
        percent_to_cut=3,
        tolerance_percent=5
    )
    bd_filtered_file_data = emg_bandpass_butter_sample(
        cut_file_data,
        5,
        450,
        2048,
        output='sos'
    )
    # step for end-cutting again to get rid of filtering artifacts
    re_cut_file_data = bad_end_cutter_for_samples(
        bd_filtered_file_data,
        percent_to_cut=3,
        tolerance_percent=5
    )
    #  and do step for ICA
    # TO BE REMOVED: components = compute_ICA_two_comp(re_cut_file_data)
    components = compute_ICA_two_comp_selective(
        re_cut_file_data,
        use_all_leads=False,
        desired_leads=our_chosen_leads,
        )
    #     the picking step!
    if picker == 'peaks':
        emg = pick_more_peaks_array(components)
    elif picker == 'heart':
        emg = pick_lowest_correlation_array(components, re_cut_file_data[0])
    else:
        emg = pick_lowest_correlation_array(components, re_cut_file_data[0])
        print("Please choose an exising picker i.e. peaks or hearts ")
    # now process it in final steps
    abs_values = abs(emg)
    final_envelope_d = emg_highpass_butter(abs_values, 150, 2048)

    return final_envelope_d
