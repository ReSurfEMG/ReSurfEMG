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
import os
import glob
from copy import copy
from sklearn.decomposition import FastICA
from resurfemg.helper_functions import bad_end_cutter_for_samples
from resurfemg.helper_functions import emg_bandpass_butter_sample
from resurfemg.helper_functions import pick_lowest_correlation_array
from resurfemg.helper_functions import pick_more_peaks_array
from resurfemg.helper_functions import emg_highpass_butter
from resurfemg.helper_functions import pick_highest_correlation_array
from resurfemg.tmsisdk_lite import Poly5Reader


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


def save_preprocessed(array, out_fname, force):
    """
    This function is written to be called by the cli module.
    It stores arrays in a directory.
    """
    if not force:
        if os.path.isfile(out_fname):
            return
    try:
        os.makedirs(os.path.dirname(out_fname))
    except FileExistsError:
        pass
    np.save(out_fname, array, allow_pickle=False)


def preprocess(
    file_directory,
    our_chosen_leads,
    algorithm,
    processed,
    force=False
):
    """
    This function is written to be called by the cli module.
    The cli module supports command line pre-processing.
    This function is currently written to accomodate Poly5 files types.
    It can be refactored later.

    :param file_directory: the directory with EMG files
    :type file_directory: str
    :param processed: the output directory
    :type processed: str
    :param our_chosen_leads: the leads selected for the pipeline to run over
    :type our_chosen_leads: list

    """
    file_directory_list = glob.glob(
        os.path.join(file_directory, '**/*.Poly5'),
        recursive=True,
    )
    for file in file_directory_list:
        reader = Poly5Reader(file)
        if algorithm == 'alternative_a_pipeline_multi':
            array = alternative_a_pipeline_multi(
                reader.samples,
                our_chosen_leads,
                picker='heart',
            )
        elif algorithm == 'alternative_b_pipeline_multi':
            array = alternative_b_pipeline_multi(
                reader.samples,
                our_chosen_leads,
                picker='heart',
            )

        else:
            array = working_pipeline_pre_ml_multi(
                reader.samples,
                our_chosen_leads,
                picker='heart',
            )
        rel_fname = os.path.relpath(file, file_directory)
        out_fname = os.path.join(processed, rel_fname)
        save_preprocessed(array, out_fname, force)


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


def alternative_a_pipeline_multi(
    our_chosen_samples,
    our_chosen_leads,
    picker='heart',
):
    """
    This is a pipeline to pre-process
    an array of non-specific dimensions
    e.g. a five lead array into an EMG singal,
    of which we want leads 0 to 2 included.
    Note it only differs in the bandpass values
    from working_pipeline_pre_ml_multi

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
        6,
        400,
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


def alternative_b_pipeline_multi(
    our_chosen_samples,
    our_chosen_leads,
    picker='heart',
):
    """
    This is a pipeline to pre-process
    an array of non-specific dimensions
    e.g. a five lead array into an EMG singal,
    of which we want leads 0 to 2 included.
    Note it only differs in the bandpass values
    from working_pipeline_pre_ml_multi or
    alternative_a_pipeline_multi

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
        4,
        300,
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

    corr_tuple = np.row_stack((ecg_lead, components))
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
