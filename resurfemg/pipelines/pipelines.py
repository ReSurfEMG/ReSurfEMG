# import os
# import glob
from resurfemg.preprocessing.filtering import bad_end_cutter_for_samples
from resurfemg.preprocessing.filtering import emg_bandpass_butter_sample
from resurfemg.preprocessing.ecg_removal import compute_ICA_two_comp_selective
from resurfemg.preprocessing.ecg_removal import pick_more_peaks_array
from resurfemg.preprocessing.ecg_removal import pick_lowest_correlation_array
from resurfemg.preprocessing.filtering import emg_highpass_butter
from resurfemg.preprocessing.filtering import bad_end_cutter
from resurfemg.preprocessing.envelope import naive_rolling_rms
from ..preprocessing.ecg_removal import compute_ica_two_comp

# from ..data_connector.tmsisdk_lite import Poly5Reader
# from ..helper_functions.helper_functions import save_preprocessed


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


def working_pipeline_exp(our_chosen_file):
    """This function is legacy.
    It produces a filtered respiratory EMG signal from a
    3 lead sEMG file. A better
    option is a corresponding function in multi_lead_type
    The inputs are :code:`our_chosen_file` which we
    give the function as a string of filename.  The output is the
    processed EMG signal filtered and seperated from ECG components.
    The algorithm to pick out the EMG here is by having
    more peaks.

    :param our_chosen_file: Poly5 file
    :type our_chosen_file: ~TMSiSDK.file_readers.Poly5Reader

    :returns: final_envelope_a
    :rtype: ~numpy.ndarray
    """
    cut_file_data = bad_end_cutter(
        our_chosen_file,
        percent_to_cut=3,
        tolerance_percent=5,
    )
    bd_filtered_file_data = emg_bandpass_butter_sample(
        cut_file_data,
        5,
        450,
        2048,
        output='sos',
    )
    # end-cutting again to get rid of filtering artifacts
    re_cut_file_data = bad_end_cutter_for_samples(
        bd_filtered_file_data,
        percent_to_cut=3,
        tolerance_percent=5,
    )
    # do ICA
    components = compute_ica_two_comp(re_cut_file_data)
    #  pick components with more peak
    emg = pick_more_peaks_array(components)
    # now process it in final steps
    abs_values = abs(emg)
    final_envelope_d = emg_highpass_butter(abs_values, 150, 2048)
    final_envelope_a = naive_rolling_rms(final_envelope_d, 300)

    return final_envelope_a


def working_pipeline_pre_ml(our_chosen_samples, picker='heart'):
    """
    This is a pipeline to pre-process
    an array of specific fixed dimensions
    i.e. a three lead array into an EMG singal,
    the function is legacy code, and most
    processsing should be done with
    :code:`multi_lead_type.working_pipeline_pre_ml_multi`
    or :code:`multi_lead_type.working_pipeline_pre_ml_multi`
    :param our_chosen_samples: the read EMG file arrays
    :type our_chosen_samples: ~numpy.ndarray
    :param picker: the picking strategy for independent components
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
    components = compute_ica_two_comp(re_cut_file_data)
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
