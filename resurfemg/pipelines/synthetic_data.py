
"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to generate mixed (signal and noise) synthetic
data.
"""
import numpy as np
import neurokit2 as nk
import resurfemg.data_connector.synthetic_data as synth

def simulate_raw_emg(
    t_end,
    fs_emg,
    emg_amp=5,
    rr=22,
    **kwargs
):
    """
    Generate realistic synthetic respiratory EMG data remixed with EMG
    :param t_end: length of synthetic EMG tracing in seconds
    :type t_end: int
    :param fs_emg:
    :type fs_emg:
    :param emg_amp:
    :type emg_amp:
    :param rr:
    :type rr:
    :param **kwargs: Optional arguments: ie_ratio, tau_mus_up, tau_mus_down,
    t_p_occs, drift_amp, noise_amp, ecg_acceleration, ecg_amplitude. See
    data_connector.synthetic_data respiratory_pattern_generator, 
    simulate_muscle_dynamics functions simulate_emg for specific
    :type **kwargs: float, float, float, list[int], float, float, float

    :returns emg_raw: The realistic synthetic EMG
    :rtype emg_raw: numpy.ndarray
    """
    sim_parameters = {
        'ie_ratio': 1/2,  # ratio btw insp + expir phase
        'tau_mus_up': 0.3,
        'tau_mus_down': 0.3,
        't_p_occs': [],
        'drift_amp': 100,
        'noise_amp': 2,
        'heart_rate':80,
        'ecg_acceleration': 1.5,
        'ecg_amplitude':200,
    }
    for key, value in kwargs.items():
        if key in sim_parameters.keys():
            sim_parameters[key] = value
        else:
            raise UserWarning('kwarg `{0}` not available.'.format(key))
            
    respiratory_pattern = synth.respiratory_pattern_generator(
        t_end=t_end,
        fs_emg=fs_emg,
        rr=rr,
        ie_ratio=sim_parameters['ie_ratio'],
        t_p_occs=sim_parameters['t_p_occs']
    )
    muscle_activation = synth.simulate_muscle_dynamics(
        block_pattern=respiratory_pattern,
        fs_emg=fs_emg,
        tau_mus_up=sim_parameters['tau_mus_up'],
        tau_mus_down=sim_parameters['tau_mus_down'],
    )
    emg_sim = synth.simulate_emg(
        muscle_activation=muscle_activation,
        fs_emg=fs_emg,
        emg_amp=emg_amp,
        drift_amp=sim_parameters['drift_amp'],
        noise_amp=sim_parameters['noise_amp'],
    )
    sim_hr = sim_parameters['heart_rate']/sim_parameters['ecg_acceleration']
    fs_ecg = int(fs_emg*sim_parameters['ecg_acceleration'])
    ecg_t_end = int(t_end / sim_parameters['ecg_acceleration'])
    ecg_sim = nk.ecg_simulate(
        duration=ecg_t_end,
        sampling_rate=fs_ecg,
        heart_rate=sim_hr,
    )
    ecg_sim = ecg_sim[:len(emg_sim)]
    emg_raw = sim_parameters['ecg_amplitude'] * ecg_sim + emg_sim
    return emg_raw


def synthetic_emg_cli(file_directory, n_emg, output_directory):
    """
    This function works with the cli
    module to makes realistic synthetic respiratory EMG data
    through command line.

    :param file_directory: file directory where synthetic ecg are
    :type file_directory: str
    :param n_emg: number of EMGs to simulate
    :type n_emg: int
    :param output_directory: file directory where synthetic emg will be put
    :type output_directory: str
    """
    file_directory_list = glob.glob(
        os.path.join(file_directory, '*.npy'),
        recursive=True,
    )
    file = file_directory_list[0]
    loaded = np.load(file)
    synthetics = make_realistic_syn_emg(loaded, n_emg)
    number_end = 0
    for single_synth in synthetics:
        out_fname = os.path.join(output_directory, str(number_end))
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        np.save(out_fname, single_synth)
        number_end += 1
