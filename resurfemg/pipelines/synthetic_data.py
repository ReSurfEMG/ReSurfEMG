
"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to generate mixed (signal and noise) synthetic
data.
"""
import os
import math

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
    Generate realistic synthetic respiratory EMG data remixed with ECG
    ---------------------------------------------------------------------------
    :param t_end: length of synthetic EMG tracing in seconds
    :type t_end: int
    :param fs_emg: Sampling rate
    :type fs_emg: int
    :param emg_amp: EMG amplitude
    :type emg_amp: ~float
    :param rr: Respiratory rate (/min)
    :type rr: float
    :param **kwargs: Optional arguments: ie_ratio, tau_mus_up, tau_mus_down,
    t_p_occs, drift_amp, noise_amp, ecg_acceleration, ecg_amplitude. See
    data_connector.synthetic_data respiratory_pattern_generator,
    simulate_muscle_dynamics, and simulate_emg functions for specifics
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
        'heart_rate': 80,
        'ecg_acceleration': 1.5,
        'ecg_amplitude': 200,
    }
    for key, value in kwargs.items():
        if key in sim_parameters:
            sim_parameters[key] = value
        else:
            raise UserWarning(f"kwarg `{key}` not available.")

    respiratory_pattern = synth.respiratory_pattern_generator(
        t_end=t_end,
        fs=fs_emg,
        rr=rr,
        ie_ratio=sim_parameters['ie_ratio'],
        t_p_occs=sim_parameters['t_p_occs']
    )
    muscle_activation = synth.simulate_muscle_dynamics(
        block_pattern=respiratory_pattern,
        fs=fs_emg,
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
    ecg_t_end = int(math.ceil(t_end / sim_parameters['ecg_acceleration']))
    ecg_sim = nk.ecg_simulate(
        duration=ecg_t_end,
        sampling_rate=fs_ecg,
        heart_rate=sim_hr,
    )
    ecg_sim = ecg_sim[:len(emg_sim)]
    emg_raw = sim_parameters['ecg_amplitude'] * ecg_sim + emg_sim
    return emg_raw


def synthetic_emg_cli(n_emg, output_directory, **kwargs):
    """
    Generate realistic, single lead, synthetic respiratory EMG data remixed
    with ECG through command line using the cli.
    ---------------------------------------------------------------------------
    :param file_directory: file directory where synthetic ecg are
    :type file_directory: str
    :param n_emg: number of EMGs to simulate
    :type n_emg: int
    :param output_directory: file directory where synthetic emg will be put
    :type output_directory: str

    :param **kwargs: Optional arguments: t_end, fs_emg, emg_amp, rr, ie_ratio,
    tau_mus_up, tau_mus_down, t_p_occs, drift_amp, noise_amp, heart_rate,
    ecg_acceleration, ecg_amplitude. See data_connector.synthetic_data
    respiratory_pattern_generator, simulate_muscle_dynamics, and simulate_emg
    functions for specifics
    :type **kwargs: float, float, float, float, float, float, float, list[int],
    float, float, float, float, float

    :returns: None
    :rtype: None
    """
    sim_parameters = {
        't_end': 7*60,
        'fs_emg': 2048,   # hertz
        'rr': 22,         # respiratory rate /min
        'ie_ratio': 1/2,  # ratio btw insp + expir phase
        'tau_mus_up': 0.3,
        'tau_mus_down': 0.3,
        't_p_occs': [],
        'drift_amp': 100,
        'noise_amp': 2,
        'heart_rate': 80,
        'ecg_acceleration': 1.5,
        'ecg_amplitude': 200,
    }
    for key, value in kwargs.items():
        if key in sim_parameters:
            sim_parameters[key] = value
        else:
            raise UserWarning(f"kwarg `{key}` not available.")

    for i in range(n_emg):
        emg_raw = simulate_raw_emg(
            **sim_parameters
        )
        out_fname = os.path.join(output_directory, 'emg_' + str(i))
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        np.save(out_fname, emg_raw)
        print(f"File(s) saved to {output_directory}.")


def simulate_ventilator_data(
    t_end,
    fs_vent,
    p_mus_amp=5,
    rr=22,
    dp=5,
    **kwargs
):
    """
    Generate realistic synthetic ventilator tracings
    ---------------------------------------------------------------------------
    :param t_end: length of synthetic ventilator tracings in seconds
    :type t_end: int
    :param fs_vent: Sampling rate
    :type fs_vent: int
    :param p_mus_amp: Respiratory muscle pressure amplitude (positive)
    :type p_mus_amp: float
    :param rr: Respiratory rate
    :type rr: float
    :param **kwargs: Optional arguments: ie_ratio, tau_mus_up, tau_mus_down,
    t_p_occs, c, r, peep, flow_cycle, flow_trigger, tau_dp_up, tau_dp_down. See
    data_connector.synthetic_data respiratory_pattern_generator,
    simulate_muscle_dynamics, simulate_ventilator_data functions for specifics
    :type **kwargs: float, float, float, list[int], float, float, float, float,
    float, float, float

    :returns y_vent: The realistic synthetic ventilator data
    :rtype y_vent: numpy.ndarray
    :returns p_mus: The respiratory muscle pressure
    :rtype p_mus: numpy.ndarray
    """
    sim_parameters = {
        'ie_ratio': 1/2,  # ratio btw insp + expir phase
        'tau_mus_up': 0.3,
        'tau_mus_down': 0.3,
        't_p_occs': [],
        'c': .050,
        'r': 5,
        'peep': 5,
        'flow_cycle': 0.25,  # Fraction F_max
        'flow_trigger': 2,   # L/min
        'tau_dp_up': 10,
        'tau_dp_down': 5,
    }

    for key, value in kwargs.items():
        if key in sim_parameters:
            sim_parameters[key] = value
        else:
            raise UserWarning(f"kwarg `{key}` not available.")

    respiratory_pattern = synth.respiratory_pattern_generator(
        t_end=t_end,
        fs=fs_vent,
        rr=rr,
        ie_ratio=sim_parameters['ie_ratio'],
        t_p_occs=sim_parameters['t_p_occs'],
    )
    p_mus = p_mus_amp * synth.simulate_muscle_dynamics(
        block_pattern=respiratory_pattern,
        fs=fs_vent,
        tau_mus_up=sim_parameters['tau_mus_up'],
        tau_mus_down=sim_parameters['tau_mus_down'],
    )
    t_occ_bool = np.zeros(p_mus.shape, dtype=bool)
    for t_occ in sim_parameters['t_p_occs']:
        t_occ_bool[int((t_occ-1)*fs_vent):
                   int((t_occ+1/rr*60)*fs_vent)] = True
    lung_mechanics = {
        'c': .050,
        'r': 5,
    }
    vent_settings = {
        'dp': dp,
        'peep': 5,
        'flow_cycle': 0.25,  # Fraction F_max
        'flow_trigger': 2,   # L/min
        'tau_dp_up': 10,
        'tau_dp_down': 5,
    }
    for key, value in sim_parameters.items():
        if key in lung_mechanics:
            lung_mechanics[key] = value
        elif key in vent_settings:
            vent_settings[key] = value

    y_vent = synth.simulate_ventilator_data(**{
        'p_mus': p_mus,
        'dp': dp,
        'fs_vent': fs_vent,
        't_occ_bool': t_occ_bool,
        **lung_mechanics,
        **vent_settings
    })
    return y_vent, p_mus


def synthetic_ventilator_data_cli(n_datasets, output_directory, **kwargs):
    """
    Generate realistic synthetic respiratory EMG data remixed with ECG through
    command line using the cli.
    ---------------------------------------------------------------------------
    :param file_directory: file directory where synthetic ecg are
    :type file_directory: str
    :param n_emg: number of EMGs to simulate
    :type n_emg: int
    :param output_directory: file directory where synthetic emg will be put
    :type output_directory: str

    :param **kwargs: Optional arguments: t_end, fs_vent, p_mus_amp, rr, dp,
    ie_ratio, tau_mus_up, tau_mus_down, t_p_occs, c, r, peep, flow_cycle,
    flow_trigger, tau_dp_up, tau_dp_down. See data_connector.synthetic_data
    respiratory_pattern_generator, simulate_muscle_dynamics,
    simulate_ventilator_data functions for specifics
    :type **kwargs: float, float, float, float, float, float, float, float,
    list[int], float, float, float, float, float, float

    :returns: None
    :rtype: None
    """
    sim_parameters = {
        't_end': 7*60,
        'fs_vent': 100,
        'p_mus_amp': 5,
        'rr': 22,
        'dp': 5,
        'ie_ratio': 1/2,  # ratio btw insp + expir phase
        'tau_mus_up': 0.3,
        'tau_mus_down': 0.3,
        't_p_occs': [],
        'c': .050,
        'r': 5,
        'peep': 5,
        'flow_cycle': 0.25,  # Fraction F_max
        'flow_trigger': 2,   # L/min
        'tau_dp_up': 10,
        'tau_dp_down': 5,
    }
    for key, value in kwargs.items():
        if key in sim_parameters:
            sim_parameters[key] = value
        else:
            raise UserWarning(f"kwarg `{key}` not available.")

    for i in range(n_datasets):
        y_vent, p_mus = simulate_ventilator_data(
            **sim_parameters
        )
        y_sig = np.vstack((y_vent, p_mus))
        out_fname = os.path.join(output_directory, 'vent_' + str(i))
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        np.save(out_fname, y_sig)
