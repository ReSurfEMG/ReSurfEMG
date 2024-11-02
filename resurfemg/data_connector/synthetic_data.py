"""
Copyright 2022 Netherlands eScience Center and Twente University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods to create synthetic data with several methods.
"""

import os
import glob
import math
import random
import pandas as pd
import numpy as np
from scipy import signal
from resurfemg.preprocessing import filtering as filt


def make_synth_emg(n_samp, max_abs_volt, n_breaths):
    """
    Function to create synthetic EMG, not to add longer expirations
    (relatively flat).
    :param n_samp: The number of samples of created synth EMG
    :type n_samp: int
    :param max_abs_volt: desired absolute maximum potential
    :type max_abs_volt: float
    :param n_breaths: desired number of inspiratory waves
    :type n_breaths: int

    :returns: synthetic_emg_raw
    :rtype: ~np.array
    """
    x = np.linspace(0, (n_samp/500), n_samp)
    raised_sin = np.sin(x * n_breaths / 60*2*math.pi) ** 2
    synth_emg1 = raised_sin * np.random.randn(
        (len(x))) + 0.1 * np.random.randn(len(x))
    volt_multiplier = max_abs_volt / abs(synth_emg1).max()
    synthetic_emg_raw = synth_emg1 * volt_multiplier
    return synthetic_emg_raw


def simulate_ventilator_with_occlusions(
    t_p_occs,
    t_start=0,
    t_end=7*60,
    fs_vent=2048,
    rr=22,
    ie_ratio=1/2,
    p_mus_max=5,
    tau_mus_up=0.3,
    tau_mus_down=0.3,
    c=.050,
    r=5,
    peep=5,
    dp=5
):
    """
    This function simulates ventilator data with occlusion manuevers.

    :param t_p_occs: Timing of occlusions (s)
    :type t_p_occs: float
    :param t_start: start time
    :type t_start: float
    :param t_end: end time
    :type t_end: float
    :param fs_vent: ventilator sampling rate (Hz)
    :type fs_vent: int
    :param rr: respiratory rate (/min)
    :type rr: float
    :param ie_ratio: Ratio between inspiratory and expiratory time
    :type ie_ratio: float
    :param p_mus_max: Maximal respiratory muscle pressure (cmH2O)
    :type p_mus_max: float
    :param tau_mus_up: Muscle contraction time constant (s)
    :type tau_mus_up: float
    :param tau_mus_down: Muscle release time constant (s)
    :type tau_mus_down: float
    :param c: Respiratory system compliance (L/cmH2O)
    :type c: float
    :param r: Respiratory system resistance (cmH2O/L/s)
    :type r: float
    :param peep: Positive end-expiratory pressure (cmH2O)
    :type peep: float
    :type dp: float
    :param dp: Driving pressure above PEEP (cmH2O)

    :returns: y_vent: np.array([p_vent, f_vent, v_vent])
    :rtype: ~np.array
    """
    # Parse input parameters
    ie_fraction = ie_ratio/(ie_ratio + 1)

    # Time axis
    t_vent = np.array(
        [i/fs_vent for i in range(int(t_start*fs_vent), int(t_end*fs_vent))])

    # Reference signal/Pattern generator
    occs_times = np.array(t_p_occs)

    t_occs = np.floor(occs_times*rr/60)*60/rr
    p_mus_block = (signal.square(t_vent*rr/60*2*np.pi + 0.1, ie_fraction)+1)/2

    # Simulate up- and downslope dynamics of respiratory muscle pressure
    pattern_gen_mus = np.zeros((len(t_vent),))
    for i in range(1, len(t_vent)):
        if (p_mus_block[i-1]-pattern_gen_mus[i-1]) > 0:
            pattern_gen_mus[i] = (pattern_gen_mus[i-1]
                                  + (p_mus_block[i-1]-pattern_gen_mus[i-1])
                                  / (tau_mus_up * fs_vent))
        else:
            pattern_gen_mus[i] = (pattern_gen_mus[i-1]
                                  + (p_mus_block[i-1] - pattern_gen_mus[i-1])
                                  / (tau_mus_down * fs_vent))

    p_mus = p_mus_max * pattern_gen_mus

    # Simulate up- and downslope dynamics of airway pressure
    p_block = dp * (signal.square(t_vent*rr/60*2*np.pi, ie_fraction)+1)/2
    tau_dp_up = 10
    tau_dp_down = 5

    p_noise = np.random.normal(0, 2, size=(len(t_vent), ))
    p_noise_series = pd.Series(p_noise)
    p_noise_ma = p_noise_series.rolling(fs_vent, min_periods=1,
                                        center=True).mean().values

    p_dp = -p_mus
    for i in range(1, len(t_vent)):
        dp_step = p_block[i-1]-(p_dp[i-1] + p_noise_ma[i-1])
        if np.any((((t_occs*fs_vent)-i) <= 0)
                  & ((((t_occs+60/rr)*fs_vent+1)-i) > 0)):
            # Occlusion pressure results into negative airway pressure:
            dp_step = (-np.mean(p_mus[i-int(1*fs_vent/2):int(i-1)])
                       - p_dp[i-1])
            p_dp[i] = p_dp[i-1]+dp_step/(tau_dp_up)
        elif (p_block[i-1]-p_dp[i-1]) > 0:
            p_dp[i] = p_dp[i-1]+dp_step/tau_dp_up
        else:
            p_dp[i] = p_dp[i-1]+dp_step/tau_dp_down

    p_vent = peep + p_dp

    # Calculate flows and volumes from equation of motion:
    v_dot_vent = np.zeros((len(t_vent),))
    v_vent = np.zeros((len(t_vent),))
    for i in range(len(t_vent)-1):
        if np.any((((t_occs*fs_vent)-i-1) <= 0)
                  & ((((t_occs+60/rr)*fs_vent)-i) > 0)):
            # During occlusion manoeuvre: flow and volume are zero
            v_dot_vent[i+1] = 0
            v_vent[i+1] = 0
        else:
            v_dot_vent[i+1] = ((p_dp[i] + p_mus[i]) - v_vent[i] / c)/r
            v_vent[i+1] = v_vent[i] + v_dot_vent[i+1] * 1/fs_vent

    y_vent = np.vstack((p_vent, v_dot_vent, v_vent))
    return y_vent

def respiratory_pattern_generator(
    t_end=7*60,
    fs_emg=2048,
    rr=22,
    ie_ratio=1/2,
    t_p_occs=None,
):
    """
    This function simulates an on/off respiratory muscle activiation pattern
    for generating a synthetic EMG.
    :param t_end: end time
    :type t_end: float
    :param fs_emg: emg sampling rate (Hz)
    :type fs_emg: int
    :param rr: respiratory rate (/min)
    :type rr: float
    :param ie_ratio: Ratio between inspiratory and expiratory time
    :type ie_ratio: float
    :param t_p_occs: Timing of occlusions (s)
    :type t_p_occs: float
    :returns respiratory_pattern: The simulated on/off respiratory muscle
    pattern.
    :rtype respiratory_pattern: ~np.array[float]
    """
    ie_fraction = ie_ratio/(ie_ratio + 1)
    if t_p_occs is None:
        t_occs = np.array([])
    else:
        t_occs = np.floor(np.array(t_p_occs)*rr/60)*60/rr

    for _, t_occ in enumerate(t_occs):
        if t_end < (t_occ + 60/rr):
            print('t=' + str(t_occ) + ':t_occ'
                  + 'should be at least a full respiratory cycle from t_end.')
    # time axis
    t_emg = np.array(
        [i/fs_emg for i in range(0, int(t_end*fs_emg))]
    )

    # reference signal pattern generator
    respiratory_pattern = (
        signal.square(t_emg*rr/60*2*np.pi + 0.5, ie_fraction)+1)/2
    for _, t_occ in enumerate(t_occs):
        i_occ = int(t_occ*fs_emg)
        blocker = np.arange(int(fs_emg*60/rr)+1)/fs_emg*rr/60*2*np.pi
        squared_wave = (signal.square(blocker, ie_fraction)+1)/2
        respiratory_pattern[i_occ:i_occ+int(fs_emg*60/rr)+1] = squared_wave
    return respiratory_pattern

def simulate_muscle_dynamics(
    block_pattern,
    fs_emg=2048,
    tau_mus_up=0.3,
    tau_mus_down=0.3,
):
    """
    This function simulates an respiratory muscle activation dynamics for
    generating a synthetic EMG.
    :param block_pattern: Simulated on/off respiratory muscle pattern.
    :type block_pattern: ~np.array[float]
    :param tau_mus_up: Muscle contraction time constant (s)
    :type tau_mus_up: float
    :param tau_mus_down: Muscle relaxation time constant (s)
    :type tau_mus_down: float

    :returns: muscle_activation: The simulated muscle activation pattern.
    :rtype: ~np.array[float]
    """
    # simulate up- and downslope dynamics of EMG
    muscle_activation = np.zeros((len(block_pattern),))
    for i in range(1, len(block_pattern)):
        pat = muscle_activation[i-1]
        if (block_pattern[i-1]-pat) > 0:
            muscle_activation[i] = pat + ((block_pattern[i-1] - pat) /
                                        (tau_mus_up * fs_emg))
        else:
            muscle_activation[i] = pat + ((block_pattern[i-1] - pat) /
                                        (tau_mus_down * fs_emg))
    return muscle_activation


def simulate_emg(
    muscle_activation,
    fs_emg=2048,
    emg_amp=5,
    drift_amp=100,
    noise_amp=2
):
    """
    This function simulates an surface respiratory emg based on the provided
    `muscle_activation` and adds noise and drift to the signal. No ecg 
    component is included, but can be added later.

    :param muscle_activation: The muscle activation pattern
    :type muscle_activation: float
    :param emg_amp: Approximate EMG-RMS amplitude (uV)
    :type emg_amp: float
    :param drift_amp: Approximate drift RMS amplitude (uV)
    :type drift_amp: float
    :param noise_amp: Approximate baseline noise RMS amplitude (uV)
    :type noise_amp: float

    :returns: emg_raw: The raw synthetic EMG without the ECG added.
    :rtype: ~np.array[float]
    """
    n_samp = len(muscle_activation)
    # make respiratory EMG component
    part_emg = muscle_activation * np.random.normal(
        0, 2, size=(n_samp, ))

    # make noise and drift components
    part_noise = np.random.normal(0, 2*noise_amp, size=(n_samp, ))
    part_drift = np.zeros((n_samp,))

    f_high = 0.05
    white_noise = np.random.normal(0,
                                   drift_amp,
                                   size=(n_samp+int(1/f_high)*fs_emg, ))
    part_drift_tmp = filt.emg_lowpass_butter(
        white_noise, f_high, fs_emg, order=3)
    part_drift = part_drift_tmp[int(1/f_high)*fs_emg:] / f_high

    # mix channels, could be remixed with an ecg
    emg_raw = emg_amp * part_emg + part_drift + part_noise
    return emg_raw


def make_realistic_syn_emg(loaded_ecg, n_emg):
    """
    This function makes realistic synthetic respiratory EMG data.
    :param loaded_ecg: synthetic emg/s as numpy array
    :type loaded_ecg: np.array
    :param n_emg: number of EMGs to simulate
    :type n_emg: int

    :returns: list_emg_raw
    :rtype: list
    """
    list_emg_raw = []
    n_emg = int(n_emg)  # added for cli
    for _ in list(range(n_emg)):
        emg = simulate_emg_with_occlusions(
            t_start=0,
            t_end=7*60,
            fs_emg=2048,   # hertz
            rr=22,         # respiratory rate /min
            ie_ratio=1/2,  # ratio btw insp + expir phase
            tau_mus_up=0.3,
            tau_mus_down=0.3,
            t_p_occs=[365, 381, 395]
        )
        emg1 = emg[:307200]
        emg2 = emg[:307200]
        emg3 = emg[:307200]
        emg_stack = np.vstack((emg1, emg2))
        emg_stack = np.vstack((emg_stack, emg3))
        heart_line = random.randint(0, 9)
        one_line_ecg = loaded_ecg[heart_line]
        t_emg = np.zeros((3, emg_stack.shape[1]))
        ecg_out = np.array(200*one_line_ecg, dtype='float64')
        t_emg[0] = ecg_out + np.array(0.05 * emg_stack[0], dtype='float64')
        t_emg[1] = ecg_out + np.array(4 * emg_stack[1], dtype='float64')
        t_emg[2] = ecg_out + np.array(8 * emg_stack[2], dtype='float64')
        list_emg_raw.append(t_emg)
    return list_emg_raw
