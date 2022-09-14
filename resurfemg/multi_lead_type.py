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
import copy
import builtins
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA


def compute_ICA_two_comp_selective(
    emg_samples,
    use_all_leads=True,
    desired_leads=[0, 2],
):
    """A function that performs an independant component analysis
    (ICA) meant for EMG data that includes stacked arrays,
    there should be at least two arrays but there can be more.

    :param emg_samples: Original signal array with three or more layers
    :type emg_samples: ~numpy.ndarray
    :param use_all_leads: True if all leads used, otherwise specify leads
    :type use_all_leads: bool
    :param desired_leads: list of leads to use starting from 0
    :type desired_leads: list

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
    for i in all_component_numbers:
        list_to_c.append(emg_samples[i])
    X = np.column_stack(list_to_c)
    ica = FastICA(n_components=2)
    S = ica.fit_transform(X)
    component_0 = S.T[0]
    component_1 = S.T[1]
    return component_0, component_1
