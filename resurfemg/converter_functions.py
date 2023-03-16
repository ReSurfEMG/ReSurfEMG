"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to work with various EMG file types
from various hardware/software combinations, and convert them down to
an array that can be further processed with helper_functions or other modules.
"""


import os
import glob
import pandas as pd
import numpy as np
import scipy.io as sio
from resurfemg.tmsisdk_lite import Poly5Reader


def poly5unpad(to_be_read):
    """This function converts a Poly5 read into an array without
    padding. Note there is a quirk in the python Poly5 interface that
    pads with zeros on the end.

    :param to_be_read: Filename of python read Poly5
    :type to_be_read: str

    :returns: unpadded array
    :rtype: ~numpy.ndarray
    """
    read_object = Poly5Reader(to_be_read)
    sample_number = read_object.num_samples
    unpadded = read_object.samples[:, :sample_number]
    return unpadded


def matlab5_jkmn_to_array(file_name):
    """
    This file reads matlab5 files as produced in the Jonkman
    laboratory, on the Biopac system
    and returns arrays in the format and shape
    our functions, those in helper_functions work on.

    :param file_name: Filename of matlab5 files
    :type file_name: str

    :returns: arrayed
    :rtype: ~numpy.ndarray
    """
    file = sio.loadmat(file_name, mdict=None, appendmat=False)
    arrayed = np.rot90(file['data_emg'])
    output_copy = arrayed.copy()
    arrayed[4] = output_copy[0]
    arrayed[3] = output_copy[1]
    arrayed[1] = output_copy[3]
    arrayed[0] = output_copy[4]
    return arrayed


def csv_from_jkmn_to_array(file_name):
    """
    This function takes a file from the Jonkman
    lab in csv format and changes it
    into the shape the library functions work on.

    :param file_name: Filename of csv files
    :type file_name: str

    :returns: arrayed
    :rtype: ~numpy.ndarray
    """
    file = pd.read_csv(file_name)
    new_df = (
        file.T.reset_index().T.reset_index(drop=True)
        .set_axis([f'lead.{i+1}' for i in range(file.shape[1])], axis=1)
    )
    arrayed = np.rot90(new_df)
    arrayed = np.flipud(arrayed)
    return arrayed


def save_j_as_np(
    file_directory,
    made,
    force=False
):
    """
    This is an implementation of the save_j_as_np_single function in the
    same module which can be run from the commmand-line cli module.

    :param file_directory: the directory with EMG files
    :type file_directory: str
    :param processed: the output directory
    :type processed: str
    :param our_chosen_leads: the leads selected for the pipeline to run over
    :type our_chosen_leads: list

    """
    file_directory_list = glob.glob(
        os.path.join(file_directory, '**/*.csv'),
        recursive=True,
    )
    for file_name in file_directory_list:
        file = pd.read_csv(file_name)
        new_df = (
            file.T.reset_index().T.reset_index(drop=True)
            .set_axis([f'lead.{i+1}' for i in range(file.shape[1])], axis=1)
        )
        arrayed = np.rot90(new_df)
        arrayed = np.flipud(arrayed)

        rel_fname = os.path.relpath(file_name, file_directory)
        out_fname = os.path.join(made, rel_fname)
        # check the directory does not exist
        if not (os.path.exists(made)):
            # create the directory you want to save to
            os.mkdir(made)

        np.save(out_fname, arrayed)


def save_j_as_np_single(file_name):
    """
    This function takes a file in csv format
    where teh sequence is top to bottom
    and changes it
    into the shape the library functions work on,
    then saves it as a numpy file

    :param file_name: Filename of csv files
    :type file_name: str

    :returns: arrayed
    :rtype: ~numpy.ndarray
    """
    file = pd.read_csv(file_name)
    new_df = (
        file.T.reset_index().T.reset_index(drop=True)
        .set_axis([f'lead.{i+1}' for i in range(file.shape[1])], axis=1)
    )
    arrayed = np.rot90(new_df)
    arrayed = np.flipud(arrayed)
    np.save(file_name, arrayed)
    return arrayed


def poly_dvrman(file_name):
    """
    This is a function to read in Duiverman type Poly5 files,
    which has 18 layers/pseudo-leads,
    and return an array of the twelve  unprocessed leads
    for further pre-processing. The leads eliminated
    were RMS calculated on other leads (leads 6-12).
    The expected organization returned is from leads 0-5
    EMG data, then the following leads
    # 6 Paw: airway pressure (not always recorded)
    # 7 Pes: esophageal pressure (not always recorded)
    # 8 Pga: gastric pressure (not always recorded)
    # 9 RR: respiratory rate I guess (very unreliable)
    # 10 HR: heart rate
    # 11 Tach: number of breath (not reliable)

    :param file_name: Filename of Poly5 Duiverman type file
    :type file_name: str

    :returns: samps
    :rtype: ~numpy.ndarray
    """
    data_samples = Poly5Reader(file_name)
    samps = np.vstack([data_samples.samples[:6], data_samples.samples[12:]])
    return samps


def dvrmn_csv_to_array(file_name):
    """
    This transformed an already preprocessed csv from the Duiverman lab
    into an EMG in the format our other functions can work on it. Note that
    some preprocessing steps are already applied so pipelines may
    need adjusting.

    :param file_name: Filename of csv file
    :type file_name: str

    :returns: arrayed
    :rtype: ~numpy.ndarray
    """
    file = pd.read_csv(file_name)
    new_df = file.drop(['Events', 'Time'], axis=1)
    arrayed = np.rot90(new_df)
    arrayed = np.flipud(arrayed)
    return arrayed


def dvrmn_csv_freq_find(file_name):
    """
    This is means to extract the frequency of a Duiverman
    type csv of EMG. Note this data may be resampled down by a
    factor of 10.

    :param file_name: Filename of csv file
    :type file_name: str

    :returns: freq
    :rtype: int
    """
    file = pd.read_csv(file_name)
    sample_points = len(file)
    time_string = file['Time'][sample_points - 1]
    seconds = float(time_string[5:10])
    minutes = float(time_string[2:4])
    hours = int(time_string[0:1])
    sum_time = (hours*3600) + (minutes*60) + seconds
    freq = round(sample_points/sum_time)
    return freq
