"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions run machine learning algorithms on arrays
over them.
"""


# basic ds
import numpy as np
# basic system
import os
import glob
# math and signals
from scipy.stats import entropy
# ml stuff
from sklearn.preprocessing import StandardScaler
import joblib
from resurfemg import helper_functions as hf


def save_ml_output(arrays, out_fname, force):
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
    np.save(out_fname, arrays, allow_pickle=False)


def applu_model(
    arrays_folder,
    model_file,
    output_folder,
    features=['mean', 'entropy'],
):
    """
    This function applies an ML model over a bunch of arrays.
    It is meant to be used by the command line interface.
    """
    file_directory_list = glob.glob(
        os.path.join(arrays_folder, '**/*.npy'),
        recursive=True,
    )
    model = joblib.load(model_file)
    for array in file_directory_list:
        array_np = np.load(array)
        our_emg_processed = abs(array_np)
        index_ml_hold = []
        predictions_made = []
        holder = []
        sc = StandardScaler()
        if features == ['mean', 'entropy']:

            for slice in hf.slices_jump_slider(our_emg_processed, 1000, 1):
                mean_feature = slice.mean()
                entropy_feature = entropy(slice)
                holder.append(slice)
                ml_index_test = [mean_feature, entropy_feature]

                index_ml_hold.append(ml_index_test)
            X_test_live = index_ml_hold
            sc.fit(np.load('ml_extras/x_trainer_for_scale.npy'))
            X_test_live = sc.transform(X_test_live)
            y_pred = model.predict(X_test_live)
            shifter = np.zeros(500) + 3
            shifted_pred = np.hstack((shifter, y_pred))
            shifted_ended_pred = np.hstack((shifted_pred, shifter))
            array_and_pred = np.vstack((array_np, shifted_ended_pred))
            rel_fname = os.path.relpath(array, arrays_folder)
            out_fname = os.path.join(output_folder, rel_fname)
            save_ml_output(array_and_pred, out_fname, force=False)
