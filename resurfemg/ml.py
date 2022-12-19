"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions run machine learning algorithms on arrays
over them.
"""


# basic ds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# basic system
import sys
import os
import glob

# math and signals
import math
from scipy.stats import entropy
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
# # demo stuff
# import ipywidgets as widgets
# import seaborn

# ml stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
# from sklearn.metrics import confusion_matrix
import joblib

# here we will import our models ,
# apply them over the arrays
# then output both the array and the decision array
# into a list


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
    np.save(out_fname, array, allow_pickle=False)


def applu_model(arrays_folder, model_file, output_folder):
    """
    This function applies an ML model over a bunch of arrays.
    """
    file_directory_list = glob.glob(
        os.path.join(arrays_folder, '**/*'),
        recursive=True,
    )
    model = joblib.load(model_file)
    arrays_and_pred = []
    for array in file_directory_list:
        y_pred = model.predict(array)
        arrays_and_pred.append(array, y_pred)

    # OK- then turn it into a 2 lead array, then save as below
    # rel_fname = os.path.relpath(file, file_directory)
    # out_fname = os.path.join(processed, rel_fname)
    # save_preprocessed(array, out_fname, force)
