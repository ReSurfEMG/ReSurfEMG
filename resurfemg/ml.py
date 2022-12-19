"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to work with arrays, and run machine learning algorithms
over them.
"""


import os

#basic ds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#basic system
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
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import confusion_matrix
import joblib

# here we will import our models ,
# apply them over the array
# then output both the array and the decision array



