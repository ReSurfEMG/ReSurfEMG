#sanity tests for the preprocessing functions, including filtering,
#ecg removal and envelope calculation

import unittest
import numpy as np
import os
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.preprocessing.filtering  import emg_bandpass_butter
from resurfemg.helper_functions.helper_functions import scale_arrays
from resurfemg.helper_functions.helper_functions import zero_one_for_jumps_base
from resurfemg.helper_functions.helper_functions import count_decision_array
from resurfemg.helper_functions.helper_functions import distance_matrix
from resurfemg.helper_functions.helper_functions import relative_levenshtein


sample_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'not_pushed',
    'Test_lung_data',
    '2022-05-13_11-51-04',
    '002',
    'EMG_recording.Poly5',
)


class TestArrayMath(unittest.TestCase):


    def test_scale_arrays(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 500)
        new_emg = scale_arrays(sample_emg_filtered , 3,0)
        self.assertEqual(
            (new_emg.shape),
            (sample_emg_filtered.shape),
        )

    def test_zero_one_for_jumps_base(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 500)
        new_emg = zero_one_for_jumps_base(sample_emg_filtered[0] , sample_emg_filtered[0].mean())
        new_emg = np.array(np.vstack((new_emg, new_emg)))
        self.assertEqual(
            (new_emg.shape[1]),
            (sample_emg_filtered.shape[1]),
        )

    def test_count_decision_array(self):
        sample_array= np.array([0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,0])
        counted = count_decision_array(sample_array)
        self.assertEqual(
            counted,
             3,
        )

    def test_distance_matrix(self):
        sample_array_a= np.array(
            [0,0,0,0,1,1,1,5,10,10,5,0,1,1,1,1,0,1,1,1,0,0,0]
        )
        sample_array_b= np.array(
            [0,0,0,0,1,1,1,5,1,1,5,0,1,1,1,1,0,1,1,1,0,0,0]
        )
        matrix = distance_matrix(sample_array_a,sample_array_b)
        self.assertEqual(
            matrix.shape,
            (1,6),
        )


class TestVentCompareMethods(unittest.TestCase):

    def test_relative_levenshtein(self):
        array1 = np.array([1,0,1,0,1,0])
        array2 = np.array([1,0,1,0,1,0])
        array3 = np.array([1,0,1,0,1,0])
        our_result12 = (relative_levenshtein(array1,array2))
        our_result13 = (relative_levenshtein(array1,array3))
        self.assertEqual(our_result12, our_result13)

