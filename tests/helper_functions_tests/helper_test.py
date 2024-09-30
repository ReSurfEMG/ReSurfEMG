"""sanity tests for the preprocessing functions, including filtering,
ecg removal and envelope calculation"""

import unittest
import os
from math import pi
import numpy as np
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.preprocessing.filtering  import emg_bandpass_butter
from resurfemg.helper_functions.math_operations import scale_arrays
from resurfemg.helper_functions.math_operations import zero_one_for_jumps_base
from resurfemg.helper_functions.math_operations import derivative


sample_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(
        __file__)))),
    'test_data',
    'emg_data_synth_quiet_breathing.Poly5',
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


class TestDerivative(unittest.TestCase):
    def test_derivative(self):
        fs = 100
        t = np.array([i/fs for i in range(fs*1)])
        y_t = np.sin(t*2*pi)
        dy_dt_ref = 2*pi*np.cos(t*2*pi)[:-1]
        dy_dt_fun = derivative(y_t, fs)
        error = np.sum(np.abs((dy_dt_ref-dy_dt_fun)))/(
            np.max(np.abs(dy_dt_ref))*len(t)-1)

        self.assertLess(error, 0.05)

if __name__ == '__main__':
    unittest.main()
