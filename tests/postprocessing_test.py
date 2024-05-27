"""sanity tests for the postprocessing module of the resurfemg library"""


import unittest
import os
import numpy as np
import scipy

from resurfemg.postprocessing.features import (
    entropy_scipy, pseudo_slope, area_under_curve, simple_area_under_curve, 
    times_under_curve, find_peak_in_breath,variability_maker)
from resurfemg.postprocessing.baseline import (
    moving_baseline, slopesum_baseline)
from resurfemg.postprocessing.event_detection import (
    onoffpeak_baseline_crossing, onoffpeak_slope_extrapolation)

sample_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'not_pushed',
    'Test_lung_data',
    '2022-05-13_11-51-04',
    '002',
    'EMG_recording.Poly5',
)

class TestEntropyMethods(unittest.TestCase):

    def test_entropy_scipy(self):
        sample_array_lo_entropy = [0,0,0,0,0,0,0,0,0,0]
        sample_array_hi_entropy = [0,4,0,5,8,0,12,0,1,0]
        ent_sample_array_lo_entropy = entropy_scipy(sample_array_lo_entropy)
        ent_sample_array_hi_entropy = entropy_scipy(sample_array_hi_entropy)
        self.assertGreater(
            ent_sample_array_hi_entropy ,
            ent_sample_array_lo_entropy ,
        )

class TestVariabilityMethods(unittest.TestCase):

    def test_variability_maker_variance(self):
        sample_array_lo_var = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        sample_array_hi_var = [0,4,0,5,8,0,12,0,1,0,0,9,0,9,0,0,9,6,0]
        var_sample_array_lo_var = variability_maker(sample_array_lo_var, 10)
        var_sample_array_hi_var = variability_maker(sample_array_hi_var, 10)
        self.assertGreater(
            np.sum(var_sample_array_hi_var) ,
            np.sum(var_sample_array_lo_var) ,
        )

    def test_variability_maker_std(self):
        sample_array_lo_var = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        sample_array_hi_var = [0,4,0,5,8,0,12,0,1,0,0,9,0,9,0,0,9,6,0]
        var_sample_array_lo_var = variability_maker(sample_array_lo_var, 10, method='std')
        var_sample_array_hi_var = variability_maker(sample_array_hi_var, 10, method='std')
        self.assertGreater(
            np.sum(var_sample_array_hi_var) ,
            np.sum(var_sample_array_lo_var) ,
        )

class TestArrayMath(unittest.TestCase):

    def test_simple_area_under_curve(self):
        sample_array= np.array(
            [0,0,0,0,1,1,1,-5,10,10,0,0,0,1,1,1,0,1,1,1,0,0,0]
        )
        counted = simple_area_under_curve(sample_array,0,10)
        self.assertEqual(
            counted,
            28,
        )

    def test_area_under_curve(self):
        sample_array= np.array(
            [0,0,0,0,1,1,1,5,10,10,5,0,1,1,1,1,0,1,1,1,0,0,0]
        )
        counted = area_under_curve(sample_array,0,20,70)
        self.assertEqual(
            counted,
            28,
        )

    def test_times_under_curve(self):
        sample_array= np.array(
            [0,1,2,3,1,5,6,-5,8,9,20,11,12,13,4,5,6,1,1,1,0]
        )
        counted = times_under_curve(sample_array,0,20)
        self.assertEqual(
            counted,
            ((10,0.5)),
        )

    def test_pseudo_slope(self):
        test_arr_1 = np.array(
            [0,9,8,7,10,11,13,15,12,16,13,17,18,6,5,4,3,2,1,0,0,0,0,0]
        )
        slope = pseudo_slope(test_arr_1,0,17)
        self.assertEqual(
            slope,
            1.5
        )

    def test_find_peak_in_breath(self):
        sample_array= np.array(
            [0,0,0,0,1,1,1,5,10,13,5,0,1,1,1,1,0,1,1,1,0,0,0]
        )
        peak =find_peak_in_breath(sample_array,0,20)
        self.assertEqual(
            peak,
            (9,13, 13)
        )

    def test_find_peak_in_breath_convy(self):
        sample_array= np.array(
            [0,0,0,0,1,1,1,5,10,13,5,0,1,1,1,1,0,1,1,1,0,0,0]
        )
        peak =find_peak_in_breath(sample_array,0,20,'convy')
        self.assertEqual(
            peak,
            (8,10, 11.5)
        )


class TestBaseline(unittest.TestCase):
    fs = 1000
    t = np.arange(0, 10, 1/fs)
    slow_component = np.sin(2 * np.pi * 0.1 * t)
    fast_component = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    breathing_signal = np.abs(slow_component + fast_component)

    def test_movingbaseline(self):
        sinusbase = moving_baseline(self.breathing_signal,
                                    self.fs,
                                    self.fs//5,
                                    33)
        self.assertEqual(
            (len(self.breathing_signal)),
            len(sinusbase),
    )

    def test_slopesum(self):
        sinusbase, _, _, _ = slopesum_baseline(
            self.breathing_signal,
            self.fs,
            self.fs//5,
            self.fs,
            set_percentile=33,
            augm_percentile=25)

        self.assertEqual(
            (len(self.breathing_signal)),
            len(sinusbase),
            )

class TestEventDetection(unittest.TestCase):
    fs = 1000
    t = np.arange(0, 10, 1/fs)
    slow_component = np.sin(2 * np.pi * 0.1 * t)
    fast_component = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    breathing_signal = np.abs(slow_component + fast_component)

    baseline = 0.5 * np.ones((len(breathing_signal), ))
    treshold = 0
    width = fs // 2
    prominence = 0.5 * (np.nanpercentile(breathing_signal - baseline, 75)
                        + np.nanpercentile(breathing_signal - baseline, 50))

    peak_idxs, _ = scipy.signal.find_peaks(
        breathing_signal,
        height=treshold,
        prominence=prominence,
        width=width)

    def test_baseline_crossing_starts(self):
        _, peak_start_idxs, _ = onoffpeak_baseline_crossing(
             self.breathing_signal, self.baseline, self.peak_idxs)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_start_idxs),
            )

    def test_baseline_crossing_ends(self):
        _, _, peak_end_idxs = onoffpeak_baseline_crossing(
             self.breathing_signal, self.baseline, self.peak_idxs)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_end_idxs),
            )
    
    def test_slope_extrapolate_starts(self):
        peak_start_idxs, _, _, _, _ = onoffpeak_slope_extrapolation(
             self.breathing_signal, self.fs, self.peak_idxs, self.fs//4)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_start_idxs),
            )

    def test_slope_extrapolate_ends(self):
        _, peak_end_idxs, _, _, _ = onoffpeak_slope_extrapolation(
             self.breathing_signal, self.fs, self.peak_idxs, self.fs//4)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_end_idxs),
            )

if __name__ == '__main__':
    unittest.main()
