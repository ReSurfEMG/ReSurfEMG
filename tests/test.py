#sanity tests for the rsemg library


import unittest
import os
import glob
import sys
import numpy as np
import scipy
from tempfile import TemporaryDirectory
from TMSiSDK.file_readers import Poly5Reader


# converter_functions
from resurfemg.converter_functions import poly5unpad
from resurfemg.config import hash_it_up_right_all
# multi_lead_type
# from resurfemg.multi_lead_type import compute_ICA_two_comp_selective
from resurfemg.multi_lead_type import compute_ICA_two_comp_selective
from resurfemg.multi_lead_type import working_pipe_multi
from resurfemg.multi_lead_type import working_pipeline_pre_ml_multi
# helper_functions
from resurfemg.helper_functions import bad_end_cutter
from resurfemg.helper_functions import bad_end_cutter_better
from resurfemg.helper_functions import bad_end_cutter_for_samples
from resurfemg.helper_functions import count_decision_array
from resurfemg.helper_functions import emg_bandpass_butter
from resurfemg.helper_functions import emg_bandpass_butter_sample
from resurfemg.helper_functions import notch_filter
from resurfemg.helper_functions import show_my_power_spectrum
from resurfemg.helper_functions import naive_rolling_rms
from resurfemg.helper_functions import vect_naive_rolling_rms
from resurfemg.helper_functions import pick_more_peaks_array
from resurfemg.helper_functions import pick_lowest_correlation_array
from resurfemg.helper_functions import zero_one_for_jumps_base
from resurfemg.helper_functions import compute_ICA_two_comp
from resurfemg.helper_functions import compute_ICA_two_comp_multi
from resurfemg.helper_functions import working_pipeline_exp
from resurfemg.helper_functions import entropical
from resurfemg.helper_functions import smooth_for_baseline
from resurfemg.helper_functions import smooth_for_baseline_with_overlay
from resurfemg.helper_functions import relative_levenshtein
from resurfemg.helper_functions import gating
from resurfemg.helper_functions import scale_arrays

sample_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'not_pushed',
    'Test_lung_data',
    '2022-05-13_11-51-04',
    '002',
    'EMG_recording.Poly5',
)



class TestDisplayConverterMethods(unittest.TestCase):

    def test_poly5unpad(self):
        reading =Poly5Reader(sample_emg)
        unpadded= poly5unpad(sample_emg)
        unpadded_line = unpadded[0]
        self.assertEqual(len(unpadded_line), reading.num_samples)


class TestHashMethods(unittest.TestCase):

    def test_hash_it_up_right_all(self):
        tempfile1 = 'sample_emg_t.Poly5'
        tempfile2 = 'sample_emg_t.Poly5'
        with TemporaryDirectory() as td:
            with open(os.path.join(td, tempfile1), 'w') as tf:
                tf.write('string')
            with open(os.path.join(td, tempfile2), 'w') as tf:
                tf.write('string')
            self.assertTrue(hash_it_up_right_all(td, '.Poly5').equals(hash_it_up_right_all(td, '.Poly5')))


class TestComponentPickingMethods(unittest.TestCase):

    def test_pick_more_peaks_array(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered[1]= sample_emg_filtered[0]*1.5
        sample_emg_filtered[2]= sample_emg_filtered[0]*1.7
        components = compute_ICA_two_comp(sample_emg_filtered)
        emg = pick_more_peaks_array(components)
        self.assertEqual(
            (len(emg)),
            len(components[0]) ,
        )
    def test_pick_lowest_correlation_array(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered[1] = sample_emg_filtered[0]
        sample_emg_filtered[2] = sample_emg_filtered[0]*0.7
        sample_emg_filtered[2,20] = 0
        sample_emg_filtered[2,40] = 0
        sample_emg_filtered[2,80] = 0
        sample_emg_filtered[2,21] = 100
        sample_emg_filtered[2,42] = 100
        sample_emg_filtered[2,81] = 100
        components = sample_emg_filtered[1], sample_emg_filtered[2]
        emg = pick_lowest_correlation_array(components, sample_emg_filtered[0])
        self.assertEqual(
            sum(emg),
            sum(sample_emg_filtered[2]) ,
        )


class TestFilteringMethods(unittest.TestCase):

    def test_emg_band_pass_butter(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        self.assertEqual(
            (len(sample_emg_filtered[0])),
            len(sample_read.samples[0]) ,
        )
    def test_emg_band_pass_butter_sample(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter_sample(sample_read.samples, 1, 10, 2048)
        self.assertEqual(
            (len(sample_emg_filtered[0])),
            len(sample_read.samples[0]) ,
        )


class TestPickingMethods(unittest.TestCase):

    def test_compute_ICA_two_comp(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered[1]= sample_emg_filtered[0]*1.5
        sample_emg_filtered[2]= sample_emg_filtered[0]*1.7
        components = compute_ICA_two_comp(sample_emg_filtered)
        self.assertEqual(
            (len(components[1])),
            len(components[0]) ,
        )
    def test_compute_ICA_two_comp_multi(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered[1]= sample_emg_filtered[0]*1.5
        sample_emg_filtered[2]= sample_emg_filtered[0]*1.7
        components = compute_ICA_two_comp_multi(sample_emg_filtered)
        self.assertEqual(
            (len(components)),
            2 ,
        )

    def test_compute_ICA_two_comp_selective(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered[1]= sample_emg_filtered[0]*1.5
        sample_emg_filtered[2]= sample_emg_filtered[0]*1.7
        components =  compute_ICA_two_comp_selective(
        sample_emg_filtered,
        use_all_leads=False,
        desired_leads=[0, 1],
        )
        self.assertEqual(
            (len(components)),
            2 ,
        )


class TestPipelineMethods(unittest.TestCase):

    # def test_working_pipeline_exp(self):
    #     sample_read= Poly5Reader(sample_emg_tampered) [# we need to augment a real one here]
    #     pipelined = working_pipeline_exp(sample_read)
    #     self.assertEqual(
    #         pipelined.shape[0],
    #         1 ,
    #     )
    def test_working_pipeline_pre_ml_multi(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered[1]= sample_emg_filtered[0]*1.5
        sample_emg_filtered[2]= sample_emg_filtered[0]*1.7
        pipelined_0_1 = working_pipeline_pre_ml_multi(sample_emg_filtered, (0,1))
        pipelined_0_2 = working_pipeline_pre_ml_multi(sample_emg_filtered, (0,2))
        self.assertEqual(
            pipelined_0_1.shape,
            pipelined_0_2.shape,
        )

        
    #\     sample_read= Poly5Reader(sample_emg)
    #     sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
    #     sample_emg_filtered[1]= sample_emg_filtered[0]*1.5
    #     sample_emg_filtered[2]= sample_emg_filtered[0]*1.7
    #     components = compute_ICA_two_comp_multi(sample_emg_filtered)
    #     self.assertEqual(
        #     (len(components)),
        #     2 ,
        # )


class TestCuttingingMethods(unittest.TestCase):

    def test_emg_bad_end_cutter(self):
        sample_ready= Poly5Reader(sample_emg)
        sample_emg_cut = bad_end_cutter(sample_ready, 1, 10)
        self.assertNotEqual(
            (len(sample_emg_cut[0])),
            len(sample_ready.samples[0]),
        )
    def test_emg_bad_end_cutter_for_samples(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_cut = bad_end_cutter_for_samples(sample_read.samples, 1, 10)
        self.assertNotEqual(
            (len(sample_emg_cut[0])),
            len(sample_read.samples[0]) ,
        )
    def test_emg_bad_end_cutter_better(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_cut = bad_end_cutter_better(sample_read, 1, 10)
        self.assertNotEqual(
            (len(sample_emg_cut[0])),
            len(sample_read.samples[0]) ,
        )

class TestVentCompareMethods(unittest.TestCase):

    def test_relative_levenshtein(self):
        array1 = np.array([1,0,1,0,1,0])
        array2 = np.array([1,0,1,0,1,0])
        array3 = np.array([1,0,1,0,1,0])
        our_result12 = (relative_levenshtein(array1,array2))
        our_result13 = (relative_levenshtein(array1,array3))
        self.assertEqual(our_result12, our_result13)

class TestGating(unittest.TestCase):
    sample_read= Poly5Reader(sample_emg)
    sample_emg_filtered = -emg_bandpass_butter(sample_read, 1, 500)
    sample_emg_filtered = sample_emg_filtered[:30*2048]
    ecg_peaks, _  = scipy.signal.find_peaks(sample_emg_filtered[0, :])

    def test_gating_method_0(self):
        ecg_gated_0 = gating(self.sample_emg_filtered[0, :], self.ecg_peaks, gate_width=205, method=0)

        self.assertEqual(
            (len(self.sample_emg_filtered[0])),
            len(ecg_gated_0) ,
        )

    def test_gating_method_1(self):
        ecg_gated_1 = gating(self.sample_emg_filtered[0, :], self.ecg_peaks, gate_width=205, method=1)

        self.assertEqual(
            (len(self.sample_emg_filtered[0])),
            len(ecg_gated_1) ,
        )
    
    def test_gating_method_2(self):
        ecg_gated_2 = gating(self.sample_emg_filtered[0, :], self.ecg_peaks, gate_width=205, method=2)

        self.assertEqual(
            (len(self.sample_emg_filtered[0])),
            len(ecg_gated_2) ,
        )
    
    def test_gating_method_3(self):
        ecg_peaks, _  = scipy.signal.find_peaks(self.sample_emg_filtered[0, :10*2048-1])
        ecg_gated_3 = gating(self.sample_emg_filtered[0, :10*2048], ecg_peaks, gate_width=205, method=3)

        self.assertEqual(
            (len(self.sample_emg_filtered[0, :10*2048])),
            len(ecg_gated_3) ,
        )


class TestArrayMath(unittest.TestCase):
    
    
    def test_scale_arrays(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 500)
        new_emg = scale_arrays(sample_emg_filtered , 3,0)
        print(new_emg)
        self.assertEqual(
            (new_emg.shape),
            (sample_emg_filtered.shape) ,
        )

if __name__ == '__main__':
    unittest.main()
