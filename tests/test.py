#sanity tests for the resurfemg library


import unittest
import os
import numpy as np
import scipy
from tempfile import TemporaryDirectory
import json
from unittest import TestCase, main

# tmsisdk_lite
from resurfemg.tmsisdk_lite import Poly5Reader
# converter_functions 
from resurfemg.converter_functions import poly5unpad
from resurfemg.config import hash_it_up_right_all
# multi_lead_type
from resurfemg.multi_lead_type import compute_ICA_n_comp
from resurfemg.multi_lead_type import compute_ICA_n_comp_selective_zeroing
from resurfemg.multi_lead_type import compute_ICA_two_comp_selective
from resurfemg.multi_lead_type import pick_highest_correlation_array_multi
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
from resurfemg.helper_functions import area_under_curve
from resurfemg.helper_functions import simple_area_under_curve
from resurfemg.helper_functions import find_peak_in_breath
from resurfemg.helper_functions import distance_matrix
from resurfemg.helper_functions import emg_lowpass_butter
from resurfemg.helper_functions import find_peaks_in_ecg_signal
# config
from resurfemg.config import Config
from resurfemg.config import make_realistic_syn_emg
# ml
from resurfemg.ml import save_ml_output

sample_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'not_pushed',
    'Test_lung_data',
    '2022-05-13_11-51-04',
    '002',
    'EMG_recording.Poly5',
)


class TestConverterMethods(unittest.TestCase):

    def test_poly5unpad(self):
        reading =Poly5Reader(sample_emg)
        unpadded= poly5unpad(sample_emg)
        unpadded_line = unpadded[0]
        self.assertEqual(len(unpadded_line), reading.num_samples)

    def Poly5Reader(self):
        reading =Poly5Reader(sample_emg)
        self.assertEqual(reading.num_channels, 3)


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

    def test_pick_highest_correlation_array(self):
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
        components = np.row_stack((sample_emg_filtered[1], sample_emg_filtered[2]))
        emg = pick_highest_correlation_array_multi(components, sample_emg_filtered[0])
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
    
    def test_emg_lowpass_butter(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_lowpass_butter(sample_read.samples, 5, 2048)
        self.assertEqual(
            (len(sample_emg_filtered[0])),
            len(sample_read.samples[0]) ,
        )

    def test_notch_filter(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = notch_filter(sample_read.samples, 2048, 80,2)
        self.assertEqual(
            (len(sample_emg_filtered[0])),
            len(sample_read.samples[0]) ,
        )

    def test_naive_rolling_rms(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = naive_rolling_rms(sample_read.samples[0], 10)
        self.assertNotEqual(
            (len(sample_emg_filtered)),
            len(sample_read.samples[0]) ,
        )
    
    def test_vect_naive_rolling_rms(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = vect_naive_rolling_rms(sample_read.samples[0], 10)
        self.assertNotEqual(
            (len(sample_emg_filtered)),
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

    def test_compute_ICA_n_comp(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered[1] = sample_emg_filtered[0]*1.5
        sample_emg_filtered[2] = sample_emg_filtered[0]*1.7
        doubled = np.vstack((sample_emg_filtered,sample_emg_filtered))
        no_zeros = compute_ICA_n_comp(doubled, 1)
        self.assertEqual(
            (no_zeros.shape[0]),
            6,
        )

    def test_compute_ICA_n_comp_selective_zeroing(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered[1] = sample_emg_filtered[0]*1.5
        sample_emg_filtered[2] = sample_emg_filtered[0]*1.7
        doubled = np.vstack((sample_emg_filtered,sample_emg_filtered))
        with_zeros = compute_ICA_n_comp_selective_zeroing(doubled, 1)
        self.assertEqual(
            (with_zeros.shape[0]),
            6,
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
    def test_working_pipe_multi(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered[1]= sample_emg_filtered[0]*1.5
        sample_emg_filtered[2]= sample_emg_filtered[0]*1.7
        pipelined_0_1 = working_pipe_multi(sample_emg_filtered, (0,1))
        pipelined_0_2 = working_pipe_multi(sample_emg_filtered, (0,2))
        self.assertEqual(
            pipelined_0_1.shape,
            pipelined_0_2.shape,
        )


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
        ecg_gated_1 = gating(
            self.sample_emg_filtered[0, :],
            self.ecg_peaks,
            gate_width=205,
            method=1
        )

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

    def test_gating_method_2_no_prior_segment(self):
        ecg_gated_2 = gating(self.sample_emg_filtered[0, :], [100], gate_width=205, method=2)

        self.assertFalse(
            np.isnan(np.sum(ecg_gated_2))
        )
    
    def test_gating_method_3(self):
        ecg_peaks, _  = scipy.signal.find_peaks(self.sample_emg_filtered[0, :10*2048-1])
        ecg_gated_3 = gating(self.sample_emg_filtered[0, :10*2048], ecg_peaks, gate_width=205, method=3)

        self.assertEqual(
            (len(self.sample_emg_filtered[0, :10*2048])),
            len(ecg_gated_3) ,
        )


class TestMl(unittest.TestCase):


    def test_save_ml_output(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 500)
        with TemporaryDirectory() as td:
            our_path = os.path.join(td,'outy.npy')
            emg_saved = save_ml_output(sample_emg_filtered, our_path , force=True)
            loaded = np.load(our_path)

        self.assertEqual(loaded.max(),sample_emg_filtered.max())


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

    def test_find_peaks_in_ecg_signal(self):
        samp_array = np.array([0,0,0,0,10,0,0,0,10,0,0,0,4,0,0,])
        peaks = find_peaks_in_ecg_signal(samp_array, lower_border_percent=50)
        self.assertEqual(
            len(peaks),
            2,
        )

    def test_count_decision_array(self):
        sample_array= np.array([0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,0])
        counted = count_decision_array(sample_array)
        self.assertEqual(
            counted,
             3,
        )

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

    def test_find_peak_in_breath(self):
        sample_array= np.array(
            [0,0,0,0,1,1,1,5,10,13,5,0,1,1,1,1,0,1,1,1,0,0,0]
        )
        peak =find_peak_in_breath(sample_array,0,20)
        self.assertEqual(
            peak,
            (9,13)
        )
    

class TestConfig(TestCase):

    required_directories = {
        'root_emg_directory',
    }
    required_directories = ['root_emg_directory']

    def test_roots_only(self):
        with TemporaryDirectory() as td:
            same_created_path = os.path.join(td, 'root')
            os.mkdir(same_created_path)
            raw_config = {
                'root_emg_directory': same_created_path,
            }
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)

            # for root in self.required_directories:
            #     os.mkdir(os.path.join(td, root))

            config = Config(config_file)
            assert config.get_directory('root_emg_directory')

    def test_missing_config_path(self):
        try:
            Config('non existent')
        except ValueError:
            pass
        else:
            assert False, 'Didn\'t notify on missing config file'
    
    def test_make_realistic_syn_emg(self):
        x_ecg = np.zeros((10,307200))
        made = make_realistic_syn_emg(x_ecg,2)
        self.assertEqual(len(made),2)
        


if __name__ == '__main__':
    unittest.main()
