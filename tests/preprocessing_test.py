#sanity tests for the preprocessing functions, including filtering,
#ecg removal and envelope calculation


import unittest
import os
import scipy
import numpy as np
from resurfemg.preprocessing.filtering import emg_bandpass_butter
from resurfemg.preprocessing.filtering import emg_bandpass_butter_sample
from resurfemg.preprocessing.filtering import bad_end_cutter
from resurfemg.preprocessing.filtering import bad_end_cutter_better
from resurfemg.preprocessing.filtering import bad_end_cutter_for_samples
from resurfemg.preprocessing.filtering import notch_filter
from resurfemg.preprocessing.filtering import emg_lowpass_butter
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.preprocessing.ecg_removal import compute_ICA_two_comp_selective
# from resurfemg.multi_lead_type import compute_ICA_n_comp
# from resurfemg.multi_lead_type import compute_ICA_n_comp_selective_zeroing
from resurfemg.preprocessing.ecg_removal import compute_ica_two_comp
from resurfemg.preprocessing.ecg_removal import compute_ica_two_comp_multi
from resurfemg.preprocessing.ecg_removal import pick_lowest_correlation_array
from resurfemg.preprocessing.ecg_removal import pick_more_peaks_array
from resurfemg.preprocessing.ecg_removal import gating
from resurfemg.preprocessing.ecg_removal import pick_highest_correlation_array_multi
from resurfemg.preprocessing.envelope import naive_rolling_rms
from resurfemg.preprocessing.envelope import vect_naive_rolling_rms
from resurfemg.preprocessing.ecg_removal import find_peaks_in_ecg_signal

sample_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'not_pushed',
    'Test_lung_data',
    '2022-05-13_11-51-04',
    '002',
    'EMG_recording.Poly5',
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

class TestComponentPickingMethods(unittest.TestCase):

    def test_pick_more_peaks_array(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered[1]= sample_emg_filtered[0]*1.5
        sample_emg_filtered[2]= sample_emg_filtered[0]*1.7
        components = compute_ica_two_comp(sample_emg_filtered)
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

    def test_pick_highest_correlation_array_multi(self):
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
            sum(components[emg]),
            sum(sample_emg_filtered[1]),
        )

    def test_find_peaks_in_ecg_signal(self):
        samp_array = np.array([0,0,0,0,10,0,0,0,10,0,0,0,4,0,0,])
        peaks = find_peaks_in_ecg_signal(samp_array, lower_border_percent=50)
        self.assertEqual(
            len(peaks),
            2,
        )
class TestPickingMethods(unittest.TestCase):


    # def test_compute_ICA_n_comp(self):
    #     sample_read= Poly5Reader(sample_emg)
    #     sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
    #     sample_emg_filtered[1] = sample_emg_filtered[0]*1.5
    #     sample_emg_filtered[2] = sample_emg_filtered[0]*1.7
    #     doubled = np.vstack((sample_emg_filtered,sample_emg_filtered))
    #     no_zeros = compute_ICA_n_comp(doubled, 1)
    #     self.assertEqual(
    #         (no_zeros.shape[0]),
    #         6,
    #     )

    # def test_compute_ICA_n_comp_selective_zeroing(self):
    #     sample_read= Poly5Reader(sample_emg)
    #     sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
    #     sample_emg_filtered[1] = sample_emg_filtered[0]*1.5
    #     sample_emg_filtered[2] = sample_emg_filtered[0]*1.7
    #     doubled = np.vstack((sample_emg_filtered,sample_emg_filtered))
    #     with_zeros = compute_ICA_n_comp_selective_zeroing(doubled, 1)
    #     self.assertEqual(
    #         (with_zeros.shape[0]),
    #         6,
    #     )

    def test_compute_ICA_two_comp(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered[1]= sample_emg_filtered[0]*1.5
        sample_emg_filtered[2]= sample_emg_filtered[0]*1.7
        components = compute_ica_two_comp(sample_emg_filtered)
        self.assertEqual(
            (len(components[1])),
            len(components[0]) ,
        )

    def test_compute_ICA_two_comp_multi(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(sample_read, 1, 10)
        t = np.array([x/2048 for x in range(len(sample_emg_filtered[0]))])
        sample_emg_filtered[1]= sample_emg_filtered[0]*1.5
        sample_emg_filtered[2]= sample_emg_filtered[0]*1.7+np.sin(
            t * 2 * np.pi)
        components = compute_ica_two_comp_multi(sample_emg_filtered)
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

class TestGating(unittest.TestCase):
    sample_read= Poly5Reader(sample_emg)
    sample_emg_filtered = -emg_bandpass_butter(sample_read, 1, 500)
    sample_emg_filtered = sample_emg_filtered[:30*2048]
    ecg_peaks, _  = scipy.signal.find_peaks(sample_emg_filtered[0, :])

    def test_gating_method_0(self):
        ecg_gated_0 = gating(self.sample_emg_filtered[0, :], self.ecg_peaks,
                             gate_width=205, method=0)

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
        ecg_gated_2 = gating(self.sample_emg_filtered[0, :], self.ecg_peaks,
                             gate_width=205, method=2)

        self.assertEqual(
            (len(self.sample_emg_filtered[0])),
            len(ecg_gated_2) ,
        )

    def test_gating_method_2_no_prior_segment(self):
        ecg_gated_2 = gating(
            self.sample_emg_filtered[0, :], [100], gate_width=205, method=2)

        self.assertFalse(
            np.isnan(np.sum(ecg_gated_2))
        )
    def test_gating_method_3(self):
        height_threshold = np.max(self.sample_emg_filtered)/2
        ecg_peaks, _  = scipy.signal.find_peaks(
            self.sample_emg_filtered[0, :10*2048-1], 
            height=height_threshold)

        ecg_gated_3 = gating(self.sample_emg_filtered[0, :10*2048], ecg_peaks,
                             gate_width=205, method=3)

        self.assertEqual(
            (len(self.sample_emg_filtered[0, :10*2048])),
            len(ecg_gated_3) ,
        )
