"""sanity tests for the preprocessing.ecg_removal functions"""

import unittest
import os
import scipy
import numpy as np

from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.preprocessing import filtering as filt
from resurfemg.preprocessing import ecg_removal as ecg_rm

sample_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(
        __file__)))),
    'test_data',
    'emg_data_synth_quiet_breathing.Poly5',
)
synth_pocc_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(
        __file__)))),
    'test_data',
    'emg_data_synth_quiet_breathing.Poly5',
)


class TestComponentPickingMethods(unittest.TestCase):

    def test_pick_more_peaks_array(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = filt.emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered = sample_emg_filtered[:, :30*2048]
        t = np.array([x/2048 for x in range(len(sample_emg_filtered[0]))])
        sample_emg_filtered[1] = sample_emg_filtered[0]*1.5
        sample_emg_filtered = np.concatenate(
            (sample_emg_filtered,
             [sample_emg_filtered[0]*1.7 + np.sin(t * 2 * np.pi)]), axis=0)
        components = ecg_rm.compute_ica_two_comp(sample_emg_filtered)
        emg = ecg_rm.pick_more_peaks_array(components)
        self.assertEqual(
            (len(emg)),
            len(components[0]) ,
        )

    def test_pick_lowest_correlation_array(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = filt.emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered = sample_emg_filtered[:, :30*2048]
        t = np.array([x/2048 for x in range(len(sample_emg_filtered[0]))])
        sample_emg_filtered[1] = sample_emg_filtered[0]
        sample_emg_filtered = np.concatenate(
            (sample_emg_filtered,
             [sample_emg_filtered[0]*0.7 + np.sin(t * 2 * np.pi)]), axis=0)
        sample_emg_filtered[2,20] = 0
        sample_emg_filtered[2,40] = 0
        sample_emg_filtered[2,80] = 0
        sample_emg_filtered[2,21] = 100
        sample_emg_filtered[2,42] = 100
        sample_emg_filtered[2,81] = 100
        components = sample_emg_filtered[1], sample_emg_filtered[2]
        emg = ecg_rm.pick_lowest_correlation_array(
            components, sample_emg_filtered[0])
        self.assertEqual(
            sum(emg),
            sum(sample_emg_filtered[2]) ,
        )

    def test_pick_highest_correlation_array_multi(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = filt.emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered = sample_emg_filtered[:, :30*2048]
        t = np.array([x/2048 for x in range(len(sample_emg_filtered[0]))])
        sample_emg_filtered[1] = sample_emg_filtered[0]
        sample_emg_filtered = np.concatenate(
            (sample_emg_filtered,
             [sample_emg_filtered[0]*1.7 + np.sin(t * 2 * np.pi)]), axis=0)
        sample_emg_filtered[2,20] = 0
        sample_emg_filtered[2,40] = 0
        sample_emg_filtered[2,80] = 0
        sample_emg_filtered[2,21] = 100
        sample_emg_filtered[2,42] = 100
        sample_emg_filtered[2,81] = 100
        components = np.vstack((sample_emg_filtered[1],
                                sample_emg_filtered[2]))
        emg = ecg_rm.pick_highest_correlation_array_multi(
            components, sample_emg_filtered[0])
        self.assertEqual(
            sum(components[emg]),
            sum(sample_emg_filtered[1]),
        )

    def test_find_peaks_in_ecg(self):
        samp_array = np.array([0,0,0,0,10,0,0,0,10,0,0,0,4,0,0,])
        peaks = ecg_rm.find_peaks_in_ecg(samp_array, lower_border_percent=50)
        self.assertEqual(
            len(peaks),
            2,
        )
class TestPickingMethods(unittest.TestCase):
    def test_compute_ica_two_comp(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = filt.emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered = sample_emg_filtered[:, :30*2048]
        t = np.array([x/2048 for x in range(len(sample_emg_filtered[0]))])
        sample_emg_filtered[1] = sample_emg_filtered[0]*1.5
        sample_emg_filtered = np.concatenate(
            (sample_emg_filtered,
             [sample_emg_filtered[0] * 1.7 + np.sin(t * 2 * np.pi)]), axis=0)
        components = ecg_rm.compute_ica_two_comp(sample_emg_filtered)
        self.assertEqual(
            (len(components[1])),
            len(components[0]) ,
        )

    def test_compute_ica_two_comp_multi(self):
        sample_read= Poly5Reader(sample_emg)
        sample_emg_filtered = filt.emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered = sample_emg_filtered[:, :30*2048]
        t = np.array([x/2048 for x in range(len(sample_emg_filtered[0]))])
        sample_emg_filtered[1]= sample_emg_filtered[0]*1.5
        sample_emg_filtered = np.concatenate(
            (sample_emg_filtered,
             [sample_emg_filtered[0] * 1.7 + np.sin(t * 2 * np.pi)]), axis=0)
        components = ecg_rm.compute_ica_two_comp_multi(sample_emg_filtered)
        self.assertEqual(
            (len(components)),
            2 ,
        )

    def test_compute_ica_two_comp_selective(self):
        sample_read = Poly5Reader(sample_emg)
        sample_emg_filtered = filt.emg_bandpass_butter(sample_read, 1, 10)
        sample_emg_filtered = sample_emg_filtered[:, :30*2048]
        t = np.array([x/2048 for x in range(len(sample_emg_filtered[0]))])
        sample_emg_filtered[1] = sample_emg_filtered[0]*1.5
        sample_emg_filtered = np.concatenate(
            (sample_emg_filtered,
             [sample_emg_filtered[0] * 1.7] + np.sin(t * 2 * np.pi)), axis=0)
        components =  ecg_rm.compute_ICA_two_comp_selective(
        sample_emg_filtered,
        use_all_leads=False,
        desired_leads=[0, 2],
        )
        self.assertEqual(
            (len(components)),
            2 ,
        )


class TestEcgPeakDetection(unittest.TestCase):
    data_emg = Poly5Reader(synth_pocc_emg)
    y_emg = data_emg.samples[:data_emg.num_samples]
    fs_emg = data_emg.sample_rate 

    def test_detect_ecg_peaks(self):
        ecg_peaks = ecg_rm.detect_ecg_peaks(
            ecg_raw=self.y_emg[0],
            fs=self.fs_emg,
            bp_filter=True,
        )
        self.assertEqual(
            len(ecg_peaks),
            449
        )


class TestGating(unittest.TestCase):
    sample_read= Poly5Reader(sample_emg)
    sample_emg_filtered = -filt.emg_bandpass_butter(sample_read, 1, 500)
    sample_emg_filtered = sample_emg_filtered[:30*2048]
    ecg_peaks, _  = scipy.signal.find_peaks(sample_emg_filtered[0, :])

    def test_gating_method_0(self):
        ecg_gated_0 = ecg_rm.gating(self.sample_emg_filtered[0, :], self.ecg_peaks,
                             gate_width=205, method=0)

        self.assertEqual(
            (len(self.sample_emg_filtered[0])),
            len(ecg_gated_0) ,
        )

    def test_gating_method_1(self):
        ecg_gated_1 = ecg_rm.gating(
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
        ecg_gated_2 = ecg_rm.gating(self.sample_emg_filtered[0, :], self.ecg_peaks,
                             gate_width=205, method=2)

        self.assertEqual(
            (len(self.sample_emg_filtered[0])),
            len(ecg_gated_2) ,
        )

    def test_gating_method_2_no_prior_segment(self):
        ecg_gated_2 = ecg_rm.gating(
            self.sample_emg_filtered[0, :], [100], gate_width=205, method=2)

        self.assertFalse(
            np.isnan(np.sum(ecg_gated_2))
        )
    def test_gating_method_3(self):
        height_threshold = np.max(self.sample_emg_filtered)/2
        ecg_peaks, _  = scipy.signal.find_peaks(
            self.sample_emg_filtered[0, :10*2048-1],
            height=height_threshold)

        ecg_gated_3 = ecg_rm.gating(self.sample_emg_filtered[0, :10*2048], ecg_peaks,
                             gate_width=205, method=3)

        self.assertEqual(
            (len(self.sample_emg_filtered[0, :10*2048])),
            len(ecg_gated_3) ,
        )

class TestWaveletDenoising(unittest.TestCase):
    sample_read = Poly5Reader(sample_emg)
    fs = sample_read.sample_rate
    sample_emg_filtered = -filt.emg_bandpass_butter(sample_read, 1, 500)
    sample_emg_filtered = sample_emg_filtered[:30*2048]
    ecg_peaks, _  = scipy.signal.find_peaks(sample_emg_filtered[0, :])

    def wavelet_denoising(self):
        ecg_denoised = ecg_rm.wavelet_denoising(
            emg_raw=self.sample_emg_filtered[0, :],
            ecg_peak_idxs=self.ecg_peaks,
            fs=self.fs,
            hard_thresholding=True,
            n=4,
            wavelet_type='db2',
            fixed_threshold=4.5
        )

        self.assertEqual(
            (len(self.sample_emg_filtered[0])),
            len(ecg_denoised),
        )
