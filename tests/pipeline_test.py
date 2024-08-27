"""sanity tests for the pipelines"""

import os
import unittest
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.preprocessing.filtering  import emg_bandpass_butter
from resurfemg.preprocessing.ecg_removal  import (
    detect_ecg_peaks)
from resurfemg.pipelines.pipelines import (
    working_pipe_multi, working_pipeline_pre_ml_multi, ecg_removal_gating)

synth_pocc_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'test_data',
    'emg_data_synth_quiet_breathing.Poly5',
)


sample_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'not_pushed',
    'Test_lung_data',
    '2022-05-13_11-51-04',
    '002',
    'EMG_recording.Poly5',
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


class TestGatingPipeline(unittest.TestCase):
    data_emg = Poly5Reader(synth_pocc_emg)
    y_emg = data_emg.samples[:data_emg.num_samples]
    fs_emg = data_emg.sample_rate 
    ecg_peaks = detect_ecg_peaks(
        ecg_raw=y_emg[0],
        fs=fs_emg,
        bp_filter=True,
    )
    def test_gating_pipeline(self):
        emg_di_gated = ecg_removal_gating(
            emg_raw=self.y_emg[1, :],
            ecg_peaks_idxs=self.ecg_peaks,
            gate_width_samples=self.fs_emg // 10,
            ecg_shift=10,
        )
        self.assertEqual(
            len(emg_di_gated),
            self.y_emg.shape[1]
        )


if __name__ == '__main__':
    unittest.main()
