#sanity tests for the pipelines

import os
import unittest
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.pipelines.pipelines import working_pipe_multi
from resurfemg.pipelines.pipelines import working_pipeline_pre_ml_multi
from resurfemg.preprocessing.filtering  import emg_bandpass_butter


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


if __name__ == '__main__':
    unittest.main()
