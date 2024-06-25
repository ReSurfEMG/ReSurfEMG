import unittest
import os
import numpy as np
from tempfile import TemporaryDirectory
from resurfemg.machine_learning.ml import save_ml_output
from resurfemg.preprocessing.filtering  import emg_bandpass_butter

sample_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'not_pushed',
    'Test_lung_data',
    '2022-05-13_11-51-04',
    '002',
    'EMG_recording.Poly5',
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
