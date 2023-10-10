#sanity tests for the converter functions

import unittest
import os
from tempfile import TemporaryDirectory
from unittest import main

# tmsisdk_lite
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
# converter_functions 
from resurfemg.data_connector.converter_functions import poly5unpad
from resurfemg.config.config import hash_it_up_right_all

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
            self.assertTrue(hash_it_up_right_all(td, '.Poly5').equals(
                hash_it_up_right_all(td, '.Poly5')))