"""sanity tests for the synthetic data functions"""

import numpy as np
from unittest import TestCase
from resurfemg.data_connector.synthetic_data import make_realistic_syn_emg

class TestSyntheticData(TestCase):
    def test_make_realistic_syn_emg(self):
        x_ecg = np.zeros((10,307200))
        made = make_realistic_syn_emg(x_ecg,2)
        self.assertEqual(len(made),2)
