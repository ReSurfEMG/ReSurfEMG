#sanity tests for the visualization functions
import unittest
import os
from unittest.mock import patch
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.preprocessing.filtering import emg_bandpass_butter
from resurfemg.visualization.visualization import show_psd_welch
from resurfemg.visualization.visualization import show_periodogram
from resurfemg.visualization.visualization import show_my_power_spectrum

sample_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'not_pushed',
    'Test_lung_data',
    '2022-05-13_11-51-04',
    '002',
    'EMG_recording.Poly5',
)


class TestVisualizationMethods(unittest.TestCase):

    def setUp(self):
        # Common setup for all tests
        sample_read = Poly5Reader(sample_emg)
        self.sample_emg_filtered = -emg_bandpass_butter(sample_read, 1, 500)
        self.sample_emg_filtered = self.sample_emg_filtered[:30*2048]


    @patch('matplotlib.pyplot.show')
    def test_show_my_power_spectrum(self, mock_show):
        f,Pxx_den = show_my_power_spectrum(self.sample_emg_filtered[0, :],
                                           2048, 1024)
        self.assertEqual(len(f),
                         len(self.sample_emg_filtered[0, :]))
        self.assertEqual(len(Pxx_den),
                         len(self.sample_emg_filtered[0, :]))
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_show_psd_welch(self, mock_show):
        f, Pxx_den = show_psd_welch(self.sample_emg_filtered[0, :],
                                    2048, 256, axis_spec=0)
        expected_length = 256 // 2 + 1
        self.assertEqual((len(f)),
                         expected_length)
        self.assertEqual(len(Pxx_den),
                         expected_length)
        mock_show.assert_called_once()


    @patch('matplotlib.pyplot.show')
    def test_show_periodogram(self, mock_show):
        f,Pxx_den = show_periodogram(self.sample_emg_filtered[0, :],
                                     2048, 0)
        expected_length = len(self.sample_emg_filtered[0, :]) // 2 + 1
        self.assertEqual(len(f), expected_length)
        self.assertEqual(len(Pxx_den), expected_length)
        mock_show.assert_called_once()