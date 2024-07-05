"""sanity tests for the data classes module of the resurfemg library"""


import unittest
import numpy as np
import matplotlib.pyplot as plt
from resurfemg.data_classes.data_classes import TimeSeriesData

# Define EMG signal
fs_emg = 2048
t_emg = np.array([s_t/fs_emg for s_t in range(10*fs_emg)])
y_amps = [2, 5, 12]
y_sin = np.cos((0.5* t_emg - 0.5)* 2 * np.pi)
y_sin[y_sin < 0] = 0
y_emg = np.zeros((3, len(t_emg)))
for idx, y_amp in enumerate(y_amps):
    y_rand = np.random.normal(0, 1, size=len(y_sin))
    y_rand_baseline = np.random.normal(0, 1, size=len(y_sin)) / 10
    y_t = y_amp * y_sin * y_rand + y_rand_baseline
    y_emg[idx, :] = y_t

class TestTimeSeriesData(unittest.TestCase):
    emg_timeseries = TimeSeriesData(
        y_emg, fs=fs_emg, labels=['ECG', 'EMGdi', 'EMGpara'], units=3*['uV'])
    emg_timeseries.envelope(signal_type='raw')
    emg_timeseries.baseline()

    def test_raw_data(self):
        self.assertEqual(
            self.emg_timeseries.y_raw.shape,
            y_emg.shape
        )

    def test_time_data(self):
        self.assertEqual(
            self.emg_timeseries.t_data.shape,
            t_emg.shape
        )

    def test_env_data(self):
        self.assertEqual(
            self.emg_timeseries.y_env.shape,
            y_emg.shape
        )

    def test_plot_full(self):
        _, axes = plt.subplots(
            nrows=y_emg.shape[0], ncols=1, figsize=(10, 6), sharex=True)
        self.emg_timeseries.plot_full(axes)

        _, y_plot_data = axes[-1].lines[0].get_xydata().T

        np.testing.assert_array_equal(
            self.emg_timeseries.y_env[-1, :],y_plot_data)
