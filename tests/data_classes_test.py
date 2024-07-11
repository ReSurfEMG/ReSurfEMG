"""sanity tests for the data classes module of the resurfemg library"""


import unittest
import numpy as np
import scipy
import matplotlib.pyplot as plt
from resurfemg.data_classes.data_classes import TimeSeriesGroup

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
emg_timeseries = TimeSeriesGroup(
        y_emg, fs=fs_emg, labels=['ECG', 'EMGdi', 'EMGpara'], units=3*['uV'])
emg_timeseries.envelope(signal_type='raw')
emg_timeseries.baseline()
emg_di = emg_timeseries.channels[1]
peaks_s, _ = scipy.signal.find_peaks(
    emg_di.y_env, prominence=1.0)
emg_di.set_peaks(
    peak_set_name='breaths',
    peaks_s=peaks_s,
    signal=emg_di.y_env)
emg_di.peaks['breaths'].detect_on_offset(
    baseline=emg_di.y_baseline
)


class TestTimeSeriesGroup(unittest.TestCase):
    def test_raw_data(self):
        self.assertEqual(
            len(emg_timeseries.channels[0].y_raw),
            len(y_emg[0, :])
        )

    def test_time_data(self):
        self.assertEqual(
            emg_timeseries.channels[0].t_data.shape,
            t_emg.shape
        )

    def test_env_data(self):
        self.assertEqual(
            len(emg_timeseries.channels[0].y_env),
            len(y_emg[0, :])
        )

    def test_plot_full(self):
        _, axes = plt.subplots(
            nrows=y_emg.shape[0], ncols=1, figsize=(10, 6), sharex=True)
        emg_timeseries.plot_full(axes)

        _, y_plot_data = axes[-1].lines[0].get_xydata().T

        np.testing.assert_array_equal(
            emg_timeseries.channels[-1].y_env, y_plot_data)

    def test_plot_peaks(self):
        _, axes = plt.subplots(
            nrows=1, ncols=len(peaks_s), figsize=(10, 6), sharex=True)
        emg_timeseries.plot_peaks(peak_set_name='breaths', axes=axes,
                                  channel_idxs=1, margin_s=0)
        emg_timeseries.plot_markers(peak_set_name='breaths', axes=axes,
                                    channel_idxs=1)
        peak_set = emg_di.peaks['breaths']
        len_last_peak = peak_set.ends_s[-1] - peak_set.starts_s[-1]
        y_plot_data_list = list()
        for _, line in enumerate(axes[-1].lines):
            _, y_plot_data = line.get_xydata().T
            y_plot_data_list.append(len(y_plot_data))

        # Length of plotted data: [signal, baseline, peak_s, start_s, end_s]
        np.testing.assert_array_equal(
            [len_last_peak, len_last_peak, 1, 1, 1],
            y_plot_data_list)
