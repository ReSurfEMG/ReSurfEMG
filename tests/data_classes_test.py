"""sanity tests for the data classes module of the resurfemg library"""

import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.data_classes.data_classes import (
    VentilatorDataGroup, EmgDataGroup)

synth_pocc_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'test_data',
    'emg_data_synth_pocc.Poly5',
)
synth_pocc_vent = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'test_data',
    'vent_data_synth_pocc.Poly5',
)

class TestTimeSeriesGroup(unittest.TestCase):
    data_vent = Poly5Reader(synth_pocc_vent)
    y_vent = data_vent.samples[:data_vent.num_samples]
    fs_vent = data_vent.sample_rate
    vent_timeseries = VentilatorDataGroup(
        y_vent,
        fs=fs_vent,
        labels=['Paw', 'F', 'Vvent'],
        units=['cmH2O', 'L/s', 'L']
    )
    def test_find_peep(self):
        self.assertEqual(
            self.vent_timeseries.peep,
            3.0
        )

    vent_timeseries.baseline(channel_idxs=[0], signal_type='raw')

    # Find occlusion pressures
    vent_timeseries.find_occluded_breaths(
        vent_timeseries.p_aw_idx, start_idx=360*vent_timeseries.fs)
    paw = vent_timeseries.channels[vent_timeseries.p_aw_idx]
    paw.peaks['Pocc'].detect_on_offset(baseline=paw.y_baseline)
    def test_find_occluded_breaths(self):
        np.testing.assert_array_equal(
            self.paw.peaks['Pocc'].peak_df['peak_idx'],
            [37465, 39101, 40465]
        )

    # Find supported breath pressures
    v_vent = vent_timeseries.channels[vent_timeseries.v_vent_idx]
    vent_timeseries.find_tidal_volume_peaks()
    def test_find_tidal_volume_peaks(self):
        peak_df = self.paw.peaks['ventilator_breaths'].peak_df
        self.assertEqual(
            len(peak_df['peak_idx']),
            151
        )

    # Calculate PTPs
    paw.calculate_time_products(
        peak_set_name='Pocc',
        aub_reference_signal=paw.y_baseline,
        parameter_name='PTPocc')

    def test_time_product(self):
        self.assertIn(
            'PTPocc',
            self.paw.peaks['Pocc'].peak_df.columns.values
        )
        np.testing.assert_array_almost_equal(
            self.paw.peaks['Pocc'].peak_df['PTPocc'].values,
            np.array([7.96794678, 7.81619293, 7.89553107])
        )

    # Test Pocc quality
    parameter_names = {
        'time_product': 'PTPocc'
    }
    paw.test_pocc_quality(
        'Pocc', parameter_names=parameter_names, verbose=False)
    def test_pocc_quality_assessment(self):
        tests = ['baseline_detection', 'consecutive_poccs', 'pocc_upslope']
        for test in tests:
            self.assertIn(
                test,
                self.paw.peaks['Pocc'].quality_outcomes_df.columns.values
            )

        np.testing.assert_array_almost_equal(
            self.paw.peaks['Pocc'].peak_df['valid'].values,
            np.array([True, True, True])
        )

    data_emg = Poly5Reader(synth_pocc_emg)
    y_emg = data_emg.samples[:data_emg.num_samples]
    fs_emg = data_emg.sample_rate
    emg_timeseries = EmgDataGroup(
        y_emg,
        fs=fs_emg,
        labels=['ECG', 'EMGdi'],
        units=3*['uV'])

    def test_raw_data(self):
        self.assertEqual(
            len(self.emg_timeseries.channels[0].y_raw),
            len(self.y_emg[0, :])
        )

    def test_time_data(self):
        self.assertEqual(
            len(self.emg_timeseries.channels[0].t_data),
            len(self.y_emg[0, :])
        )

    emg_timeseries.filter()
    emg_timeseries.gating()
    def test_clean_data(self):
        self.assertEqual(
            len(self.emg_timeseries.channels[0].y_clean),
            len(self.y_emg[0, :])
        )

    emg_timeseries.envelope(env_type='rms', signal_type='clean')
    def test_env_data_rms(self):
        self.assertEqual(
            len(self.emg_timeseries.channels[0].y_env),
            len(self.y_emg[0, :])
        )

    emg_timeseries.envelope(env_type='arv', signal_type='clean')
    def test_env_data_arv(self):
        self.assertEqual(
            len(self.emg_timeseries.channels[0].y_env),
            len(self.y_emg[0, :])
        )

    emg_timeseries.baseline()
    def test_baseline_data(self):
        self.assertEqual(
            len(self.emg_timeseries.channels[0].y_baseline),
            len(self.y_emg[0, :])
        )

    # Find sEAdi peaks in one channel (sEAdi)
    emg_di = emg_timeseries.channels[1]
    emg_di.detect_emg_breaths(peak_set_name='breaths')
    emg_di.peaks['breaths'].detect_on_offset(
        baseline=emg_di.y_baseline
    )
    def test_find_peaks(self):
        self.assertEqual(
            len(self.emg_di.peaks['breaths'].peak_df),
            154
        )

    # Link ventilator Pocc peaks to EMG breaths
    t_pocc_peaks_vent = paw.peaks['Pocc'].peak_df['peak_idx']/paw.fs
    emg_di.link_peak_set(
        peak_set_name='breaths',
        t_reference_peaks=t_pocc_peaks_vent,
        linked_peak_set_name='Pocc',
    )

    def test_link_peak_set(self):
        self.assertEqual(
            len(self.emg_di.peaks['Pocc'].peak_df),
            3
        )

    # Calculate ETPs
    emg_di.calculate_time_products(
        peak_set_name='Pocc',
        parameter_name='ETPdi')

    def test_emg_time_product(self):
        self.assertIn(
            'ETPdi',
            self.emg_di.peaks['Pocc'].peak_df.columns.values
        )
        np.testing.assert_array_almost_equal(
            self.emg_di.peaks['Pocc'].peak_df['ETPdi'].values,
            np.array([3.59565976, 3.78080979, 3.55626967])
        )

    # Test emg_quality_assessment
    parameter_names = {
        'time_product': 'ETPdi'
    }
    emg_di.test_emg_quality(
        'Pocc', verbose=False, parameter_names=parameter_names)
    def test_emg_quality_assessment(self):
        tests = ['baseline_detection', 'interpeak_distance', 'snr', 'aub']
        for test in tests:
            self.assertIn(
                test,
                self.emg_di.peaks['Pocc'].quality_outcomes_df.columns.values
            )

        np.testing.assert_array_almost_equal(
            self.emg_di.peaks['Pocc'].peak_df['valid'].values,
            np.array([True, True, True])
        )

    def test_plot_full(self):
        _, axes = plt.subplots(
            nrows=self.y_emg.shape[0], ncols=1, figsize=(10, 6), sharex=True)
        self.emg_timeseries.plot_full(axes)

        _, y_plot_data = axes[-1].lines[0].get_xydata().T

        np.testing.assert_array_equal(
            self.emg_timeseries.channels[-1].y_env, y_plot_data)

    def test_plot_peaks(self):
        _, axes = plt.subplots(
            nrows=1, ncols=3, figsize=(10, 6), sharex=True)
        self.emg_timeseries.plot_peaks(peak_set_name='Pocc', axes=axes,
                                  channel_idxs=1, margin_s=0)
        self.emg_timeseries.plot_markers(peak_set_name='Pocc', axes=axes,
                                    channel_idxs=1)
        peak_df = self.emg_di.peaks['Pocc'].peak_df
        len_peaks = len(peak_df)
        len_last_peak = (peak_df.loc[len_peaks-1, 'end_idx']
                        - peak_df.loc[len_peaks-1, 'start_idx'])
        y_plot_data_list = list()
        for _, line in enumerate(axes[-1].lines):
            _, y_plot_data = line.get_xydata().T
            y_plot_data_list.append(len(y_plot_data))

        # Length of plotted data:
        # [signal, baseline, peak_idx, start_idx, end_idx]
        np.testing.assert_array_equal(
            [len_last_peak, len_last_peak, 1, 1, 1],
            y_plot_data_list)
