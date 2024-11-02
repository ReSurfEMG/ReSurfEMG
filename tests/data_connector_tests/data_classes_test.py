"""sanity tests for the data classes module of the resurfemg library"""

import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.data_connector.data_classes import (
    VentilatorDataGroup, EmgDataGroup)
from resurfemg.postprocessing import features as feat

synth_pocc_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__)))),
    'test_data',
    'emg_data_synth_pocc.Poly5',
)
synth_pocc_vent = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__)))),
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
        labels=['Pvent', 'F', 'Vvent'],
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
        vent_timeseries.p_vent_idx,
        start_idx=360*vent_timeseries.param['fs'])
    p_vent = vent_timeseries.channels[vent_timeseries.p_vent_idx]
    p_vent.peaks['Pocc'].detect_on_offset(baseline=p_vent.y_baseline)
    def test_find_occluded_breaths(self):
        np.testing.assert_array_equal(
            self.p_vent.peaks['Pocc'].peak_df['peak_idx'],
            [37465, 39101, 40465]
        )

    # Find supported breath pressures
    v_vent = vent_timeseries.channels[vent_timeseries.v_vent_idx]
    vent_timeseries.find_tidal_volume_peaks()
    def test_find_tidal_volume_peaks(self):
        peak_df = self.p_vent.peaks['ventilator_breaths'].peak_df
        self.assertEqual(
            len(peak_df['peak_idx']),
            151
        )

    # Calculate PTPs
    p_vent.calculate_time_products(
        peak_set_name='Pocc',
        aub_reference_signal=p_vent.y_baseline,
        parameter_name='PTPocc')

    def test_time_product(self):
        self.assertIn(
            'PTPocc',
            self.p_vent.peaks['Pocc'].peak_df.columns.values
        )
        np.testing.assert_array_almost_equal(
            self.p_vent.peaks['Pocc'].peak_df['PTPocc'].values,
            np.array([7.96794678, 7.81619293, 7.89553107])
        )

    # Test Pocc quality
    parameter_names = {
        'time_product': 'PTPocc'
    }
    p_vent.test_pocc_quality(
        'Pocc', parameter_names=parameter_names, verbose=False)
    def test_pocc_quality_assessment(self):
        tests = ['baseline_detection', 'consecutive_poccs', 'pocc_upslope']
        for test in tests:
            self.assertIn(
                test,
                self.p_vent.peaks['Pocc'].quality_outcomes_df.columns.values
            )

        np.testing.assert_array_almost_equal(
            self.p_vent.peaks['Pocc'].peak_df['valid'].values,
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
<<<<<<< HEAD
=======
    print(emg_timeseries.fs)
>>>>>>> 34c784f (Release 2 0 0/wavelet denoising (#336))
    emg_timeseries.filter()
    def test_filtered_data(self):
        self.assertEqual(
            len(self.emg_timeseries.channels[0].y_filt),
            len(self.y_emg[0, :])
        )
    emg_timeseries.wavelet_denoising(overwrite=True)
    def test_clean_data_wavelet_denosing(self):
        self.assertEqual(
            len(self.emg_timeseries.channels[0].y_clean),
            len(self.y_emg[0, :])
        )

    emg_timeseries.gating(overwrite=True)
    def test_clean_data_gating(self):
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
    t_pocc_peaks_vent = (p_vent.peaks['Pocc'].peak_df['peak_idx'] /
                         p_vent.param['fs'])
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
            np.array([3.575323, 3.722479, 3.461432])
        )

    # Test emg_quality_assessment
    parameter_names = {
        'time_product': 'ETPdi'
    }
    emg_di.test_emg_quality(
        'Pocc', verbose=False, parameter_names=parameter_names)

    def test_emg_quality_assessment(self):
        tests = ['interpeak_distance', 'snr', 'aub', 'bell']
        for test in tests:
            self.assertIn(
                test,
                self.emg_di.peaks['Pocc'].quality_outcomes_df.columns.values
            )

        np.testing.assert_array_equal(
            self.emg_di.peaks['Pocc'].peak_df['valid'].values,
            np.array([True, True, True])
        )

    # Test the ventilatory Pocc peaks against the EMG peaks
    rr_vent, _ = feat.respiratory_rate(
        v_vent.peaks['ventilator_breaths'].peak_df['peak_idx'].to_numpy(),
        v_vent.param['fs'])
    p_vent.param['rr_occ'] = 60*len(
        p_vent.peaks['Pocc'].peak_df)/(p_vent.t_data[-1])
    cutoff = {
        'fraction_emg_breaths': 0.1,
        'delta_min': 0.5*rr_vent/60,
        'delta_max': 0.6
    }
    parameter_names = {
        'rr': 'rr_occ'
    }

    emg_di.test_linked_peak_sets(
        peak_set_name='Pocc',
        linked_timeseries=p_vent,
        linked_peak_set_name='Pocc',
        verbose=True,
        cutoff=cutoff,
        parameter_names=parameter_names,
    )
    def test_test_linked_peak_sets(self):
        tests = ['detected_fraction', 'event_timing']
        for test in tests:
            self.assertIn(
                test,
                self.emg_di.peaks['Pocc'].quality_outcomes_df.columns.values
            )

        np.testing.assert_array_equal(
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
        self.emg_di.plot_curve_fits(axes=axes[1], peak_set_name='Pocc')
        self.emg_di.plot_aub(
            axes=axes[1], signal_type='env', peak_set_name='Pocc')
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
