"""sanity tests for the postprocessing module of the resurfemg library"""


import unittest
import os
import numpy as np
import scipy
from scipy.integrate import trapezoid

from resurfemg.preprocessing import envelope as evl
import resurfemg.postprocessing.baseline as bl
import resurfemg.postprocessing.event_detection as evt
import resurfemg.postprocessing.features as feat
import resurfemg.postprocessing.quality_assessment as qa

sample_emg = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'test_data',
    'emg_data_synth_quiet_breathing.Poly5',
)
# Dummy EMG signal
fs_emg = 2048
rr = 30
t_r = 60/rr
f_r = 1/t_r
t_emg = np.array([s_t/fs_emg for s_t in range(10*fs_emg)])
y_sin = np.cos((f_r* t_emg - 0.5)* 2 * np.pi)
y_sin[y_sin < 0] = 0
y_rand = np.random.normal(0, 1, size=len(y_sin))
y_rand_baseline = np.random.normal(0, 1, size=len(y_sin)) / 10
y_t_emg = y_sin * y_rand + y_rand_baseline

y_env_emg = evl.full_rolling_rms(y_t_emg, fs_emg // 5)
y_emg_baseline = bl.moving_baseline(y_env_emg, 5*fs_emg, fs_emg//2)

peaks_env, _ = scipy.signal.find_peaks(y_env_emg, prominence=0.1)
emg_start_idxs, emg_end_idxs, *_ = evt.onoffpeak_baseline_crossing(
             y_env_emg, y_emg_baseline, peaks_env)
etps = feat.time_product(
    signal=y_env_emg,
    fs=fs_emg,
    start_idxs=emg_start_idxs,
    end_idxs=emg_end_idxs)

# Dummy Pocc signal
fs_vent = 100
s_vent = np.array([(s_t) for s_t in range(10*fs_vent)])
t_vent = (s_vent + 1)/fs_vent
y_sin = np.sin((f_r* t_vent)* 2 * np.pi)
y_sin[y_sin > 0] = 0
y_t_p_vent = 5 * y_sin

pocc_peaks_valid, _ = scipy.signal.find_peaks(-y_t_p_vent, prominence=0.1)
pocc_starts = s_vent[((t_vent+t_r/2)%t_r == 0)]
pocc_ends = s_vent[(t_vent%t_r == 0)]

PTP_occs = np.zeros(pocc_peaks_valid.shape)
for _idx, _ in enumerate(pocc_peaks_valid):
    PTP_occs[_idx] = trapezoid(
        -y_t_p_vent[pocc_starts[_idx]:pocc_ends[_idx]],
        dx=1/fs_vent
    )

class TestSnrPseudo(unittest.TestCase):
    fs_emg = 2048
    t_emg = np.array([s_t/fs_emg for s_t in range(15*fs_emg)])

    y_block = np.array(
        10*scipy.signal.square((t_emg - 1.25)/5 * 2 * np.pi, duty=0.5))
    y_block[y_block < 0] = 0
    y_baseline = np.ones(y_block.shape)
    peak_idxs = [(5//2 + x*5) * 2048 for x in range(3)]

    snr_values = qa.snr_pseudo(y_block, peak_idxs, y_baseline, fs_emg)

    def test_snr_length(self):
        self.assertEqual(
            len(self.snr_values),
            len(self.peak_idxs),
            )

    def test_snr_values(self):
        median_snr = np.median(self.snr_values)
        self.assertAlmostEqual(
            median_snr, 10.0, 3
            )

class TestPoccQuality(unittest.TestCase):
    valid_poccs, _ = qa.pocc_quality(
        y_t_p_vent, pocc_peaks_valid, pocc_ends, PTP_occs)
    def test_valid_pocc(self):
        self.assertFalse(
            np.any(~self.valid_poccs)
            )

    def test_negative_upslope(self):
        y_sin_shifted = np.sin(((f_r-0.025)* t_vent - 0.11)* 2 * np.pi)
        y_sin_shifted[y_sin_shifted > 0] = 0
        y_sin_shifted[t_vent < 7.0] = 0
        y_t_shifted = 5 * y_sin + 4 * y_sin_shifted

        invalid_upslope, _ = qa.pocc_quality(
            y_t_shifted, pocc_peaks_valid, pocc_ends, PTP_occs)

        self.assertFalse(
            invalid_upslope[-1]
            )

    def test_steep_upslope(self):
        y_sin_shifted = np.sin((f_r* t_vent - 0.4)* 2 * np.pi)
        y_sin_shifted[y_sin_shifted > 0] = 0
        y_sin_shifted = y_sin_shifted ** 4
        y_t_steeper = 1000 * y_sin * y_sin_shifted

        peak_idxs_steeper, _ = scipy.signal.find_peaks(
            -y_t_steeper, prominence=0.1)
        y_baseline = bl.moving_baseline(-y_t_steeper, 7.5*fs_vent, fs_vent//5)

        peak_start_idxsteep, peak_end_idxs_steep, _, _, _ = \
            evt.onoffpeak_baseline_crossing(
                y_t_steeper, y_baseline, peak_idxs_steeper)

        ptp_occs_steep = np.zeros(peak_idxs_steeper.shape)
        for idx, _ in enumerate(peak_idxs_steeper):
            ptp_occs_steep[idx] = trapezoid(
                -y_t_steeper[peak_start_idxsteep[idx]:
                             peak_end_idxs_steep[idx]],
                dx=1/fs_vent
            )

        steep_upslope, _ = qa.pocc_quality(
            y_t_steeper,
            peak_idxs_steeper,
            peak_end_idxs_steep,
            ptp_occs_steep)

        self.assertFalse(
            steep_upslope[-1]
            )

    def test_consec_manoeuvres(self):
        sim_breaths = np.arange(1,20,2)
        sim_occ = np.arange(1,20,10)
        sim_occ_false = np.array([1, 7, 9])
        valid_manoeuvres = qa.detect_non_consecutive_manoeuvres(
            ventilator_breath_idxs=sim_breaths,
            manoeuvres_idxs=sim_occ)
        self.assertTrue(
            np.all(valid_manoeuvres)
        )
        valid_manoeuvres_false = qa.detect_non_consecutive_manoeuvres(
            sim_breaths, sim_occ_false)
        self.assertFalse(
            np.all(valid_manoeuvres_false)
        )


class TestAreaUnderBaselineQuality(unittest.TestCase):
    # Define signal
    fs_emg = 2048
    t_emg = np.array([s_t/fs_emg for s_t in range(15*fs_emg)])

    y_block = np.array(
        3*scipy.signal.square((t_emg - 1.25)/5 * 2 * np.pi, duty=0.5))
    y_block[y_block < 0] = 0

    peak_idxs = [(5//2 + x*5) * 2048 for x in range(3)]
    start_idxs = [(5 + x*5*4) * 2048 //4 for x in range(3)]
    end_idxs = [(15 + x*5*4) * 2048 //4 - 1 for x in range(3)]

    def test_percentage_aub_good(self):
        y_baseline = np.ones(self.y_block.shape)
        valid_timeproducts, _, _ = qa.percentage_under_baseline(
            self.y_block,
            self.fs_emg,
            self.peak_idxs,
            self.start_idxs,
            self.end_idxs,
            y_baseline,
            aub_window_s=None,
            ref_signal=None,
            aub_threshold=40,
        )

        self.assertTrue(
            np.all(valid_timeproducts)
            )

    def test_percentage_aub_wrong(self):
        y_baseline = 2*np.ones(self.y_block.shape)
        valid_timeproducts, _, _ = qa.percentage_under_baseline(
            self.y_block,
            self.fs_emg,
            self.peak_idxs,
            self.start_idxs,
            self.end_idxs,
            y_baseline,
            aub_window_s=None,
            ref_signal=None,
            aub_threshold=40,
        )

        self.assertFalse(np.all(valid_timeproducts))


class TestBellFit(unittest.TestCase):
    def test_evaluate_bell_curve_error(self):
        output = qa.evaluate_bell_curve_error(
            peak_idxs=peaks_env,
            start_idxs=emg_start_idxs,
            end_idxs=emg_end_idxs,
            signal=y_env_emg,
            fs=fs_emg,
            time_products=etps,
            bell_window_s=None,
            bell_threshold=40,
        )
        (valid_peak, _, percentage_bell_error, *_) = output

        np.testing.assert_equal(
            valid_peak,
            np.array([True, True, True, True, True])
        )
        np.testing.assert_equal(
            np.abs(percentage_bell_error - 5) < 5,
            np.array([True, True, True, True, True])
        )


class TestInterpeakMethods(unittest.TestCase):
    def test_interpeak_dist(self):
        sim_ECG=np.arange(1, 11)
        sim_EMG=np.linspace(1, 10, 4)
        valid_interpeak = qa.interpeak_dist(sim_ECG, sim_EMG, threshold=1.1)

        self.assertTrue(valid_interpeak, "The interpeak_dist function"
                        "did not return True as expected.")

class TestEvaluateRespiratoryRates(unittest.TestCase):
    def test_evaluate_respiratory_rates(self):
        print(rr, peaks_env)
        fraction_emg_breaths, crit_met = qa.evaluate_respiratory_rates(
            emg_breath_idxs=peaks_env,
            t_emg=max(t_emg),
            rr_vent=rr,
        )
        self.assertAlmostEqual(fraction_emg_breaths, 1.0, 2)
        self.assertTrue(crit_met)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
