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
    'not_pushed',
    'Test_lung_data',
    '2022-05-13_11-51-04',
    '002',
    'EMG_recording.Poly5',
)

# Dummy EMG signal
fs_emg = 2048
t_emg = np.array([s_t/fs_emg for s_t in range(10*fs_emg)])
y_sin = np.cos((0.5* t_emg - 0.5)* 2 * np.pi)
y_sin[y_sin < 0] = 0
y_rand = np.random.normal(0, 1, size=len(y_sin))
y_rand_baseline = np.random.normal(0, 1, size=len(y_sin)) / 10
y_t_emg = y_sin * y_rand + y_rand_baseline

y_env_emg = evl.full_rolling_rms(y_t_emg, fs_emg // 5)
y_emg_baseline = bl.moving_baseline(y_env_emg, 5*fs_emg, fs_emg//2)

peaks_env, _ = scipy.signal.find_peaks(y_env_emg, prominence=0.1)
_, emg_starts_s, emg_ends_s, *_ = evt.onoffpeak_baseline_crossing(
             y_env_emg, y_emg_baseline, peaks_env)
etps = feat.time_product(
    signal=y_env_emg, fs=fs_emg, starts_s=emg_starts_s, ends_s=emg_ends_s)

# Dummy Pocc signal
fs_vent = 100
s_vent = np.array([(s_t) for s_t in range(10*fs_vent)])
t_vent = (s_vent + 1)/fs_vent
rr = 12
t_r = 60/rr
f_r = 1/t_r
y_sin = np.sin((f_r* t_vent)* 2 * np.pi)
y_sin[y_sin > 0] = 0
y_t_paw = 5 * y_sin

pocc_peaks_valid, _ = scipy.signal.find_peaks(-y_t_paw, prominence=0.1)
pocc_starts = s_vent[((t_vent+t_r/2)%t_r == 0)]
pocc_ends = s_vent[(t_vent%t_r == 0)]

PTP_occs = np.zeros(pocc_peaks_valid.shape)
for _idx, _ in enumerate(pocc_peaks_valid):
    PTP_occs[_idx] = np.trapz(
        -y_t_paw[pocc_starts[_idx]:pocc_ends[_idx]],
        dx=1/fs_vent
    )

class TestBaseline(unittest.TestCase):
    fs = 1000
    t = np.arange(0, 10, 1/fs)
    slow_component = np.sin(2 * np.pi * 0.1 * t)
    fast_component = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    breathing_signal = np.abs(slow_component + fast_component)

    def test_movingbaseline(self):
        sinusbase = bl.moving_baseline(self.breathing_signal,
                                    self.fs,
                                    self.fs//5,
                                    33)
        self.assertEqual(
            (len(self.breathing_signal)),
            len(sinusbase),
    )

    def test_slopesum(self):
        sinusbase, _, _, _ = bl.slopesum_baseline(
            self.breathing_signal,
            self.fs,
            self.fs//5,
            self.fs,
            set_percentile=33,
            augm_percentile=25)

        self.assertEqual(
            (len(self.breathing_signal)),
            len(sinusbase),
            )

class TestEventDetection(unittest.TestCase):
    fs = 1000
    t = np.arange(0, 10, 1/fs)
    slow_component = np.sin(2 * np.pi * 0.1 * t)
    fast_component = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    breathing_signal = np.abs(slow_component + fast_component)

    baseline = 0.5 * np.ones((len(breathing_signal), ))
    treshold = 0
    width = fs // 2
    prominence = 0.5 * (np.nanpercentile(breathing_signal - baseline, 75)
                        + np.nanpercentile(breathing_signal - baseline, 50))

    peak_idxs, _ = scipy.signal.find_peaks(
        breathing_signal,
        height=treshold,
        prominence=prominence,
        width=width)

    def test_baseline_crossing_starts(self):
        _, peak_start_idxs, _, _, _, _ = evt.onoffpeak_baseline_crossing(
             self.breathing_signal, self.baseline, self.peak_idxs)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_start_idxs),
            )

    def test_baseline_crossing_ends(self):
        _, _, peak_end_idxs, _, _, _ = evt.onoffpeak_baseline_crossing(
             self.breathing_signal, self.baseline, self.peak_idxs)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_end_idxs),
            )

    def test_slope_extrapolate_starts(self):
        peak_start_idxs, _, _, _, _ = evt.onoffpeak_slope_extrapolation(
             self.breathing_signal, self.fs, self.peak_idxs, self.fs//4)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_start_idxs),
            )

    def test_slope_extrapolate_ends(self):
        _, peak_end_idxs, _, _, _ = evt.onoffpeak_slope_extrapolation(
             self.breathing_signal, self.fs, self.peak_idxs, self.fs//4)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_end_idxs),
            )

    def test_detect_ventilator_breath(self):
        ventilator_breath_idxs = evt.detect_ventilator_breath(
            V_signal=self.breathing_signal,
            start_idx=1,
            end_s=10000,
            width_s=1
            )
        self.assertEqual(
            len(ventilator_breath_idxs),
            2,
            )

class TestPoccDetection(unittest.TestCase):
    def test_baseline_crossing_starts(self):
        peak_idxs_detected = evt.find_occluded_breaths(
            p_aw=y_t_paw,
            peep=0,
            fs=fs_vent,
        )

        np.testing.assert_array_equal(
            peak_idxs_detected,
            pocc_peaks_valid,
            )

class TestFindLinkedPeaks(unittest.TestCase):
    def test_find_linked_peaks(self):
        t_1 = [10.0, 15.0, 20.0]
        t_2 = [x + 0.2 for x in range(30)]
        linked_peaks = evt.find_linked_peaks(t_1, t_2)
        np.testing.assert_array_equal(
            linked_peaks,
            np.array([10, 15, 20])
        )

class TestDetectEmgBreaths(unittest.TestCase):
    def test_detect_emg_breaths(self):
        detected_peaks = evt.detect_emg_breaths(
            y_env_emg, y_emg_baseline)
        np.testing.assert_array_equal(
            detected_peaks,
            peaks_env
        )

class TestSnrPseudo(unittest.TestCase):
    fs_emg = 2048
    t_emg = np.array([s_t/fs_emg for s_t in range(15*fs_emg)])

    y_block = np.array(
        10*scipy.signal.square((t_emg - 1.25)/5 * 2 * np.pi, duty=0.5))
    y_block[y_block < 0] = 0
    y_baseline = np.ones(y_block.shape)
    peaks_s = [(5//2 + x*5) * 2048 for x in range(3)]

    snr_values = qa.snr_pseudo(y_block, peaks_s, y_baseline, fs_emg)

    def test_snr_length(self):
        self.assertEqual(
            len(self.snr_values),
            len(self.peaks_s),
            )

    def test_snr_values(self):
        median_snr = np.median(self.snr_values)
        self.assertAlmostEqual(
            median_snr, 10.0, 3
            )

class TestPoccQuality(unittest.TestCase):
    valid_poccs, _ = qa.pocc_quality(
        y_t_paw, pocc_peaks_valid, pocc_ends, PTP_occs)
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

        peaks_steeper, _ = scipy.signal.find_peaks(-y_t_steeper, prominence=0.1)
        y_baseline = bl.moving_baseline(-y_t_steeper, 7.5*fs_vent, fs_vent//5)

        _, peak_starts_steep, peak_ends_steep, _, _, _ = \
            evt.onoffpeak_baseline_crossing(
                y_t_steeper, y_baseline, peaks_steeper)

        ptp_occs_steep = np.zeros(peaks_steeper.shape)
        for idx, _ in enumerate(peaks_steeper):
            ptp_occs_steep[idx] = trapezoid(
                -y_t_steeper[peak_starts_steep[idx]:peak_ends_steep[idx]],
                dx=1/fs_vent
            )

        steep_upslope, _ = qa.pocc_quality(
            y_t_steeper, peaks_steeper, peak_ends_steep, ptp_occs_steep)

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

class TestInterpeakMethods(unittest.TestCase):
    def test_interpeak_dist(self):
        sim_ECG=np.arange(1, 11)
        sim_EMG=np.linspace(1, 10, 4)
        valid_interpeak = qa.interpeak_dist(sim_ECG, sim_EMG, threshold=1.1)

        self.assertTrue(valid_interpeak, "The interpeak_dist function"
                        "did not return True as expected.")


class TestTimeProduct(unittest.TestCase):
    # Define signal
    fs_emg = 2048
    t_emg = np.array([s_t/fs_emg for s_t in range(15*fs_emg)])

    y_block = np.array(
        3*scipy.signal.square((t_emg - 1.25)/5 * 2 * np.pi, duty=0.5))
    y_block[y_block < 0] = 0

    peaks_s = [(5//2 + x*5) * 2048 for x in range(3)]
    starts_s = [(5 + x*5*4) * 2048 //4 for x in range(3)]
    ends_s = [(15 + x*5*4) * 2048 //4 - 1 for x in range(3)]

    y_baseline = np.ones(y_block.shape)

    def test_timeproduct(self):
        aob = feat.time_product(
            self.y_block,
            self.fs_emg,
            self.starts_s,
            self.ends_s,
            self.y_baseline,
        )
        self.assertAlmostEqual(np.median(aob), 5.0, 2)

    def test_area_under_baseline(self):
        aub = feat.area_under_baseline(
            self.y_block,
            self.fs_emg,
            self.peaks_s,
            self.starts_s,
            self.ends_s,
            aub_window_s=self.fs_emg*5,
            baseline=self.y_baseline,
            ref_signal=self.y_block,
        )
        self.assertAlmostEqual(np.median(aub), 2.5, 2)

class TestAreaUnderBaselineQuality(unittest.TestCase):
    # Define signal
    fs_emg = 2048
    t_emg = np.array([s_t/fs_emg for s_t in range(15*fs_emg)])

    y_block = np.array(
        3*scipy.signal.square((t_emg - 1.25)/5 * 2 * np.pi, duty=0.5))
    y_block[y_block < 0] = 0

    peaks_s = [(5//2 + x*5) * 2048 for x in range(3)]
    starts_s = [(5 + x*5*4) * 2048 //4 for x in range(3)]
    ends_s = [(15 + x*5*4) * 2048 //4 - 1 for x in range(3)]

    def test_percentage_aub_good(self):
        y_baseline = np.ones(self.y_block.shape)
        valid_timeproducts, _ = qa.percentage_under_baseline(
            self.y_block,
            self.fs_emg,
            self.peaks_s,
            self.starts_s,
            self.ends_s,
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
        valid_timeproducts, _ = qa.percentage_under_baseline(
            self.y_block,
            self.fs_emg,
            self.peaks_s,
            self.starts_s,
            self.ends_s,
            y_baseline,
            aub_window_s=None,
            ref_signal=None,
            aub_threshold=40,
        )

        self.assertFalse(np.all(valid_timeproducts))


class TestBellFit(unittest.TestCase):
    def test_evaluate_bell_curve_error(self):
        output = qa.evaluate_bell_curve_error(
            peaks_s=peaks_env,
            starts_s=emg_starts_s,
            ends_s=emg_ends_s,
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

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
