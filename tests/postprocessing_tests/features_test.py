"""
sanity tests for the postprocessing.features submodule of the resurfemg 
library
"""
import unittest
import os
import numpy as np
import scipy
from scipy.integrate import trapezoid
import resurfemg.preprocessing.envelope as evl
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
t_emg = np.array([s_t/fs_emg for s_t in range(10*fs_emg)])
y_sin = np.cos((0.5* t_emg - 0.5)* 2 * np.pi)
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
rr = 12
t_r = 60/rr
f_r = 1/t_r
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

class TestArrayMath(unittest.TestCase):
    def test_times_under_curve(self):
        sample_array= np.array(
            [0,1,2,3,1,5,6,-5,8,9,20,11,12,13,4,5,6,1,1,1,0]
        )
        counted = feat.times_under_curve(sample_array,0,20)
        self.assertEqual(
            counted,
            ((10,0.5)),
        )

    def test_pseudo_slope(self):
        test_arr_1 = np.array(
            [0,9,8,7,10,11,13,15,12,16,13,17,18,6,5,4,3,2,1,0,0,0,0,0]
        )
        slope = feat.pseudo_slope(test_arr_1,0,17)
        self.assertEqual(
            slope,
            1.5
        )


class TestTimeProduct(unittest.TestCase):
    # Define signal
    fs_emg = 2048
    t_emg = np.array([s_t/fs_emg for s_t in range(15*fs_emg)])

    y_block = np.array(
        3*scipy.signal.square((t_emg - 1.25)/5 * 2 * np.pi, duty=0.5))
    y_block[y_block < 0] = 0

    peak_idxs = [(5//2 + x*5) * 2048 for x in range(3)]
    start_idxs = [(5 + x*5*4) * 2048 //4 for x in range(3)]
    end_idxs = [(15 + x*5*4) * 2048 //4 - 1 for x in range(3)]

    y_baseline = np.ones(y_block.shape)

    def test_timeproduct(self):
        aob = feat.time_product(
            self.y_block,
            self.fs_emg,
            self.start_idxs,
            self.end_idxs,
            self.y_baseline,
        )
        self.assertAlmostEqual(np.median(aob), 5.0, 2)

    def test_area_under_baseline(self):
        aub, _ = feat.area_under_baseline(
            self.y_block,
            self.fs_emg,
            self.peak_idxs,
            self.start_idxs,
            self.end_idxs,
            aub_window_s=self.fs_emg*5,
            baseline=self.y_baseline,
            ref_signal=self.y_block,
        )
        self.assertAlmostEqual(np.median(aub), 2.5, 2)


class TestRespiratoryRate(unittest.TestCase):
    peak_idxs = [(5//2 + x*5) * 2048 for x in range(3)]
    def test_rr(self):
        rr_median, _ = feat.respiratory_rate(
            self.peak_idxs,
            fs_emg)
        self.assertAlmostEqual(rr_median, 12, 2)


if __name__ == '__main__':
    unittest.main()
