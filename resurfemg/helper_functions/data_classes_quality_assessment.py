"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
This file contains the quality assessment methods for data classes.
"""

from resurfemg.postprocessing import quality_assessment as qa

def test_snr(self, peak_set, quality_outcomes_df, quality_values_df, cutoff):
    """Test signal-to-noise ratio"""
    if self.baseline is None:
        raise ValueError('Baseline not determined, but required for '
                         + ' SNR evaluaton.')
    snr_peaks = qa.snr_pseudo(
        src_signal=peak_set.signal,
        peaks=peak_set.peak_df['peak_idx'].to_numpy(),
        baseline=self.y_baseline,
        fs=self.fs,
    )
    quality_values_df['snr'] = snr_peaks
    valid_snr = snr_peaks > cutoff['snr']
    quality_outcomes_df['snr'] = valid_snr
    return quality_outcomes_df

def test_aub(self, peak_set, quality_outcomes_df, quality_values_df, cutoff):
    """Test percentage area under the baseline"""
    if self.baseline is None:
        raise ValueError('Baseline not determined, but required for '
                         + ' area under the baseline (AUB) evaluaton.')
    if 'start_idx' not in peak_set.peak_df.columns:
        raise ValueError('start_idxs not determined, but required for '
                         + ' area under the baseline (AUB) evaluaton.')
    if 'end_idx' not in peak_set.peak_df.columns:
        raise ValueError('end_idxs not determined, but required for '
                         + ' area under the baseline (AUB) evaluaton.')
    outputs = qa.percentage_under_baseline(
        signal=peak_set.signal,
        fs=self.fs,
        peak_idxs=peak_set.peak_df['peak_idx'].to_numpy(),
        start_idxs=peak_set.peak_df['start_idx'].to_numpy(),
        end_idxs=peak_set.peak_df['end_idx'].to_numpy(),
        baseline=self.y_baseline,
        aub_window_s=None,
        ref_signal=None,
        aub_threshold=cutoff['aub'],
    )

    (valid_timeproducts, percentages_aub, y_refs) = outputs
    quality_values_df['aub'] = percentages_aub
    quality_values_df['aub_y_refs'] = y_refs
    quality_outcomes_df['aub'] = valid_timeproducts
    return quality_outcomes_df, quality_values_df

def test_curve_fits(
        self, peak_set, quality_outcomes_df, quality_values_df, cutoff,
        parameter_names):
    """Test curve fit"""
    if self.baseline is None:
        raise ValueError('Baseline not determined, but required for '
                         + ' area under the baseline (AUB) evaluaton.')
    if 'start_idx' not in peak_set.peak_df.columns:
        raise ValueError('start_idxs not determined, but required for '
                         + ' curve fit evaluaton.')
    if 'end_idx' not in peak_set.peak_df.columns:
        raise ValueError('end_idxs not determined, but required for '
                         + ' curve fit evaluaton.')
    if parameter_names['time_product'] not in peak_set.peak_df.columns:
        raise ValueError('ETPs not determined, but required for curve '
                         + 'fit evaluaton.')
    outputs = qa.evaluate_bell_curve_error(
        peak_idxs=peak_set.peak_df['peak_idx'].to_numpy(),
        start_idxs=peak_set.peak_df['start_idx'].to_numpy(),
        end_idxs=peak_set.peak_df['end_idx'].to_numpy(),
        signal=peak_set.signal,
        fs=self.fs,
        time_products=peak_set.peak_df[
            parameter_names['time_product']].to_numpy(),
        bell_window_s=cutoff['bell_window_s'],
        bell_threshold=cutoff['curve_fit'],
    )
    (valid_bell_shape,
     _,
     percentage_bell_error,
     y_min,
     parameters) = outputs

    peak_set.peak_df['bell_y_min'] = y_min
    for idx, parameter in enumerate(['a', 'b', 'c']):
        peak_set.peak_df['bell_' + parameter] = parameters[:, idx]

    quality_values_df['bell'] = percentage_bell_error
    quality_outcomes_df['bell'] = valid_bell_shape
    return quality_outcomes_df, quality_values_df, peak_set

def test_relative_aub(peak_set, quality_outcomes_df, cutoff):
    """Test the relative area under the baseline"""
    if 'AUB' not in peak_set.peak_df.columns:
        raise ValueError('AUB not determined, but required for '
                         +'relative area under the baseline (AUB) '
                         +'evaluaton.')
    valid_relative_aubs = qa.detect_local_high_aub(
        aubs=peak_set.peak_df['AUB'].to_numpy(),
        threshold_percentile=cutoff['relative_aub_percentile'],
        threshold_factor=cutoff['relative_aub_factor'],
    )
    quality_outcomes_df['relative_aub'] = valid_relative_aubs
    return quality_outcomes_df


def test_relative_etp(peak_set, quality_outcomes_df, cutoff,
        parameter_names):
    """Evaluate extremely low and high timeproducts"""
    if parameter_names['time_product'] not in peak_set.peak_df.columns:
        raise ValueError('ETPs not determined, but required for curve '
                         + 'fit evaluaton.')
    print(peak_set.peak_df[
            parameter_names['time_product']].to_numpy().shape)
    valid_etps = qa.detect_extreme_time_products(
        time_products=peak_set.peak_df[
            parameter_names['time_product']].to_numpy(),
        upper_percentile=cutoff['relative_etp_upper_percentile'],
        upper_factor=cutoff['relative_etp_upper_factor'],
        lower_percentile=cutoff['relative_etp_lower_percentile'],
        lower_factor=cutoff['relative_etp_lower_factor'],
    )
    quality_outcomes_df['relative_etp'] = valid_etps
    return quality_outcomes_df
