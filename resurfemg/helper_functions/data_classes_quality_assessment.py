"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains the quality assessment methods for data classes.
"""
import numpy as np
import pandas as pd
from resurfemg.postprocessing import quality_assessment as qa


def initialize_emg_tests(
    timeseries,
    peak_set_name,
    cutoff,
    skip_tests,
    parameter_names
):
    """
    Initialize local parameters. See TimeSeries.test_emg_quality method in
    resurfemg.data_connector.data_classes for more information.
    """
    if peak_set_name in timeseries.peaks:
        peak_set = timeseries.peaks[peak_set_name]
    else:
        raise KeyError("Non-existent PeaksSet key")

    if skip_tests is None:
        skip_tests = []

    if (cutoff is None
            or (isinstance(cutoff, str) and cutoff == 'tolerant')):
        cutoff = {
            'interpeak_distance': 1.1,
            'snr': 1.4,
            'aub': 40,
            'curve_fit': 30,
            'aub_window_s': 5*timeseries.param['fs'],
            'bell_window_s': 5*timeseries.param['fs'],
            'relative_aub_percentile': 75.0,
            'relative_aub_factor': 4.0,
            'relative_etp_upper_percentile': 95.0,
            'relative_etp_upper_factor': 10.0,
            'relative_etp_lower_percentile': 5.0,
            'relative_etp_lower_factor': 0.1,
        }
    elif (isinstance(cutoff, str) and cutoff == 'strict'):
        cutoff = {
            'interpeak_distance': 1.1,
            'snr': 1.75,
            'aub': 30,
            'curve_fit': 25,
            'aub_window_s': 5*timeseries.param['fs'],
            'bell_window_s': 5*timeseries.param['fs'],
            'relative_aub_percentile': 75.0,
            'relative_aub_factor': 2.0,
            'relative_etp_upper_percentile': 95.0,
            'relative_etp_upper_factor': 10.0,
            'relative_etp_lower_percentile': 5.0,
            'relative_etp_lower_factor': 0.1,
        }
    elif isinstance(cutoff, dict):
        tests = [
            'interpeak_distance', 'snr', 'aub', 'curve_fit', ]
        for _, test in enumerate(tests):
            if test not in skip_tests:
                if test not in cutoff:
                    raise KeyError(
                        'No cut-off value provided for: ' + test)
                elif isinstance(cutoff, float):
                    raise ValueError(
                        'Invalid cut-off value provided for: ' + test)

        if 'aub' not in skip_tests and 'aub_window_s' not in cutoff:
            raise KeyError('No area under the baseline window provided'
                           + ' for: ' + test)
        if 'relative_aub' not in skip_tests:
            if 'relative_aub_percentile' not in cutoff:
                raise KeyError('No relative area under the baseline percentile'
                               + ' provided for: relative_aub')
            if 'relative_aub_factor' not in cutoff:
                raise KeyError('No relative area under the baseline factor '
                               + 'provided for: relative_aub')
        if 'relative_etp' not in skip_tests:
            _params = [
                'upper_percentile', 'upper_factor', 'lower_percentile',
                'lower_factor']
            for _param in _params:
                if ('relative_etp_' + _param) not in cutoff:
                    raise KeyError(f"No relative_etp_{_param} "
                                   + "provided for: relative_aub")

    if parameter_names is None:
        parameter_names = dict()

    for parameter in ['ecg', 'time_product']:
        if parameter not in parameter_names:
            parameter_names[parameter] = parameter

    n_peaks = len(peak_set.peak_df['peak_idx'].to_numpy())
    quality_values_df = pd.DataFrame(
        data=peak_set.peak_df['peak_idx'], columns=['peak_idx'])
    quality_outcomes_df = pd.DataFrame(
        data=peak_set.peak_df['peak_idx'], columns=['peak_idx'])
    output = (
        skip_tests,
        cutoff,
        peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df
    )
    return output


def test_interpeak_distance(
        timeseries, peak_set, quality_outcomes_df, n_peaks, cutoff):
    """Test interpeak distance. See TimeSeries.test_emg_quality method in
    resurfemg.data_connector.data_classes for more information."""
    if 'ecg' not in timeseries.peaks:
        raise ValueError('ECG peaks not determined, but required for interpeak'
                         + ' distance evaluation.')
    ecg_peaks = timeseries.peaks['ecg'].peak_df['peak_idx'].to_numpy()
    valid_interpeak = qa.interpeak_dist(
        ECG_peaks=ecg_peaks,
        EMG_peaks=peak_set.peak_df['peak_idx'].to_numpy(),
        threshold=cutoff['interpeak_distance'])

    valid_interpeak_vec = np.array(n_peaks * [valid_interpeak])
    quality_outcomes_df['interpeak_distance'] = valid_interpeak_vec
    return quality_outcomes_df


def test_snr(
        timeseries, peak_set, quality_outcomes_df, quality_values_df, cutoff):
    """Test signal-to-noise ratio. See TimeSeries.test_emg_quality method in
    resurfemg.data_connector.data_classes for more information."""
    if timeseries.baseline is None:
        raise ValueError('Baseline not determined, but required for '
                         + ' SNR evaluaton.')
    snr_peaks = qa.snr_pseudo(
        src_signal=peak_set.signal,
        peaks=peak_set.peak_df['peak_idx'].to_numpy(),
        baseline=timeseries.y_baseline,
        fs=timeseries.param['fs'],
    )
    quality_values_df['snr'] = snr_peaks
    valid_snr = snr_peaks > cutoff['snr']
    quality_outcomes_df['snr'] = valid_snr
    return quality_outcomes_df


def test_aub(
        timeseries, peak_set, quality_outcomes_df, quality_values_df, cutoff):
    """Test percentage area under the baselineSee TimeSeries.test_emg_quality
    method in resurfemg.data_connector.data_classes for more information."""
    if timeseries.baseline is None:
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
        fs=timeseries.param['fs'],
        peak_idxs=peak_set.peak_df['peak_idx'].to_numpy(),
        start_idxs=peak_set.peak_df['start_idx'].to_numpy(),
        end_idxs=peak_set.peak_df['end_idx'].to_numpy(),
        baseline=timeseries.y_baseline,
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
        timeseries, peak_set, quality_outcomes_df, quality_values_df, cutoff,
        parameter_names):
    """Test curve fit. See TimeSeries.test_emg_quality method in
    resurfemg.data_connector.data_classes for more information."""
    if timeseries.baseline is None:
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
        fs=timeseries.param['fs'],
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
    """Test the relative area under the baseline. See
    TimeSeries.test_emg_quality method in resurfemg.data_connector.data_classes
    for more information."""
    if 'AUB' not in peak_set.peak_df.columns:
        raise ValueError('AUB not determined, but required for relative area '
                         + 'under the baseline (AUB) evaluaton.')
    valid_relative_aubs = qa.detect_local_high_aub(
        aubs=peak_set.peak_df['AUB'].to_numpy(),
        threshold_percentile=cutoff['relative_aub_percentile'],
        threshold_factor=cutoff['relative_aub_factor'],
    )
    quality_outcomes_df['relative_aub'] = valid_relative_aubs
    return quality_outcomes_df


def test_relative_etp(peak_set, quality_outcomes_df, cutoff, parameter_names):
    """Evaluate extremely low and high timeproducts. See
    TimeSeries.test_emg_quality method in resurfemg.data_connector.data_classes
    for more information."""
    if parameter_names['time_product'] not in peak_set.peak_df.columns:
        raise ValueError('ETPs not determined, but required for curve '
                         + 'fit evaluaton.')
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


def initialize_pocc_tests(
    timeseries,
    peak_set_name,
    cutoff,
    skip_tests,
    parameter_names
):
    """Initialize local parameters. See TimeSeries.test_pocc_quality method in
    resurfemg.data_connector.data_classes for more information.
    """
    if peak_set_name in timeseries.peaks:
        peak_set = timeseries.peaks[peak_set_name]
    else:
        raise KeyError("Non-existent PeaksSet key")

    if skip_tests is None:
        skip_tests = []

    if (cutoff is None
            or (isinstance(cutoff, str) and cutoff == 'tolerant')
            or (isinstance(cutoff, str) and cutoff == 'strict')):
        cutoff = dict()
        cutoff['consecutive_poccs'] = 0
        cutoff['dP_up_10'] = 0.0
        cutoff['dP_up_90'] = 2.0
        cutoff['dP_up_90_norm'] = 0.8
    elif isinstance(cutoff, dict):
        tests = ['consecutive_poccs', 'pocc_upslope']
        tests_crit = dict()
        tests_crit['consecutive_poccs'] = ['consecutive_poccs']
        tests_crit['pocc_upslope'] = [
            'dP_up_10', 'dP_up_90', 'dP_up_90_norm']

        for _, test in enumerate(tests):
            if test not in skip_tests:
                for test_crit in tests_crit[test]:
                    if test_crit not in cutoff:
                        raise KeyError(
                            'No cut-off value provided for: ' + test)

    if parameter_names is None:
        parameter_names = dict()

    for parameter in ['ventilator_breaths', 'time_product', 'AUB']:
        if parameter not in parameter_names:
            parameter_names[parameter] = parameter

    n_peaks = len(peak_set.peak_df['peak_idx'].to_numpy())
    quality_values_df = pd.DataFrame(data=peak_set.peak_df['peak_idx'],
                                     columns=['peak_idx'])
    quality_outcomes_df = pd.DataFrame(data=peak_set.peak_df['peak_idx'],
                                       columns=['peak_idx'])
    output = (
        skip_tests,
        cutoff,
        peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df
    )
    return output


def test_consecutive_poccs(
        timeseries, peak_set, quality_outcomes_df, parameter_names):
    """Test for consecutive Pocc manoeuvres. See TimeSeries.test_pocc_quality
    method in resurfemg.data_connector.data_classes for more information."""
    if parameter_names['ventilator_breaths'] not in timeseries.peaks:
        raise ValueError('Ventilator breaths not determined, but required for '
                         + 'consecutive Pocc evaluation.')
    vent_breaths = parameter_names['ventilator_breaths']
    ventilator_breath_idxs = \
        timeseries.peaks[vent_breaths].peak_df['peak_idx'].to_numpy()
    valid_manoeuvres = qa.detect_non_consecutive_manoeuvres(
        ventilator_breath_idxs=ventilator_breath_idxs,
        manoeuvres_idxs=peak_set.peak_df['peak_idx'].to_numpy(),
    )
    quality_outcomes_df['consecutive_poccs'] = valid_manoeuvres
    return quality_outcomes_df


def test_pocc_upslope(
    timeseries, peak_set, quality_outcomes_df, quality_values_df, cutoff,
    parameter_names
):
    """Test for sudden Pocc release. See TimeSeries.test_pocc_quality method in
    resurfemg.data_connector.data_classes for more information."""
    if 'end_idx' not in peak_set.peak_df.columns:
        raise ValueError('Pocc end_idx not determined, but required for Pocc '
                         + 'upslope evaluaton.')
    if parameter_names['time_product'] not in peak_set.peak_df.columns:
        raise ValueError('PTPs not determined, but required for Pocc upslope '
                         + 'evaluaton.')
    valid_poccs, criteria_matrix = qa.pocc_quality(
        p_vent_signal=timeseries.y_raw,
        pocc_peaks=peak_set.peak_df['peak_idx'].to_numpy(),
        pocc_ends=peak_set.peak_df['end_idx'].to_numpy(),
        ptp_occs=peak_set.peak_df[
            parameter_names['time_product']].to_numpy(),
        dp_up_10_threshold=cutoff['dP_up_10'],
        dp_up_90_threshold=cutoff['dP_up_90'],
        dp_up_90_norm_threshold=cutoff['dP_up_90_norm'],
    )
    quality_values_df['dP_up_10'] = criteria_matrix[0, :]
    quality_values_df['dP_up_90'] = criteria_matrix[1, :]
    quality_values_df['dP_up_90_norm'] = criteria_matrix[2, :]

    quality_outcomes_df['pocc_upslope'] = valid_poccs
    return quality_outcomes_df, quality_values_df


def initialize_linked_peaks_tests(
    timeseries,
    peak_set_name,
    linked_timeseries,
    linked_peak_set_name,
    cutoff,
    skip_tests,
    parameter_names,
):
    """Initialize local parameter. See TimeSeries.test_linked_peak_sets method
    in resurfemg.data_connector.data_classes for more information."""
    if peak_set_name in timeseries.peaks:
        peak_set = timeseries.peaks[peak_set_name]
    else:
        raise KeyError("Non-existent PeaksSet key")

    if linked_peak_set_name in linked_timeseries.peaks:
        linked_peak_set = linked_timeseries.peaks[linked_peak_set_name]
    else:
        raise KeyError("Non-existent linked PeaksSet key")

    if skip_tests is None:
        skip_tests = []

    if (cutoff is None) or (isinstance(cutoff, str) and cutoff == 'tolerant'):
        cutoff = dict()
        cutoff['fraction_emg_breaths'] = 0.1
        cutoff['delta_min'] = -0.5
        cutoff['delta_max'] = 2.0
    elif (isinstance(cutoff, str) and cutoff == 'strict'):
        cutoff['fraction_emg_breaths'] = 0.5
        cutoff['delta_min'] = -0.5
        cutoff['delta_max'] = 2.0
    elif isinstance(cutoff, dict):
        tests = ['fraction_emg_breaths', 'event_timing']
        tests_crit = dict()
        tests_crit['fraction_emg_breaths'] = ['fraction_emg_breaths']
        tests_crit['event_timing'] = ['delta_min', 'delta_max']

        for _, test in enumerate(tests):
            if test not in skip_tests:
                for crit in tests_crit[test]:
                    if crit not in cutoff:
                        raise KeyError(
                            'No cut-off value provided for: ' + test)

    if parameter_names is None:
        parameter_names = dict()

    for parameter in ['ventilator_breaths', 'rr']:
        if parameter not in parameter_names:
            parameter_names[parameter] = parameter

    n_peaks = len(peak_set.peak_df['peak_idx'].to_numpy())
    quality_values_df = pd.DataFrame(data=peak_set.peak_df['peak_idx'],
                                     columns=['peak_idx'])
    quality_outcomes_df = pd.DataFrame(data=peak_set.peak_df['peak_idx'],
                                       columns=['peak_idx'])
    output = (
        skip_tests,
        cutoff,
        peak_set,
        linked_peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df
    )
    return output


def test_fraction_detected_breaths(
    native_peak_set, linked_time_series, quality_outcomes_df,
    quality_values_df, n_peaks, cutoff, parameter_names
):
    """Test detected peak fraction. See TimeSeries.test_linked_peak_sets method
    in resurfemg.data_connector.data_classes for more information."""
    native_peak_idxs = native_peak_set.peak_df['peak_idx'].to_numpy()

    fraction_emg_breaths, _ = qa.evaluate_respiratory_rates(
        emg_breath_idxs=native_peak_idxs,
        t_emg=max(native_peak_set.t_data),
        rr_vent=linked_time_series.param[parameter_names['rr']],
    )
    valid_fraction = fraction_emg_breaths > cutoff['fraction_emg_breaths']
    fraction_emg_breaths_vec = np.array(n_peaks * [fraction_emg_breaths])
    valid_fraction_vec = np.array(n_peaks * [valid_fraction])
    quality_outcomes_df['detected_fraction'] = valid_fraction_vec
    quality_values_df['detected_fraction'] = fraction_emg_breaths_vec
    return quality_outcomes_df, quality_values_df


def test_event_timing(
    timeseries, native_peak_set, linked_timeseries, linked_peak_set,
    quality_outcomes_df, quality_values_df, cutoff
):
    """Test event timing relative to other peak_set. See
    TimeSeries.test_linked_peak_sets method in
    resurfemg.data_connector.data_classes for more information."""
    native_peak_idxs = native_peak_set.peak_df['peak_idx'].to_numpy()
    linked_peak_idxs = linked_peak_set.peak_df['peak_idx'].to_numpy()

    t_events_1 = native_peak_idxs / timeseries.param['fs']
    t_events_2 = linked_peak_idxs / linked_timeseries.param['fs']

    correct_timing, delta_time = qa.evaluate_event_timing(
        t_events_1,
        t_events_2,
        delta_min=cutoff['delta_min'],
        delta_max=cutoff['delta_max']
    )
    quality_outcomes_df['event_timing'] = correct_timing
    quality_values_df['event_delta_time'] = delta_time
    return quality_outcomes_df, quality_values_df
