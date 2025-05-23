import numpy as np
import pytest
from analisis import (
    calculate_height,
    find_signal_peaks,
    calculate_omega_and_period,
    get_initial_amplitude,
    amplitude_over_time
)

def test_calculate_height():
    signal = np.array([0, 1, 2, 3, 4, 5])
    expected = 0.5 * (5 - 0) + 0  # 2.5
    result = calculate_height(signal)
    assert result == expected, f"Expected height {expected}, got {result}"

def test_calculate_height_negative():
    signal = np.array([-5, -4, -3, -2, -1, 0])
    expected = 0.5 * (0 - (-5)) + (-5)  # -2.5
    result = calculate_height(signal)
    assert result == expected, f"Expected height {expected}, got {result}"
 
def test_find_signal_peaks_basic():
    signal = np.array([0, 2, 1, 3, 0, 1, 0])
    height = 1.5
    distance = 1
    peaks = find_signal_peaks(signal, height, distance)
    expected_peaks = np.array([1, 3])
    np.testing.assert_array_equal(peaks, expected_peaks)

def test_calculate_omega_and_period_basic():
    t = np.array([0, 1, 2, 3, 4, 5])
    peaks = np.array([1, 3, 5])
    omega, sigma_omega, period, sigma_period = calculate_omega_and_period(t, peaks)
    assert np.isclose(period, 2.0), f"Expected period ~2.0, got {period}"
    assert omega > 0, f"Expected positive omega, got {omega}"
    assert sigma_omega >= 0, f"Expected non-negative sigma_omega, got {sigma_omega}"
    assert sigma_period >= 0, f"Expected non-negative sigma_period, got {sigma_period}"

def test_calculate_omega_and_period_not_enough_peaks():
    t = np.array([0, 1, 2, 3, 4, 5])
    peaks = np.array([1])  # Only one peak
    omega, sigma_omega, period, sigma_period = calculate_omega_and_period(t, peaks)
    assert np.isnan(omega), "Expected NaN omega for insufficient peaks"
    assert np.isnan(sigma_omega), "Expected NaN sigma_omega for insufficient peaks"
    assert np.isnan(period), "Expected NaN period for insufficient peaks"
    assert np.isnan(sigma_period), "Expected NaN sigma_period for insufficient peaks"

def test_get_initial_amplitude():
    signal = np.array([0, -2, 3, -1])
    peaks = np.array([1, 2, 3])
    amp = get_initial_amplitude(signal, peaks)
    expected = 2
    assert amp == expected, f"Expected amplitude {expected}, got {amp}"

def test_get_initial_amplitude_no_peaks():
    signal = np.array([0, 1, 2])
    peaks = np.array([])
    amp = get_initial_amplitude(signal, peaks)
    assert np.isnan(amp), "Expected NaN amplitude when no peaks"

def test_amplitude_over_time():
    signal = np.array([0, -2, 3, -1])
    t = np.array([0, 1, 2, 3])
    peaks = np.array([1, 2])
    times, amps = amplitude_over_time(signal, t, peaks)
    np.testing.assert_array_equal(times, np.array([1, 2]))
    np.testing.assert_array_equal(amps, np.array([2, 3]))
