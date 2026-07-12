"""Tests for the Welch-PSD heart rate estimator."""

import numpy as np

from signal_processing.hr_estimation import HeartRateEstimator

FS = 30


def _sinusoid(bpm, seconds=10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(FS * seconds) / FS
    return np.sin(2 * np.pi * (bpm / 60.0) * t) + rng.normal(0, 0.1, len(t))


def test_recovers_known_bpm():
    est = HeartRateEstimator(FS)
    bpm, conf = est.estimate(_sinusoid(72), fs=FS)
    assert bpm is not None
    assert abs(bpm - 72.0) < 3.0
    assert conf > 0.4


def test_respects_measured_fs():
    # Same samples, different declared rate -> proportionally different BPM.
    est = HeartRateEstimator(FS)
    signal = _sinusoid(72)
    bpm30, _ = est.estimate(signal, fs=30)
    bpm60, _ = est.estimate(signal, fs=60)
    assert bpm60 > bpm30 * 1.8  # doubling fs roughly doubles the reported BPM


def test_confidence_in_unit_interval():
    est = HeartRateEstimator(FS)
    _, conf = est.estimate(_sinusoid(90), fs=FS)
    assert 0.0 <= conf <= 1.0


def test_empty_signal_returns_none():
    est = HeartRateEstimator(FS)
    bpm, conf = est.estimate(np.array([]), fs=FS)
    assert bpm is None
    assert conf == 0.0
