"""Tests for the preprocessing chain (detrend -> band-pass -> robust normalize)."""

import numpy as np
from scipy.signal import welch

from signal_processing.preprocessing import SignalPreprocessor

FS = 30


def test_bandpass_preserves_pulse_and_removes_drift():
    t = np.arange(FS * 10) / FS
    pulse = np.sin(2 * np.pi * (72 / 60.0) * t)
    drift = 5.0 * np.sin(2 * np.pi * 0.05 * t)  # slow lighting drift, out of band
    pre = SignalPreprocessor(FS)
    out = pre.enhance_heart_rate_signal(pulse + drift + 128.0, fs=FS)

    freqs, power = welch(out, fs=FS, nperseg=min(256, len(out)), nfft=2**12)
    band = (freqs >= 0.7) & (freqs <= 3.0)
    peak_bpm = freqs[band][np.argmax(power[band])] * 60.0
    assert abs(peak_bpm - 72.0) < 3.0


def test_normalize_robust_clips_outliers():
    pre = SignalPreprocessor(FS)
    values = np.concatenate([np.random.default_rng(0).normal(0, 1, 200), [1000.0]])
    out = pre.normalize_robust(values)
    assert np.all(np.abs(out) <= pre.OUTLIER_CLIP_MAD + 1e-9)


def test_bandpass_redesigned_on_fs_change():
    pre = SignalPreprocessor(FS)
    assert pre.design_fs == 30.0
    signal = np.sin(2 * np.pi * (72 / 60.0) * np.arange(FS * 10) / FS)
    pre.enhance_heart_rate_signal(signal, fs=60.0)
    assert pre.design_fs == 60.0  # coefficients were re-derived for the new rate
