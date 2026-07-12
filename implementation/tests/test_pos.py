"""Tests for POS colour-projection pulse extraction."""

import numpy as np
import pytest
from scipy.signal import welch

from signal_processing.pos import pos_pulse

FS = 30


def _peak_bpm(signal, fs=FS):
    freqs, power = welch(signal, fs=fs, nperseg=min(256, len(signal)), nfft=2**12)
    band = (freqs >= 0.7) & (freqs <= 3.0)
    return freqs[band][np.argmax(power[band])] * 60.0


def _rgb_trace(bpm, seconds=12, common=None, seed=0):
    """Build an (N,3) RGB trace: pulse in all channels (green strongest)."""
    rng = np.random.default_rng(seed)
    n = FS * seconds
    t = np.arange(n) / FS
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)
    if common is None:
        common = np.zeros(n)
    r = 150 + 0.33 * pulse + common + rng.normal(0, 0.3, n)
    g = 120 + 1.00 * pulse + common + rng.normal(0, 0.3, n)
    b = 110 + 0.20 * pulse + common + rng.normal(0, 0.3, n)
    return np.stack([r, g, b], axis=1)


def test_recovers_known_frequency():
    pulse = pos_pulse(_rgb_trace(72), FS)
    assert abs(_peak_bpm(pulse) - 72.0) < 3.0


def test_suppresses_common_mode_better_than_green():
    # A large common-mode spike + drift: POS should track the pulse better
    # than the raw green channel, which sees the full artifact.
    n = FS * 12
    common = 3.0 * np.sin(2 * np.pi * 0.05 * np.arange(n) / FS)
    common[150:158] += 25.0
    rgb = _rgb_trace(72, common=common)
    pos_err = abs(_peak_bpm(pos_pulse(rgb, FS)) - 72.0)
    green_err = abs(_peak_bpm(rgb[:, 1]) - 72.0)
    assert pos_err <= green_err
    assert pos_err < 3.0


def test_short_input_uses_global_projection():
    rgb = _rgb_trace(72, seconds=1)  # shorter than the 1.6 s window
    out = pos_pulse(rgb, FS)
    assert out.shape == (rgb.shape[0],)
    assert np.all(np.isfinite(out))


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        pos_pulse(np.zeros(100), FS)
