"""Offline evaluation utilities: accuracy metrics and synthetic rPPG signals.

Used by the benchmark harness (src/benchmark.py) to measure pipeline accuracy
before/after signal-processing changes.
"""

import numpy as np


def mae(estimates, true_bpm):
    """Mean absolute error of BPM estimates against a reference value."""
    estimates = np.asarray(estimates, dtype=float)
    if estimates.size == 0:
        return float('nan')
    return float(np.mean(np.abs(estimates - true_bpm)))


def rmse(estimates, true_bpm):
    """Root mean square error of BPM estimates against a reference value."""
    estimates = np.asarray(estimates, dtype=float)
    if estimates.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean((estimates - true_bpm) ** 2)))


def bias(estimates, true_bpm):
    """Mean signed error (positive = overestimating)."""
    estimates = np.asarray(estimates, dtype=float)
    if estimates.size == 0:
        return float('nan')
    return float(np.mean(estimates - true_bpm))


class SyntheticSignalGenerator:
    """Generate a realistic green-channel trace containing a known pulse.

    The trace is centered around a typical 8-bit skin brightness (~128) with a
    sub-1% pulse amplitude, optional sensor noise, and optional slow baseline
    drift (simulating gradual lighting change).
    """

    def __init__(self, bpm=75, sampling_rate=30, amplitude=1.5, noise_level=0.3,
                 drift_amplitude=0.0, drift_freq_hz=0.05, mean_level=128.0, seed=None):
        self.bpm = bpm
        self.fs = sampling_rate
        self.amplitude = amplitude
        self.noise_level = noise_level
        self.drift_amplitude = drift_amplitude
        self.drift_freq_hz = drift_freq_hz
        self.mean_level = mean_level
        self.rng = np.random.default_rng(seed)

    def generate(self, duration_sec):
        """Return a 1-D green-channel trace of length duration_sec * fs."""
        n = int(self.fs * duration_sec)
        t = np.arange(n) / self.fs

        pulse = self.amplitude * np.sin(2 * np.pi * (self.bpm / 60.0) * t)
        noise = self.rng.normal(0, self.noise_level, size=n)
        drift = self.drift_amplitude * np.sin(2 * np.pi * self.drift_freq_hz * t)

        return self.mean_level + pulse + noise + drift
