"""Offline evaluation utilities: accuracy metrics and synthetic rPPG signals.

Used by the benchmark harness (src/benchmark.py) to measure pipeline accuracy
before/after signal-processing changes.
"""

import numpy as np


def mae(estimates, true_bpm):
    """Mean absolute error of BPM estimates against a reference value."""
    estimates = np.asarray(estimates, dtype=float)
    if estimates.size == 0:
        return float("nan")
    return float(np.mean(np.abs(estimates - true_bpm)))


def rmse(estimates, true_bpm):
    """Root mean square error of BPM estimates against a reference value."""
    estimates = np.asarray(estimates, dtype=float)
    if estimates.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((estimates - true_bpm) ** 2)))


def bias(estimates, true_bpm):
    """Mean signed error (positive = overestimating)."""
    estimates = np.asarray(estimates, dtype=float)
    if estimates.size == 0:
        return float("nan")
    return float(np.mean(estimates - true_bpm))


# Per-channel model for generate_rgb (channel order R, G, B).
# Pulse is strongest in green (hemoglobin absorbs ~530 nm); ratios are relative
# to the green amplitude. Means are a rough warm-skin tone.
_CHANNEL_PULSE_RATIO = (0.33, 1.0, 0.20)
_CHANNEL_MEAN = (150.0, 120.0, 110.0)


class SyntheticSignalGenerator:
    """Generate a realistic green-channel trace containing a known pulse.

    The trace is centered around a typical 8-bit skin brightness (~128) with a
    sub-1% pulse amplitude, optional sensor noise, optional slow baseline
    drift (simulating gradual lighting change), and optional motion spikes
    (brief brightness bumps simulating head movement).
    """

    def __init__(
        self,
        bpm=75,
        sampling_rate=30,
        amplitude=1.5,
        noise_level=0.3,
        drift_amplitude=0.0,
        drift_freq_hz=0.05,
        spike_rate_per_s=0.0,
        spike_amplitude=8.0,
        settle_amplitude=0.0,
        settle_tau_s=1.5,
        harmonic_ratio=0.0,
        bpm_end=None,
        step_time_s=None,
        mean_level=128.0,
        seed=None,
    ):
        self.bpm = bpm
        self.bpm_end = bpm_end  # if set, heart rate steps to this value
        self.step_time_s = step_time_s  # ...at this time (models e.g. standing up)
        self.fs = sampling_rate
        self.amplitude = amplitude
        self.noise_level = noise_level
        self.drift_amplitude = drift_amplitude
        self.drift_freq_hz = drift_freq_hz
        self.spike_rate_per_s = spike_rate_per_s
        self.spike_amplitude = spike_amplitude
        self.settle_amplitude = settle_amplitude  # camera auto-exposure settling
        self.settle_tau_s = settle_tau_s
        self.harmonic_ratio = harmonic_ratio  # 2nd-harmonic content (real PPG ~0.3-0.5)
        self.mean_level = mean_level
        self.rng = np.random.default_rng(seed)

    def generate(self, duration_sec):
        """Return a 1-D green-channel trace of length duration_sec * fs."""
        n = int(self.fs * duration_sec)
        t = np.arange(n) / self.fs

        pulse = self.amplitude * np.sin(2 * np.pi * (self.bpm / 60.0) * t)
        if self.harmonic_ratio:
            pulse += (
                self.amplitude * self.harmonic_ratio * np.sin(4 * np.pi * (self.bpm / 60.0) * t)
            )
        noise = self.rng.normal(0, self.noise_level, size=n)
        drift = self.drift_amplitude * np.sin(2 * np.pi * self.drift_freq_hz * t)
        trace = self.mean_level + pulse + noise + drift

        if self.settle_amplitude:
            # Camera auto-exposure convergence: large brightness decay at start
            trace += self.settle_amplitude * np.exp(-t / self.settle_tau_s)

        if self.spike_rate_per_s > 0:
            n_spikes = int(round(duration_sec * self.spike_rate_per_s))
            samples = np.arange(n)
            for _ in range(n_spikes):
                center = self.rng.uniform(0, n)
                width = self.rng.uniform(2, 6)  # spike std in samples (~0.1-0.2 s)
                sign = 1.0 if self.rng.random() < 0.5 else -1.0
                trace += (
                    sign * self.spike_amplitude * np.exp(-0.5 * ((samples - center) / width) ** 2)
                )

        return trace

    def generate_rgb(self, duration_sec):
        """Return an (N, 3) RGB trace (columns R, G, B) modeling real rPPG colour.

        The pulse appears in every channel (green strongest); intensity
        artifacts (drift, camera auto-exposure settling, motion spikes) are
        common-mode across channels, exactly the structure POS is designed to
        exploit; sensor noise is independent per channel.
        """
        n = int(self.fs * duration_sec)
        t = np.arange(n) / self.fs

        # Instantaneous phase from the (possibly stepping) heart rate. Using
        # cumulative phase keeps the waveform continuous across a rate step.
        if self.bpm_end is not None and self.step_time_s is not None:
            inst_hz = np.where(t < self.step_time_s, self.bpm / 60.0, self.bpm_end / 60.0)
            phase = 2 * np.pi * np.cumsum(inst_hz) / self.fs
        else:
            phase = 2 * np.pi * (self.bpm / 60.0) * t

        # Unit-amplitude pulse (with optional 2nd harmonic), scaled per channel
        pulse = np.sin(phase)
        if self.harmonic_ratio:
            pulse += self.harmonic_ratio * np.sin(2 * phase)
        pulse *= self.amplitude

        # Common-mode intensity artifacts shared by all channels
        common = self.drift_amplitude * np.sin(2 * np.pi * self.drift_freq_hz * t)
        if self.settle_amplitude:
            common = common + self.settle_amplitude * np.exp(-t / self.settle_tau_s)
        if self.spike_rate_per_s > 0:
            samples = np.arange(n)
            for _ in range(int(round(duration_sec * self.spike_rate_per_s))):
                center = self.rng.uniform(0, n)
                width = self.rng.uniform(2, 6)
                sign = 1.0 if self.rng.random() < 0.5 else -1.0
                common = common + sign * self.spike_amplitude * np.exp(
                    -0.5 * ((samples - center) / width) ** 2
                )

        channels = []
        for mean, ratio in zip(_CHANNEL_MEAN, _CHANNEL_PULSE_RATIO, strict=True):
            noise = self.rng.normal(0, self.noise_level, size=n)
            channels.append(mean + ratio * pulse + common + noise)
        return np.stack(channels, axis=1)
