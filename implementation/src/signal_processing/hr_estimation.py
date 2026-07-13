"""Frequency-domain heart rate estimation (see HeartRateEstimator)."""

import numpy as np
from scipy.signal import find_peaks, welch


class HeartRateEstimator:
    """Estimate heart rate from a cleaned rPPG signal via Welch PSD peak-picking.

    The estimator does one thing: report the dominant heart-rate frequency and
    a confidence for the current window. Temporal smoothing and outlier
    rejection live downstream in HeartRateFilter, so responsiveness is tuned in
    one place rather than spread across several interacting layers.
    """

    def __init__(self, sampling_rate):
        self.fs = sampling_rate
        self.min_hr_hz = 40 / 60.0
        self.max_hr_hz = 180 / 60.0
        self.peak_prominence_factor = 0.15  # min peak prominence vs max band power
        self.physiological_range = (40, 180)

    def estimate(self, signal, fs=None):
        """Return (bpm, confidence) for the current window, or (None, 0.0).

        `fs` is the measured sampling rate of `signal` (defaults to nominal).
        """
        try:
            if fs is None:
                fs = self.fs

            # Welch PSD with a long FFT for fine frequency (heart-rate) resolution
            window_size = min(512, len(signal))
            freqs, power = welch(signal, fs=fs, nperseg=window_size, nfft=2**12)

            # Restrict to the physiological heart-rate band
            band = (freqs >= self.min_hr_hz) & (freqs <= self.max_hr_hz)
            freqs, power = freqs[band], power[band]
            if len(power) == 0:
                return None, 0.0

            # Most prominent peak in the band
            min_prominence = self.peak_prominence_factor * np.max(power)
            peaks, props = find_peaks(power, distance=5, prominence=min_prominence)
            if len(peaks) == 0:
                return None, 0.0

            prominences = props["prominences"]
            best = int(peaks[np.argmax(prominences)])
            hr_bpm = freqs[best] * 60.0

            if not (self.physiological_range[0] <= hr_bpm <= self.physiological_range[1]):
                return None, 0.0

            confidence = self._confidence(power, best, float(np.max(prominences)))
            return float(hr_bpm), confidence

        except Exception as e:
            print(f"HR estimation error: {e}")
            return None, 0.0

    def _confidence(self, power, peak_idx, peak_prominence):
        """Confidence from how cleanly the dominant peak stands out of the band.

        Combines the peak's prominence relative to the mean band power with its
        height relative to the band's median (background) power. Both are high
        only when a single sharp peak dominates the heart-rate band; a flat or
        multi-peak spectrum scores low.
        """
        prominence_ratio = peak_prominence / (np.mean(power) + 1e-6)
        background_snr = power[peak_idx] / (np.median(power) + 1e-6)

        prominence_score = min(1.0, prominence_ratio / 5.0)
        snr_score = min(1.0, background_snr / 20.0)
        return float(min(1.0, 0.5 * prominence_score + 0.5 * snr_score))
