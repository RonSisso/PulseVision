"""Signal preprocessing for rPPG: detrend, bandpass, robust normalization.

Design notes:
- The filter passband (0.5-4.0 Hz) is deliberately wider than the heart-rate
  search band used by the estimator (0.67-3.0 Hz). With zero-phase filtering
  (filtfilt) the response is essentially flat across all reportable heart
  rates; the filter only removes baseline drift / respiration below the band
  and sensor noise above it. Keeping the edges outside the search band avoids
  attenuating real pulses at 40-50 or 150-180 BPM.
- No notch filters: at a 30 Hz sampling rate, mains-driven light flicker
  (100/120 Hz) aliases to ~10 Hz or DC — nothing in the heart-rate band.
  Notches at 1-3 Hz would sit exactly on real heart rates (60-180 BPM) and
  carve them out.
- No additional smoothing: Welch's windowed PSD in the estimator handles
  residual noise; extra Savitzky-Golay passes act as a ~2 Hz low-pass that
  suppresses high heart rates.
"""

import numpy as np
from scipy.signal import butter, filtfilt, detrend


class SignalPreprocessor:
    BAND_LOW_HZ = 0.5     # below the 40 BPM (0.67 Hz) search edge
    BAND_HIGH_HZ = 4.0    # above the 180 BPM (3.0 Hz) search edge
    FILTER_ORDER = 3
    OUTLIER_CLIP_MAD = 5.0  # clip normalized samples beyond +/- 5 MAD units

    def __init__(self, sampling_rate):
        self.fs = sampling_rate
        self.b_bandpass, self.a_bandpass = self._design_bandpass()

    def _design_bandpass(self):
        """Design the Butterworth bandpass; returns (None, None) on failure."""
        try:
            nyquist = 0.5 * self.fs
            low = self.BAND_LOW_HZ / nyquist
            high = min(self.BAND_HIGH_HZ / nyquist, 0.99)
            return butter(self.FILTER_ORDER, [low, high], btype='band')
        except Exception as e:
            print(f"Bandpass filter setup error: {e}")
            return None, None

    def enhance_heart_rate_signal(self, signal):
        """Clean the raw green-channel trace: detrend -> bandpass -> normalize."""
        try:
            enhanced = detrend(signal)
            if self.b_bandpass is not None:
                enhanced = filtfilt(self.b_bandpass, self.a_bandpass, enhanced)
            return self.normalize_robust(enhanced)
        except Exception as e:
            print(f"Signal enhancement error: {e}")
            return signal

    def normalize_robust(self, values):
        """Median/MAD normalization with outlier clipping.

        MAD (median absolute deviation) is robust to motion spikes, so a few
        corrupted samples do not rescale the whole trace; clipping then limits
        how much any single spike can dominate the spectrum.
        """
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return values

        median = np.median(values)
        mad = np.median(np.abs(values - median)) + 1e-6
        normalized = (values - median) / mad

        return np.clip(normalized, -self.OUTLIER_CLIP_MAD, self.OUTLIER_CLIP_MAD)
