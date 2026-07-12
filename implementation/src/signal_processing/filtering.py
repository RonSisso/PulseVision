import numpy as np
import cv2
from collections import deque

class HeartRateFilter:
    """Single-stage heart-rate post-filter: median outlier rejection + EMA.

    A median over the last few readings rejects transient single-frame outliers
    (the median ignores a lone spike) while a sustained change shifts the median
    within a couple of frames. A confidence-weighted exponential moving average
    then smooths residual jitter: confident readings move the output faster, so
    a real heart-rate change is tracked promptly without the output chattering
    on noisy frames. This replaces the previous stack of baseline anchoring,
    z-score rejection, and change caps with one explainable step.
    """

    def __init__(self, window_size=5, min_alpha=0.15, max_alpha=0.6):
        self.recent = deque(maxlen=window_size)
        self.smoothed = None
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.physiological_range = (40, 180)

    def update(self, new_bpm, confidence=0.5):
        """Fold one estimate into the running heart rate; returns the smoothed BPM."""
        if new_bpm is None:
            return self.smoothed

        lo, hi = self.physiological_range
        if not (lo <= new_bpm <= hi):
            print(f"Physiological range violation: {new_bpm:.1f} BPM outside {self.physiological_range}")
            return self.smoothed

        self.recent.append(new_bpm)
        robust = float(np.median(self.recent))  # rejects lone outliers

        if self.smoothed is None:
            self.smoothed = robust
        else:
            c = min(1.0, max(0.0, confidence))
            alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * c
            self.smoothed = (1 - alpha) * self.smoothed + alpha * robust

        return self.smoothed

    def reset(self):
        """Clear filter state for a fresh measurement."""
        self.recent.clear()
        self.smoothed = None

class ROIStabilityChecker:
    def __init__(self, min_std=3.0):  # Reduced from 5.0 to 3.0 for less sensitivity
        self.min_std = min_std

    def is_stable(self, roi_patch):
        gray = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        
        # Adaptive threshold based on ROI size
        # Smaller ROIs get more lenient thresholds
        roi_area = roi_patch.shape[0] * roi_patch.shape[1]
        if roi_area < 1000:  # Small ROI (like cheeks)
            adaptive_threshold = self.min_std * 0.7  # 30% more lenient
        elif roi_area < 2000:  # Medium ROI
            adaptive_threshold = self.min_std * 0.85  # 15% more lenient
        else:  # Large ROI (like forehead)
            adaptive_threshold = self.min_std
        
        return std_dev >= adaptive_threshold