import numpy as np
import cv2
from collections import deque
from scipy.stats import zscore
from scipy.signal import butter, filtfilt

class HeartRateFilter:
    def __init__(self, window_size=5):  # Smaller window for more responsive filtering
        self.history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        
        # Simple filtering parameters
        self.base_z_threshold = 2.5  # More lenient threshold
        
        # Physiological constraints
        self.physiological_range = (40, 180)
        self.max_physiological_change = 25  # More lenient for cleaner signals

    def update(self, new_bpm, confidence=0.5):
        """Simple heart rate filtering focused on outlier rejection."""
        if new_bpm is None:
            return None
        
        # Check physiological range first
        if not (self.physiological_range[0] <= new_bpm <= self.physiological_range[1]):
            print(f"Physiological range violation: {new_bpm:.1f} BPM outside {self.physiological_range}")
            return self._get_fallback_bpm()
        
        # Add to history
        self.history.append(new_bpm)
        self.confidence_history.append(confidence)
        
        if len(self.history) < 3:
            return new_bpm
        
        # Simple outlier detection
        filtered_bpm = self._simple_outlier_detection(new_bpm, confidence)
        
        return filtered_bpm

    def _simple_outlier_detection(self, new_bpm, confidence):
        """Simple outlier detection based on z-score."""
        history_array = np.array(list(self.history))
        
        # Calculate basic statistics
        median_bpm = np.median(history_array)
        std_dev = np.std(history_array)
        
        # Handle edge cases
        if std_dev == 0 or np.isnan(std_dev):
            return median_bpm
        
        # Calculate z-score
        mean_val = np.mean(history_array)
        z_score = (new_bpm - mean_val) / std_dev
        
        # Check for NaN or infinite values
        if np.isnan(z_score) or np.isinf(z_score):
            return median_bpm
        
        # Apply simple outlier rejection
        if abs(z_score) > self.base_z_threshold:
            print(f"Outlier detected: BPM={new_bpm:.1f}, z-score={z_score:.2f}, using median={median_bpm:.1f}")
            return median_bpm
        
        # Check for physiologically unreasonable changes
        if len(self.history) > 1:
            last_bpm = self.history[-2]
            bpm_change = abs(new_bpm - last_bpm)
            if bpm_change > self.max_physiological_change and confidence < 0.8:
                print(f"Physiological change constraint: {bpm_change:.1f} BPM change > "
                      f"{self.max_physiological_change} BPM with low confidence")
                return last_bpm
        
        return new_bpm

    def _get_fallback_bpm(self):
        """Get a fallback BPM value when current measurement is invalid."""
        if len(self.history) > 0:
            return np.median(list(self.history))
        else:
            return 70.0  # Default resting heart rate

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