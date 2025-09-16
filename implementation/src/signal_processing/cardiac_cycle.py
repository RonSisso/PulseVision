#!/usr/bin/env python3
"""
Cardiac Cycle Detection for rPPG Signals

This module implements systole and diastole detection from rPPG signals.
The rPPG signal reflects blood volume changes:
- Systole: Heart contraction → increased blood volume → decreased light reflection → signal valley
- Diastole: Heart relaxation → decreased blood volume → increased light reflection → signal peak

Heart rate is calculated from the time intervals between consecutive systoles (R-R intervals).
"""

import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter, butter, filtfilt
from scipy.stats import zscore
from collections import deque
import time

class CardiacCycleDetector:
    def __init__(self, sampling_rate=30):
        """
        Initialize cardiac cycle detector.
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 30 Hz)
        """
        self.fs = sampling_rate
        
        # Cardiac cycle parameters
        self.min_hr_bpm = 40
        self.max_hr_bpm = 180
        self.min_cycle_time = 60.0 / self.max_hr_bpm  # Minimum time between systoles
        self.max_cycle_time = 60.0 / self.min_hr_bpm  # Maximum time between systoles
        
        # Peak/valley detection parameters - more sensitive for better detection
        self.min_prominence = 0.05  # Reduced from 0.1 for more sensitive detection
        self.min_distance = int(self.min_cycle_time * self.fs * 0.8)  # Reduced distance for better detection
        
        # Signal processing parameters
        self.smoothing_window = 7  # Savitzky-Golay filter window
        self.smoothing_order = 2   # Savitzky-Golay filter order
        
        # Tracking variables
        self.systole_times = deque(maxlen=20)  # Track systole timestamps
        self.diastole_times = deque(maxlen=20)  # Track diastole timestamps
        self.rr_intervals = deque(maxlen=10)   # Track R-R intervals
        self.last_systole_time = None
        self.last_diastole_time = None
        
        # Heart rate calculation
        self.hr_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)
        
        # Signal quality tracking
        self.signal_quality_history = deque(maxlen=5)
        
    def detect_cardiac_cycles(self, signal, timestamp=None):
        """
        Detect systole and diastole from rPPG signal.
        
        Args:
            signal: rPPG signal array
            timestamp: Current timestamp (optional)
            
        Returns:
            dict: {
                'systole_times': list of systole timestamps,
                'diastole_times': list of diastole timestamps,
                'rr_intervals': list of R-R intervals in seconds,
                'heart_rate_bpm': calculated heart rate,
                'confidence': confidence score (0-1),
                'signal_quality': signal quality score (0-1)
            }
        """
        try:
            if timestamp is None:
                timestamp = time.time()
            
            # 1. Preprocess signal for cardiac cycle detection
            processed_signal = self._preprocess_for_cycle_detection(signal)
            
            # 2. Detect systoles (valleys in rPPG signal)
            systole_indices = self._detect_systoles(processed_signal)
            
            # 3. Detect diastoles (peaks in rPPG signal)
            diastole_indices = self._detect_diastoles(processed_signal)
            
            # 4. Calculate R-R intervals from systole times
            rr_intervals = self._calculate_rr_intervals(systole_indices, timestamp)
            
            # 5. Calculate heart rate from R-R intervals
            heart_rate_bpm, hr_confidence = self._calculate_heart_rate_from_rr(rr_intervals)
            
            # 6. Assess signal quality
            signal_quality = self._assess_signal_quality(processed_signal, systole_indices, diastole_indices)
            
            # 7. Update tracking variables
            self._update_tracking_variables(systole_indices, diastole_indices, timestamp)
            
            return {
                'systole_times': [timestamp - (len(signal) - idx) / self.fs for idx in systole_indices],
                'diastole_times': [timestamp - (len(signal) - idx) / self.fs for idx in diastole_indices],
                'rr_intervals': rr_intervals,
                'heart_rate_bpm': heart_rate_bpm,
                'confidence': hr_confidence,
                'signal_quality': signal_quality,
                'systole_indices': systole_indices,
                'diastole_indices': diastole_indices
            }
            
        except Exception as e:
            print(f"Cardiac cycle detection error: {e}")
            return {
                'systole_times': [],
                'diastole_times': [],
                'rr_intervals': [],
                'heart_rate_bpm': None,
                'confidence': 0.0,
                'signal_quality': 0.0,
                'systole_indices': [],
                'diastole_indices': []
            }
    
    def _preprocess_for_cycle_detection(self, signal):
        """Preprocess signal specifically for cardiac cycle detection."""
        try:
            # 1. Apply Savitzky-Golay filter for smoothing while preserving peaks
            if len(signal) >= self.smoothing_window:
                smoothed = savgol_filter(signal, self.smoothing_window, self.smoothing_order)
            else:
                smoothed = signal
            
            # 2. Apply bandpass filter to focus on cardiac frequencies
            nyquist = self.fs / 2
            low = 0.5 / nyquist  # 30 BPM
            high = 3.0 / nyquist  # 180 BPM
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, smoothed)
            
            # 3. Remove baseline drift
            if len(filtered) > 30:
                baseline = np.convolve(filtered, np.ones(30)/30, mode='same')
                processed = filtered - baseline
            else:
                processed = filtered
            
            return processed
            
        except Exception as e:
            print(f"Signal preprocessing error: {e}")
            return signal
    
    def _detect_systoles(self, signal):
        """Detect systoles (valleys) in the rPPG signal."""
        try:
            # Invert signal to find valleys as peaks
            inverted_signal = -signal
            
            # Find peaks in inverted signal (valleys in original)
            peaks, properties = find_peaks(
                inverted_signal,
                distance=self.min_distance,
                prominence=self.min_prominence * np.std(signal),
                height=np.percentile(inverted_signal, 20)  # Only consider significant valleys
            )
            
            # Filter peaks based on physiological constraints
            valid_systoles = []
            for peak in peaks:
                if self._is_valid_systole(peak, signal):
                    valid_systoles.append(peak)
            
            return valid_systoles
            
        except Exception as e:
            print(f"Systole detection error: {e}")
            return []
    
    def _detect_diastoles(self, signal):
        """Detect diastoles (peaks) in the rPPG signal."""
        try:
            # Find peaks in original signal
            peaks, properties = find_peaks(
                signal,
                distance=self.min_distance,
                prominence=self.min_prominence * np.std(signal),
                height=np.percentile(signal, 80)  # Only consider significant peaks
            )
            
            # Filter peaks based on physiological constraints
            valid_diastoles = []
            for peak in peaks:
                if self._is_valid_diastole(peak, signal):
                    valid_diastoles.append(peak)
            
            return valid_diastoles
            
        except Exception as e:
            print(f"Diastole detection error: {e}")
            return []
    
    def _is_valid_systole(self, peak_idx, signal):
        """Check if a detected peak represents a valid systole."""
        try:
            # Check if peak is deep enough (significant valley) - more lenient
            peak_value = signal[peak_idx]
            local_mean = np.mean(signal[max(0, peak_idx-10):min(len(signal), peak_idx+10)])
            
            if peak_value > local_mean * 0.9:  # More lenient threshold (was 0.8)
                return False
            
            # Check temporal constraints
            if len(self.systole_times) > 0:
                last_systole_time = self.systole_times[-1]
                current_time = peak_idx / self.fs
                time_diff = current_time - last_systole_time
                
                if time_diff < self.min_cycle_time or time_diff > self.max_cycle_time:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _is_valid_diastole(self, peak_idx, signal):
        """Check if a detected peak represents a valid diastole."""
        try:
            # Check if peak is high enough (significant peak) - more lenient
            peak_value = signal[peak_idx]
            local_mean = np.mean(signal[max(0, peak_idx-10):min(len(signal), peak_idx+10)])
            
            if peak_value < local_mean * 1.1:  # More lenient threshold (was 1.2)
                return False
            
            # Check temporal constraints
            if len(self.diastole_times) > 0:
                last_diastole_time = self.diastole_times[-1]
                current_time = peak_idx / self.fs
                time_diff = current_time - last_diastole_time
                
                if time_diff < self.min_cycle_time or time_diff > self.max_cycle_time:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_rr_intervals(self, systole_indices, timestamp):
        """Calculate R-R intervals from systole detection."""
        try:
            rr_intervals = []
            
            if len(systole_indices) < 2:
                return rr_intervals
            
            # Calculate intervals between consecutive systoles
            for i in range(1, len(systole_indices)):
                interval_samples = systole_indices[i] - systole_indices[i-1]
                interval_seconds = interval_samples / self.fs
                
                # Validate interval
                if self.min_cycle_time <= interval_seconds <= self.max_cycle_time:
                    rr_intervals.append(interval_seconds)
            
            return rr_intervals
            
        except Exception as e:
            print(f"R-R interval calculation error: {e}")
            return []
    
    def _calculate_heart_rate_from_rr(self, rr_intervals):
        """Calculate heart rate from R-R intervals."""
        try:
            if len(rr_intervals) == 0:
                return None, 0.0
            
            # Calculate heart rate from average R-R interval
            avg_rr = np.mean(rr_intervals)
            heart_rate_bpm = 60.0 / avg_rr
            
            # Calculate confidence based on R-R interval consistency
            if len(rr_intervals) > 1:
                rr_std = np.std(rr_intervals)
                rr_cv = rr_std / avg_rr  # Coefficient of variation
                confidence = max(0.0, 1.0 - rr_cv * 2)  # Lower CV = higher confidence
            else:
                confidence = 0.5  # Single interval, moderate confidence
            
            # Apply physiological constraints
            if not (self.min_hr_bpm <= heart_rate_bpm <= self.max_hr_bpm):
                return None, 0.0
            
            return heart_rate_bpm, confidence
            
        except Exception as e:
            print(f"Heart rate calculation error: {e}")
            return None, 0.0
    
    def _assess_signal_quality(self, signal, systole_indices, diastole_indices):
        """Assess the quality of cardiac cycle detection."""
        try:
            quality_score = 0.0
            
            # 1. Systole detection quality
            if len(systole_indices) > 0:
                systole_quality = min(1.0, len(systole_indices) / 5.0)  # Expect ~5 systoles in 10s
                quality_score += 0.4 * systole_quality
            
            # 2. Diastole detection quality
            if len(diastole_indices) > 0:
                diastole_quality = min(1.0, len(diastole_indices) / 5.0)
                quality_score += 0.3 * diastole_quality
            
            # 3. Signal-to-noise ratio
            signal_power = np.var(signal)
            noise_estimate = np.var(np.diff(signal))
            snr = signal_power / (noise_estimate + 1e-6)
            snr_quality = min(1.0, snr / 10.0)  # Normalize SNR
            quality_score += 0.3 * snr_quality
            
            return min(quality_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _update_tracking_variables(self, systole_indices, diastole_indices, timestamp):
        """Update internal tracking variables."""
        try:
            # Update systole times
            for idx in systole_indices:
                systole_time = timestamp - (len(systole_indices) - idx) / self.fs
                self.systole_times.append(systole_time)
            
            # Update diastole times
            for idx in diastole_indices:
                diastole_time = timestamp - (len(diastole_indices) - idx) / self.fs
                self.diastole_times.append(diastole_time)
            
            # Update R-R intervals
            if len(systole_indices) >= 2:
                for i in range(1, len(systole_indices)):
                    interval = (systole_indices[i] - systole_indices[i-1]) / self.fs
                    if self.min_cycle_time <= interval <= self.max_cycle_time:
                        self.rr_intervals.append(interval)
            
        except Exception as e:
            print(f"Tracking update error: {e}")
    
    def get_current_heart_rate(self):
        """Get the most recent heart rate estimate."""
        if len(self.hr_history) > 0:
            return self.hr_history[-1], self.confidence_history[-1]
        return None, 0.0
    
    def reset(self):
        """Reset all tracking variables."""
        self.systole_times.clear()
        self.diastole_times.clear()
        self.rr_intervals.clear()
        self.hr_history.clear()
        self.confidence_history.clear()
        self.signal_quality_history.clear()
        self.last_systole_time = None
        self.last_diastole_time = None
