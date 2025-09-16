import numpy as np
import time
from scipy.signal import welch, find_peaks, peak_prominences, butter, filtfilt
from scipy.stats import entropy
from collections import deque

class HeartRateEstimator:
    def __init__(self, sampling_rate):
        self.fs = sampling_rate
        self.min_hr_hz = 40 / 60.0
        self.max_hr_hz = 180 / 60.0
        
        # Simple tracking without complex locking
        self.last_hr_freq = None
        self.hr_history = deque(maxlen=10)  # Track recent heart rate estimates
        self.confidence_history = deque(maxlen=10)  # Track recent confidence values
        
        # Peak selection parameters
        self.min_confidence_threshold = 0.4  # Lower threshold for more responsive detection
        self.peak_prominence_factor = 0.15  # Minimum prominence relative to max power
        
        # Physiological constraints
        self.physiological_range = (40, 180)  # BPM range
        self.max_physiological_change = 20  # Maximum reasonable change per update
        
        # Signal quality tracking
        self.signal_quality_history = deque(maxlen=5)


    def _assess_signal_quality_simple(self, signal, freqs, power):
        """Simple signal quality assessment focused on heart rate content."""
        try:
            # 1. Signal-to-noise ratio in heart rate band
            hr_power = np.sum(power)
            total_power = np.sum(power)  # Already filtered to HR band
            snr = hr_power / (total_power + 1e-6)
            
            # 2. Peak clarity - how distinct the main peak is
            peaks, _ = find_peaks(power, distance=5, prominence=0.1*np.max(power))
            if len(peaks) > 0:
                peak_heights = power[peaks]
                max_peak = np.max(peak_heights)
                avg_peak = np.mean(peak_heights)
                peak_clarity = max_peak / (avg_peak + 1e-6)
            else:
                peak_clarity = 0.0
            
            # 3. Temporal consistency (if we have history)
            temporal_consistency = 0.5  # Default
            if len(self.signal_quality_history) > 2:
                recent_qualities = list(self.signal_quality_history)[-3:]
                quality_std = np.std(recent_qualities)
                temporal_consistency = 1.0 / (1.0 + quality_std * 5)
            
            # Combined quality score
            quality = (0.4 * min(snr / 2.0, 1.0) +  # SNR component
                      0.4 * min(peak_clarity / 3.0, 1.0) +  # Peak clarity
                      0.2 * temporal_consistency)  # Temporal consistency
            
            return min(quality, 1.0)
            
        except Exception:
            return 0.0


    def estimate(self, signal):
        """Simple and robust heart rate estimation based on clean signal analysis."""
        try:
            # 1. Frequency domain analysis with longer window for better resolution
            window_size = min(512, len(signal))
            freqs, power = welch(signal, fs=self.fs, nperseg=window_size, nfft=2**12)
            
            # 2. Focus on heart rate band
            band_mask = (freqs >= self.min_hr_hz) & (freqs <= self.max_hr_hz)
            freqs = freqs[band_mask]
            power = power[band_mask]
            
            if len(power) == 0:
                return None, 0.0
            
            # 3. Find the most prominent peak
            min_prominence = self.peak_prominence_factor * np.max(power)
            peaks, properties = find_peaks(power, distance=5, prominence=min_prominence)
            
            if len(peaks) == 0:
                return None, 0.0
            
            # 4. Select the peak with highest prominence
            prominences = properties['prominences']
            best_peak_idx = peaks[np.argmax(prominences)]
            hr_freq = freqs[best_peak_idx]
            hr_bpm = hr_freq * 60.0
            
            # 5. Calculate confidence based on peak quality
            max_prominence = np.max(prominences)
            avg_power = np.mean(power)
            peak_quality = max_prominence / (avg_power + 1e-6)
            
            # 6. Assess signal quality
            signal_quality = self._assess_signal_quality_simple(signal, freqs, power)
            
            # 7. Calculate final confidence
            confidence = min(1.0, 0.6 * peak_quality + 0.4 * signal_quality)
            
            # 8. Apply physiological constraints
            if not (self.physiological_range[0] <= hr_bpm <= self.physiological_range[1]):
                print(f"Physiological constraint: {hr_bpm:.1f} BPM outside range")
                return None, 0.0
            
            # 9. Check for reasonable changes from last measurement
            if self.last_hr_freq is not None:
                last_bpm = self.last_hr_freq * 60.0
                bpm_change = abs(hr_bpm - last_bpm)
                
                # More aggressive change detection to prevent dips
                if bpm_change > self.max_physiological_change and confidence < 0.8:
                    print(f"Large change constraint: {bpm_change:.1f} BPM change with confidence {confidence:.2f}")
                    return last_bpm, 0.3  # Return previous value with low confidence
                elif bpm_change > 15 and confidence < 0.6:  # Medium changes need good confidence
                    print(f"Medium change constraint: {bpm_change:.1f} BPM change with confidence {confidence:.2f}")
                    return last_bpm, 0.4
            
            # 10. Temporal consistency check using history
            if len(self.hr_history) >= 3:
                recent_hrs = list(self.hr_history)[-3:]
                hr_std = np.std(recent_hrs)
                
                # If current measurement is very different from recent trend, reduce confidence
                if hr_std < 5.0:  # Recent measurements are consistent
                    recent_mean = np.mean(recent_hrs)
                    if abs(hr_bpm - recent_mean) > 10 and confidence < 0.7:
                        print(f"Temporal consistency: {hr_bpm:.1f} BPM differs from recent trend {recent_mean:.1f} BPM")
                        return recent_mean, 0.5  # Return recent trend with medium confidence
            
            # 10. Update tracking
            self.last_hr_freq = hr_freq
            self.hr_history.append(hr_bpm)
            self.confidence_history.append(confidence)
            self.signal_quality_history.append(signal_quality)
            
            print(f"HR Estimation: {hr_bpm:.1f} BPM, confidence: {confidence:.2f}, "
                  f"signal_quality: {signal_quality:.2f}, peak_quality: {peak_quality:.2f}")
            
            return hr_bpm, confidence
            
        except Exception as e:
            print(f"HR estimation error: {e}")
            return None, 0.0
