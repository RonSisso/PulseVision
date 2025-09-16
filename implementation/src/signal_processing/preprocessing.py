import numpy as np
from scipy.signal import savgol_filter, iirnotch, filtfilt, butter, detrend
from scipy.stats import zscore

class SignalPreprocessor:
    def __init__(self, sampling_rate):
        self.fs = sampling_rate
        self.prev_savgol_len = 15 if sampling_rate >= 30 else 9  # ~0.5 sec smoothing
        self.prev_savgol_poly = 2
        
        # Enhanced notch filtering for common noise sources
        self.dynamic_notch_freqs = [1.0, 2.0, 3.0, 4.0]  # Extended frequency range
        self.q_factor = 25.0  # Slightly reduced Q for less aggressive filtering
        
        # Heart rate band parameters
        self.hr_min_hz = 40 / 60.0  # 40 BPM
        self.hr_max_hz = 180 / 60.0  # 180 BPM
        
        # Adaptive filtering parameters
        self.adaptive_smoothing = True
        self.signal_quality_threshold = 0.6
        
        # Initialize bandpass filter for heart rate enhancement
        self._setup_bandpass_filter()

    def _setup_bandpass_filter(self):
        """Setup bandpass filter for heart rate frequency range."""
        try:
            nyquist = 0.5 * self.fs
            low = self.hr_min_hz / nyquist
            high = self.hr_max_hz / nyquist
            
            # Use Butterworth filter with moderate order for good phase response
            self.b_bandpass, self.a_bandpass = butter(4, [low, high], btype='band')
        except Exception as e:
            print(f"Bandpass filter setup error: {e}")
            self.b_bandpass, self.a_bandpass = None, None

    def smooth_temporal(self, signal):
        """Enhanced temporal smoothing with adaptive parameters."""
        if len(signal) < self.prev_savgol_len:
            return signal  # Not enough samples
        
        # Assess signal quality to determine smoothing aggressiveness
        signal_quality = self._assess_signal_quality(signal)
        
        if self.adaptive_smoothing and signal_quality < self.signal_quality_threshold:
            # More aggressive smoothing for poor quality signals
            window_length = min(self.prev_savgol_len + 4, len(signal) // 2)
            poly_order = min(self.prev_savgol_poly + 1, window_length // 2)
        else:
            # Standard smoothing for good quality signals
            window_length = self.prev_savgol_len
            poly_order = self.prev_savgol_poly
        
        # Ensure window length is odd and polynomial order is valid
        if window_length % 2 == 0:
            window_length -= 1
        if window_length < poly_order + 1:
            window_length = poly_order + 1
        if window_length > len(signal):
            window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
        
        try:
            return savgol_filter(signal, window_length, poly_order)
        except Exception:
            # Fallback to simple moving average if Savitzky-Golay fails
            return self._simple_moving_average(signal, window_length)

    def _simple_moving_average(self, signal, window_length):
        """Fallback simple moving average filter."""
        if window_length <= 1:
            return signal
        
        # Use convolution for moving average
        kernel = np.ones(window_length) / window_length
        padded_signal = np.pad(signal, (window_length//2, window_length//2), mode='edge')
        smoothed = np.convolve(padded_signal, kernel, mode='valid')
        
        return smoothed

    def apply_adaptive_notch(self, signal):
        """Enhanced adaptive notch filtering with signal quality assessment."""
        filtered = signal.copy()
        nyquist = 0.5 * self.fs
        
        # Assess signal quality to determine filtering aggressiveness
        signal_quality = self._assess_signal_quality(signal)
        
        # Adjust Q factor based on signal quality
        if signal_quality > 0.8:
            q_factor = self.q_factor * 1.2  # Less aggressive for high quality signals
        elif signal_quality < 0.4:
            q_factor = self.q_factor * 0.8  # More aggressive for poor quality signals
        else:
            q_factor = self.q_factor
        
        # Apply notch filters with error handling
        for f0 in self.dynamic_notch_freqs:
            try:
                # Check if frequency is within valid range
                if f0 >= nyquist * 0.9:  # Too close to Nyquist frequency
                    continue
                
                b, a = iirnotch(f0 / nyquist, q_factor)
                
                # Check filter stability
                if np.all(np.abs(np.roots(a)) < 1.0):
                    filtered = filtfilt(b, a, filtered)
                else:
                    print(f"Unstable notch filter at {f0} Hz, skipping")
                    
            except Exception as e:
                print(f"Notch filter error at {f0} Hz: {e}")
                continue  # skip if unstable
        
        return filtered

    def normalize_robust(self, values):
        """Enhanced robust normalization with outlier protection."""
        values = np.array(values, dtype=float)
        
        if len(values) == 0:
            return values
        
        # Remove DC component first
        values = detrend(values)
        
        # Use median absolute deviation (MAD) for robust normalization
        median = np.median(values)
        mad = np.median(np.abs(values - median)) + 1e-6
        
        # Apply MAD normalization
        normalized = (values - median) / mad
        
        # Clip extreme outliers to prevent numerical issues
        outlier_threshold = 5.0  # 5 MAD units
        normalized = np.clip(normalized, -outlier_threshold, outlier_threshold)
        
        return normalized

    def enhance_heart_rate_signal(self, signal):
        """Enhanced heart rate signal enhancement pipeline focused on cleaning."""
        try:
            # 1. Remove DC component and linear trend
            enhanced = detrend(signal)
            
            # 2. Apply aggressive bandpass filter to focus on heart rate frequencies
            if self.b_bandpass is not None and self.a_bandpass is not None:
                enhanced = filtfilt(self.b_bandpass, self.a_bandpass, enhanced)
            
            # 3. Apply adaptive notch filtering to remove common noise
            enhanced = self.apply_adaptive_notch(enhanced)
            
            # 4. Apply robust normalization
            enhanced = self.normalize_robust(enhanced)
            
            # 5. Apply motion artifact removal
            enhanced = self._remove_motion_artifacts(enhanced)
            
            # 6. Apply adaptive temporal smoothing
            enhanced = self.smooth_temporal(enhanced)
            
            # 7. Final bandpass filter to ensure only heart rate frequencies remain
            if self.b_bandpass is not None and self.a_bandpass is not None:
                enhanced = filtfilt(self.b_bandpass, self.a_bandpass, enhanced)
            
            return enhanced
            
        except Exception as e:
            print(f"Signal enhancement error: {e}")
            return signal

    def _remove_motion_artifacts(self, signal):
        """Remove motion artifacts using statistical methods with enhanced detection."""
        try:
            # Calculate moving statistics with smaller window for better responsiveness
            window_size = min(15, len(signal) // 8)  # 0.5 second window
            if window_size < 3:
                return signal
            
            # Calculate moving median and MAD
            moving_median = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
            
            # Calculate moving MAD
            abs_diff = np.abs(signal - moving_median)
            moving_mad = np.convolve(abs_diff, np.ones(window_size)/window_size, mode='same')
            
            # More aggressive outlier detection
            threshold = 2.5 * moving_mad  # Reduced from 3.0 to 2.5 for more aggressive cleaning
            outlier_mask = abs_diff > threshold
            
            # Also detect sudden changes (potential movement artifacts)
            if len(signal) > 1:
                signal_diff = np.abs(np.diff(signal))
                signal_diff = np.concatenate([[0], signal_diff])  # Pad to match signal length
                
                # Detect sudden changes that are much larger than typical variations
                change_threshold = 3.0 * np.std(signal_diff)
                sudden_change_mask = signal_diff > change_threshold
                
                # Combine both outlier detection methods
                outlier_mask = outlier_mask | sudden_change_mask
            
            # Replace outliers with interpolated values
            if np.any(outlier_mask):
                # Simple linear interpolation for outliers
                clean_signal = signal.copy()
                outlier_indices = np.where(outlier_mask)[0]
                
                for idx in outlier_indices:
                    # Find nearest non-outlier values
                    left_idx = idx - 1
                    while left_idx >= 0 and outlier_mask[left_idx]:
                        left_idx -= 1
                    
                    right_idx = idx + 1
                    while right_idx < len(signal) and outlier_mask[right_idx]:
                        right_idx += 1
                    
                    # Interpolate if we have valid neighbors
                    if left_idx >= 0 and right_idx < len(signal):
                        # Linear interpolation
                        alpha = (idx - left_idx) / (right_idx - left_idx)
                        clean_signal[idx] = (1 - alpha) * signal[left_idx] + alpha * signal[right_idx]
                    elif left_idx >= 0:
                        clean_signal[idx] = signal[left_idx]
                    elif right_idx < len(signal):
                        clean_signal[idx] = signal[right_idx]
                    else:
                        clean_signal[idx] = moving_median[idx]  # Fallback to median
                
                return clean_signal
            
            return signal
            
        except Exception as e:
            print(f"Motion artifact removal error: {e}")
            return signal

    def _assess_signal_quality(self, signal):
        """Assess signal quality for adaptive processing."""
        try:
            if len(signal) < 10:
                return 0.0
            
            # 1. Signal-to-noise ratio estimation
            signal_power = np.var(signal)
            noise_estimate = np.var(np.diff(signal))  # High-frequency content as noise estimate
            snr = signal_power / (noise_estimate + 1e-6)
            snr_quality = min(1.0, snr / 10.0)  # Normalize SNR
            
            # 2. Temporal consistency
            if len(signal) > 20:
                # Check for sudden amplitude changes
                diff_signal = np.abs(np.diff(signal))
                temporal_consistency = 1.0 / (1.0 + np.std(diff_signal) / (np.mean(diff_signal) + 1e-6))
            else:
                temporal_consistency = 0.5
            
            # 3. Frequency domain quality
            # Check for dominant frequency content in heart rate band
            from scipy.signal import welch
            freqs, power = welch(signal, fs=self.fs, nperseg=min(256, len(signal)))
            
            hr_mask = (freqs >= self.hr_min_hz) & (freqs <= self.hr_max_hz)
            if np.any(hr_mask):
                hr_power = np.sum(power[hr_mask])
                total_power = np.sum(power)
                frequency_quality = hr_power / (total_power + 1e-6)
            else:
                frequency_quality = 0.0
            
            # Combined quality score
            quality = (0.4 * snr_quality + 
                      0.3 * temporal_consistency + 
                      0.3 * frequency_quality)
            
            return min(quality, 1.0)
            
        except Exception:
            return 0.0