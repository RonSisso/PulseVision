"""Multi-ROI rPPG signal processing orchestrator.

Combines the green-channel traces from the forehead and cheek ROIs into a
single signal, cleans it, and estimates heart rate in the frequency domain.
"""

from collections import deque
import logging
import time

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, welch

from .preprocessing import SignalPreprocessor
from .hr_estimation import HeartRateEstimator
from .filtering import HeartRateFilter, ROIStabilityChecker


class SignalProcessor:
    def __init__(self, sampling_rate=30):
        self.logger = logging.getLogger(__name__)
        self.fs = sampling_rate

        # Multiple ROI buffers for different regions
        self.buffers = {
            'forehead': deque(maxlen=self.fs * 10),
            'left_cheek': deque(maxlen=self.fs * 10),
            'right_cheek': deque(maxlen=self.fs * 10)
        }

        # Combined signal buffer
        self.combined_buffer = deque(maxlen=self.fs * 10)

        self.preprocessor = SignalPreprocessor(sampling_rate)
        self.hr_estimator = HeartRateEstimator(sampling_rate)
        self.hr_filter = HeartRateFilter()
        self.roi_checker = ROIStabilityChecker()
        self.last_bpm = None

        # Initialization delay to avoid startup noise
        self.init_delay = 2.0  # 2 seconds delay
        self.start_time = None
        self.initialized = False

        # ROI tracking for automatic reset
        self.last_roi_stable = {
            'forehead': False,
            'left_cheek': False,
            'right_cheek': False
        }
        self.roi_lost = False

        # Dynamic signal combination weights (adjusted based on ROI stability)
        self.base_weights = {
            'forehead': 0.5,    # Forehead typically has good signal
            'left_cheek': 0.25,  # Cheeks may have different signal quality
            'right_cheek': 0.25
        }
        self.roi_weights = self.base_weights.copy()

        # ROI health tracking for dynamic weight adjustment
        self.roi_health = {
            'forehead': 1.0,    # Health score (0.0 = bad, 1.0 = perfect)
            'left_cheek': 1.0,
            'right_cheek': 1.0
        }
        self.roi_stability_history = {
            'forehead': deque(maxlen=10),    # Track last 10 stability checks
            'left_cheek': deque(maxlen=10),
            'right_cheek': deque(maxlen=10)
        }

        # Baseline heart rate system
        self.baseline_hr = None
        self.baseline_established = False
        self.baseline_time = 20.0  # Establish baseline after 20 seconds
        self.baseline_alpha = 0.1  # How much new measurements influence baseline
        self.smoothed_hr = None
        self.smoothing_alpha = 0.3  # How much new measurements influence smoothed HR

    def process_frame(self, frame, roi):
        """Legacy method for single ROI processing - maintained for backward compatibility."""
        if isinstance(roi, dict):
            # New multi-ROI format
            return self.process_multiple_rois(frame, roi)
        else:
            # Legacy single ROI format - convert to multi-ROI format
            rois = {'forehead': roi}
            return self.process_multiple_rois(frame, rois)

    def process_multiple_rois(self, frame, rois):
        """Process multiple ROIs and combine their signals for heart rate estimation."""
        # Check for complete failure (no frame or no ROIs at all)
        if frame is None:
            print("Frame is None - resetting signal processor")
            self.reset()
            self.roi_lost = True
            return None, 0.0

        if rois is None or not any(rois.values()):
            print("All ROIs are None - face left frame, resetting signal processor")
            self.reset()
            self.roi_lost = True
            return None, 0.0

        # Initialize start time if not set
        if self.start_time is None:
            self.start_time = time.time()
            print(f"Starting initialization delay: {self.init_delay} seconds")

        # Check if initialization delay is complete
        current_time = time.time()
        if not self.initialized and (current_time - self.start_time) < self.init_delay:
            remaining = self.init_delay - (current_time - self.start_time)
            print(f"Initialization delay: {remaining:.1f}s remaining")
            return None, 0.0
        elif not self.initialized:
            self.initialized = True
            print("Initialization complete - starting heart rate measurement")

        # Process each ROI with graceful degradation
        roi_signals = {}
        valid_rois = 0

        for roi_name, roi in rois.items():
            if roi is None:
                # ROI is missing - reduce its health
                self._update_roi_health(roi_name, False)
                continue

            x, y, w, h = roi
            roi_patch = frame[y:y + h, x:x + w]

            # Check if ROI is stable
            current_roi_stable = self.roi_checker.is_stable(roi_patch)
            self.last_roi_stable[roi_name] = current_roi_stable

            # Update ROI health based on stability
            self._update_roi_health(roi_name, current_roi_stable)

            # Always try to extract signal, but with reduced weight if unstable
            try:
                green = roi_patch[:, :, 1].astype(np.float32)
                mean_val = np.mean(green)
                self.buffers[roi_name].append(mean_val)
                roi_signals[roi_name] = mean_val
                valid_rois += 1
            except Exception as e:
                print(f"Error processing {roi_name} ROI: {e}")
                self._update_roi_health(roi_name, False)

        # Update dynamic weights based on ROI health
        weights_changed = self._update_dynamic_weights()

        if valid_rois == 0:
            print("No valid ROIs found - returning last valid measurement")
            return self.last_bpm, 0.0, None, None, {}

        # Combine signals from all valid ROIs
        combined_signal = self._combine_roi_signals(roi_signals)

        # Apply smoothing if weights changed to prevent signal jumps
        if weights_changed and len(self.combined_buffer) > 0:
            # Smooth the transition when weights change
            last_signal = self.combined_buffer[-1]
            combined_signal = 0.7 * last_signal + 0.3 * combined_signal

        self.combined_buffer.append(combined_signal)

        if len(self.combined_buffer) < self.fs * 2:
            return None, 0.0

        # Convert combined buffer to numpy array
        signal = np.array(self.combined_buffer)

        # Step 1: Enhanced signal preprocessing pipeline
        signal = self.preprocessor.enhance_heart_rate_signal(signal)

        # Step 2: Additional PPG signal enhancement
        signal = self.enhance_ppg_signal(signal)

        # Step 3: Heart rate estimation using FFT method
        freq_bpm, freq_confidence = self.hr_estimator.estimate(signal)

        if freq_bpm is not None and freq_confidence > 0.4:
            bpm, confidence = freq_bpm, freq_confidence
            method_used = "frequency_domain"
            print(f"Using FFT method: {bpm:.1f} BPM, confidence: {confidence:.2f}")
        else:
            # FFT method failed
            bpm, confidence = None, 0.0
            method_used = "none"
            print("FFT heart rate estimation failed")

        # Step 4: Enhanced confidence validation
        if bpm is not None and confidence is not None:
            # Additional confidence checks - using FFT method only
            min_confidence_threshold = 0.4  # Threshold for FFT method

            # Check for physiologically reasonable values
            if not (40 <= bpm <= 180):
                print(f"Physiologically unreasonable BPM: {bpm:.1f} - rejecting")
                bpm = None
                confidence = 0.0

            # Check confidence threshold
            elif confidence < min_confidence_threshold:
                print(f"Low confidence measurement: {confidence:.2f} < {min_confidence_threshold} - rejecting")
                bpm = None
                confidence = 0.0

            # Check for sudden changes from last valid measurement
            elif self.last_bpm is not None:
                bpm_change = abs(bpm - self.last_bpm)
                # Simple constraint - only reject if change is very large with low confidence
                if bpm_change > 20 and confidence < 0.6:
                    print(f"Large BPM change ({bpm_change:.1f}) with low confidence ({confidence:.2f}) - rejecting")
                    bpm = None
                    confidence = 0.0

        # Step 5: Baseline-based heart rate calculation
        if bpm is not None:
            # Check if we should establish baseline
            current_time = time.time()
            time_since_start = current_time - self.start_time if self.start_time else 0

            if not self.baseline_established and time_since_start >= self.baseline_time:
                # Establish baseline after 20 seconds
                self.baseline_hr = bpm
                self.baseline_established = True
                self.smoothed_hr = bpm
                print(f"Baseline HR established: {bpm:.1f} BPM")

            if self.baseline_established:
                # Use baseline-based calculation
                # Gradually update baseline with new measurements
                self.baseline_hr = (1 - self.baseline_alpha) * self.baseline_hr + self.baseline_alpha * bpm

                # Calculate final HR as weighted combination of baseline and new measurement
                # Higher confidence = more weight to new measurement
                baseline_weight = 0.6  # 60% baseline weight
                new_weight = 0.4 * confidence  # Up to 40% new measurement weight
                total_weight = baseline_weight + new_weight

                if total_weight > 0:
                    final_bpm = (baseline_weight * self.baseline_hr + new_weight * bpm) / total_weight
                else:
                    final_bpm = self.baseline_hr

                # Apply additional smoothing
                if self.smoothed_hr is None:
                    self.smoothed_hr = final_bpm
                else:
                    self.smoothed_hr = (1 - self.smoothing_alpha) * self.smoothed_hr + self.smoothing_alpha * final_bpm

                # Apply outlier rejection
                filtered_bpm = self.hr_filter.update(self.smoothed_hr, confidence)
                self.last_bpm = filtered_bpm

                print(f"HR: {bpm:.1f} -> Baseline: {self.baseline_hr:.1f} -> Smoothed: {self.smoothed_hr:.1f} -> Final: {filtered_bpm:.1f} BPM")
            else:
                # Before baseline establishment, use normal filtering
                filtered_bpm = self.hr_filter.update(bpm, confidence)
                self.last_bpm = filtered_bpm
        else:
            filtered_bpm = self.last_bpm  # Keep last valid measurement
            confidence = 0.0

        # Calculate FFT for display if we have enough signal data
        fft_freqs, fft_power = None, None
        if len(signal) >= 64:  # Need minimum samples for meaningful FFT
            try:
                window_size = min(512, len(signal))
                fft_freqs, fft_power = welch(signal, fs=self.fs, nperseg=window_size, nfft=2**10)

                # Focus on heart rate frequency range (0.5-3 Hz)
                hr_mask = (fft_freqs >= 0.5) & (fft_freqs <= 3.0)
                fft_freqs = fft_freqs[hr_mask]
                fft_power = fft_power[hr_mask]
            except Exception as e:
                print(f"FFT calculation error: {e}")
                fft_freqs, fft_power = None, None

        method_data = {'method_used': method_used}

        return filtered_bpm, confidence, fft_freqs, fft_power, method_data

    def _update_roi_health(self, roi_name, is_stable):
        """Update ROI health based on stability."""
        # Record stability in history
        self.roi_stability_history[roi_name].append(is_stable)

        # Calculate health based on recent stability (last 10 checks)
        if len(self.roi_stability_history[roi_name]) > 0:
            stability_ratio = sum(self.roi_stability_history[roi_name]) / len(self.roi_stability_history[roi_name])

            # Smooth health updates to avoid rapid changes
            alpha = 0.05  # Reduced learning rate for smoother weight transitions
            self.roi_health[roi_name] = (1 - alpha) * self.roi_health[roi_name] + alpha * stability_ratio

            # Ensure health stays in [0, 1] range
            self.roi_health[roi_name] = max(0.0, min(1.0, self.roi_health[roi_name]))

    def _update_dynamic_weights(self):
        """Update ROI weights based on their health scores."""
        # Store old weights to detect changes
        old_weights = self.roi_weights.copy()

        # Calculate total health-weighted base weights
        total_weighted_base = 0.0
        for roi_name in self.base_weights:
            total_weighted_base += self.base_weights[roi_name] * self.roi_health[roi_name]

        # Normalize weights so they sum to 1.0
        if total_weighted_base > 0:
            for roi_name in self.roi_weights:
                self.roi_weights[roi_name] = (self.base_weights[roi_name] * self.roi_health[roi_name]) / total_weighted_base
        else:
            # Fallback to equal weights if all ROIs are unhealthy
            for roi_name in self.roi_weights:
                self.roi_weights[roi_name] = 1.0 / len(self.roi_weights)

        # Check if weights changed significantly
        weights_changed = False
        for roi_name in self.roi_weights:
            if abs(self.roi_weights[roi_name] - old_weights[roi_name]) > 0.05:  # 5% change threshold
                weights_changed = True
                break

        # Print weight changes for debugging
        weight_changes = []
        for roi_name in self.roi_weights:
            health = self.roi_health[roi_name]
            weight = self.roi_weights[roi_name]
            weight_changes.append(f"{roi_name}: {weight:.2f} (health: {health:.2f})")
        print(f"ROI weights: {', '.join(weight_changes)}")

        return weights_changed

    def _combine_roi_signals(self, roi_signals):
        """Combine signals from multiple ROIs using dynamic weighted average."""
        if not roi_signals:
            return 0.0

        # Calculate weighted average using current dynamic weights
        total_weight = 0.0
        weighted_sum = 0.0

        for roi_name, signal_value in roi_signals.items():
            weight = self.roi_weights.get(roi_name, 0.0)
            weighted_sum += signal_value * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback to simple average if no weights
            return np.mean(list(roi_signals.values()))

    def enhance_ppg_signal(self, signal):
        """Enhanced PPG signal processing without ICA - more stable approach."""
        try:
            # 1. Apply bandpass filter to focus on heart rate frequencies
            nyquist = self.fs / 2
            low = 0.8 / nyquist  # 48 BPM
            high = 3.0 / nyquist  # 180 BPM
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)

            # 2. Apply adaptive smoothing based on signal quality
            signal_quality = self._assess_signal_quality(filtered)
            if signal_quality > 0.7:  # High quality signal
                # Light smoothing
                smoothed = savgol_filter(filtered, 7, 2)
            else:  # Lower quality signal
                # More aggressive smoothing
                smoothed = savgol_filter(filtered, 11, 3)

            # 3. Remove baseline wander with more robust method
            window_size = min(30, len(smoothed) // 4)
            if window_size > 5:
                baseline = np.convolve(smoothed, np.ones(window_size)/window_size, mode='same')
                enhanced = smoothed - baseline
            else:
                enhanced = smoothed

            # 4. Apply temporal consistency check
            enhanced = self._apply_temporal_consistency(enhanced)

            print(f"Signal enhancement: quality={signal_quality:.2f}, smoothing={'light' if signal_quality > 0.7 else 'aggressive'}")

            return enhanced

        except Exception as e:
            self.logger.error(f"Signal enhancement error: {e}")
            return signal

    def _apply_temporal_consistency(self, signal):
        """Apply temporal consistency to prevent sudden changes."""
        try:
            # Check for sudden amplitude changes
            diff = np.abs(np.diff(signal))
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)

            # If there are sudden large changes, apply additional smoothing
            if std_diff > 2 * mean_diff:
                signal = savgol_filter(signal, 9, 2)
                print("Applied additional smoothing due to temporal inconsistency")

            return signal

        except Exception:
            return signal

    def _assess_signal_quality(self, signal):
        """Enhanced assessment of PPG signal quality."""
        try:
            # 1. Basic signal statistics
            signal_std = np.std(signal)
            signal_mean = np.abs(np.mean(signal))

            # Check for flat or constant signals
            if signal_std < 1e-6 or signal_mean < 1e-6:
                return 0.0

            # 2. Calculate signal-to-noise ratio in heart rate band
            freqs, power = welch(signal, fs=self.fs, nperseg=min(256, len(signal)))

            # Heart rate band power (0.8-3.0 Hz = 48-180 BPM)
            hr_mask = (freqs >= 0.8) & (freqs <= 3.0)
            hr_power = np.sum(power[hr_mask])
            total_power = np.sum(power)

            # 3. Check for dominant frequency content
            if total_power < 1e-6:
                return 0.0

            # SNR in heart rate band
            snr = hr_power / (total_power - hr_power + 1e-6)

            # 4. Temporal consistency check
            # Check for sudden amplitude changes that might indicate motion artifacts
            diff_signal = np.abs(np.diff(signal))
            temporal_consistency = 1.0 / (1.0 + np.std(diff_signal) / (np.mean(diff_signal) + 1e-6))

            # 5. Frequency domain quality
            # Look for clear peaks in the heart rate band
            hr_power_spectrum = power[hr_mask]

            if len(hr_power_spectrum) > 0:
                # Find the dominant peak
                max_power_idx = np.argmax(hr_power_spectrum)
                max_power = hr_power_spectrum[max_power_idx]
                avg_power = np.mean(hr_power_spectrum)

                # Peak prominence (how much the peak stands out)
                peak_prominence = max_power / (avg_power + 1e-6)
                frequency_quality = min(peak_prominence / 3.0, 1.0)  # Normalize
            else:
                frequency_quality = 0.0

            # 6. Combined quality score with multi-ROI considerations
            # Weight different quality metrics
            base_quality = (0.4 * min(snr / 2.0, 1.0) +  # SNR component
                           0.3 * temporal_consistency +    # Temporal consistency
                           0.3 * frequency_quality)        # Frequency domain quality

            # Multi-ROI quality boost: if we have multiple ROIs contributing, boost quality
            roi_contribution_factor = 1.0
            if hasattr(self, 'roi_weights'):
                # Count how many ROIs are contributing significantly
                contributing_rois = sum(1 for weight in self.roi_weights.values() if weight > 0.1)
                if contributing_rois >= 2:
                    roi_contribution_factor = 1.1  # 10% boost for multi-ROI
                elif contributing_rois >= 3:
                    roi_contribution_factor = 1.2  # 20% boost for all ROIs

            quality = base_quality * roi_contribution_factor

            # Ensure quality is in [0,1] range
            quality = max(0.0, min(1.0, quality))

            print(f"Signal quality: SNR={snr:.2f}, Temporal={temporal_consistency:.2f}, "
                  f"Freq={frequency_quality:.2f}, Multi-ROI={roi_contribution_factor:.2f}, Overall={quality:.2f}")

            return quality

        except Exception as e:
            print(f"Signal quality assessment error: {e}")
            return 0.0

    def reset(self):
        """Reset the signal processor state for a fresh measurement."""
        # Clear all buffers
        for buffer in self.buffers.values():
            buffer.clear()
        self.combined_buffer.clear()

        self.last_bpm = None
        self.start_time = None
        self.initialized = False

        # Reset ROI stability tracking
        self.last_roi_stable = {
            'forehead': False,
            'left_cheek': False,
            'right_cheek': False
        }

        # Reset health tracking to give ROIs a fresh start
        self.roi_health = {
            'forehead': 1.0,
            'left_cheek': 1.0,
            'right_cheek': 1.0
        }
        for roi_name in self.roi_stability_history:
            self.roi_stability_history[roi_name].clear()

        # Reset weights to base values
        self.roi_weights = self.base_weights.copy()

        # Reset baseline system
        self.baseline_hr = None
        self.baseline_established = False
        self.smoothed_hr = None

        self.logger.debug("Signal processor reset - initialization delay will be applied")

    def was_roi_lost(self):
        """Check if ROI was lost and reset the flag."""
        if self.roi_lost:
            self.roi_lost = False
            return True
        return False
