"""Multi-ROI rPPG signal processing orchestrator.

Combines the green-channel traces from the forehead and cheek ROIs into a
single signal, cleans it, and estimates heart rate in the frequency domain.

Timing model: every sample carries an explicit timestamp (seconds). The
actual sampling rate is measured from those timestamps and the samples are
resampled onto a uniform grid before spectral analysis, so dropped/jittered
frames or video files with fps != 30 do not skew the estimated heart rate.
"""

from collections import deque
from dataclasses import dataclass
import logging
import time

import numpy as np
from scipy.signal import welch

from .preprocessing import SignalPreprocessor
from .hr_estimation import HeartRateEstimator
from .filtering import HeartRateFilter, ROIStabilityChecker
from .pos import pos_pulse


@dataclass
class ProcessorResult:
    """Output of one processing step."""
    bpm: float | None
    confidence: float
    fft_freqs: np.ndarray | None = None
    fft_power: np.ndarray | None = None
    method: str = 'none'


class SignalProcessor:
    # Accept measured sampling rates only within this plausible range;
    # otherwise fall back to the nominal rate.
    MIN_PLAUSIBLE_FS = 5.0
    MAX_PLAUSIBLE_FS = 120.0

    # Do not estimate until this much signal has accumulated: a short window
    # has too little spectral resolution and the first readings lock onto
    # noise or the pulse's second harmonic.
    MIN_ANALYSIS_SECONDS = 5.0

    # The first reported reading must be backed by this many high-confidence
    # estimates agreeing within the tolerance, so startup transients (camera
    # auto-exposure, harmonic lock-in) never reach the display. The confidence
    # bar excludes the estimator's "sticky" repeats (confidence <= 0.5), which
    # would otherwise fake agreement. If warm-up hasn't passed after
    # WARMUP_RELAX_AFTER_S, the bar drops to the normal display threshold so
    # a persistently mediocre signal still produces a reading eventually.
    FIRST_READING_CONSISTENCY_N = 3
    FIRST_READING_TOLERANCE_BPM = 10.0
    FIRST_READING_MIN_CONFIDENCE = 0.6
    WARMUP_RELAX_AFTER_S = 17.0  # ~2s init + 5s buffer + 10s grace

    # During warm-up, hold the gate while the raw buffer contains artifact
    # samples (deviating more than WARMUP_ARTIFACT_MAD robust units from the
    # buffer median): a motion/exposure artifact inside the analysis window
    # can produce a high-confidence, self-consistent but wrong estimate.
    # Checked on the raw (pre-filter) signal where spikes keep full amplitude.
    WARMUP_ARTIFACT_MAD = 5.0
    WARMUP_MAX_ARTIFACT_FRACTION = 0.01

    def __init__(self, sampling_rate=30, use_pos=True):
        self.logger = logging.getLogger(__name__)
        self.fs = sampling_rate  # nominal rate; actual rate is measured per buffer
        self.use_pos = use_pos   # POS colour projection vs. green-channel only

        # Combined RGB signal buffer and per-sample timestamps (~10 s at nominal rate)
        self.combined_buffer = deque(maxlen=self.fs * 10)
        self.sample_times = deque(maxlen=self.fs * 10)

        self.preprocessor = SignalPreprocessor(sampling_rate)
        self.hr_estimator = HeartRateEstimator(sampling_rate)
        self.hr_filter = HeartRateFilter()
        self.roi_checker = ROIStabilityChecker()
        self.last_bpm = None

        # Warm-up: hold back output until the first estimates are consistent
        self.reporting_started = False
        self.first_reading_candidates = deque(maxlen=self.FIRST_READING_CONSISTENCY_N)

        # Initialization delay to avoid startup noise (in sample time)
        self.init_delay = 2.0  # seconds
        self.start_time = None  # timestamp of the first processed sample
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

    def process_frame(self, frame, roi, timestamp=None):
        """Process one frame given a single ROI or a dict of ROIs."""
        if isinstance(roi, dict):
            return self.process_multiple_rois(frame, roi, timestamp)
        else:
            # Legacy single ROI format - convert to multi-ROI format
            return self.process_multiple_rois(frame, {'forehead': roi}, timestamp)

    def process_multiple_rois(self, frame, rois, timestamp=None):
        """Process multiple ROIs and combine their signals for heart rate estimation.

        `timestamp` is the sample time in seconds (any consistent origin).
        When omitted, wall-clock time is used.
        """
        if timestamp is None:
            timestamp = time.time()

        # Check for complete failure (no frame or no ROIs at all)
        if frame is None:
            print("Frame is None - resetting signal processor")
            self.reset()
            self.roi_lost = True
            return ProcessorResult(None, 0.0)

        if rois is None or not any(rois.values()):
            print("All ROIs are None - face left frame, resetting signal processor")
            self.reset()
            self.roi_lost = True
            return ProcessorResult(None, 0.0)

        # Initialize start time from the first sample
        if self.start_time is None:
            self.start_time = timestamp
            print(f"Starting initialization delay: {self.init_delay} seconds")

        # Check if initialization delay is complete (in sample time)
        if not self.initialized and (timestamp - self.start_time) < self.init_delay:
            return ProcessorResult(None, 0.0)
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

            # Extract per-channel means. OpenCV frames are BGR; store as
            # R, G, B so POS receives channels in its expected order.
            try:
                b = float(np.mean(roi_patch[:, :, 0]))
                g = float(np.mean(roi_patch[:, :, 1]))
                r = float(np.mean(roi_patch[:, :, 2]))
                roi_signals[roi_name] = np.array([r, g, b], dtype=float)
                valid_rois += 1
            except Exception as e:
                print(f"Error processing {roi_name} ROI: {e}")
                self._update_roi_health(roi_name, False)

        # Update dynamic weights based on ROI health
        weights_changed = self._update_dynamic_weights()

        if valid_rois == 0:
            print("No valid ROIs found - returning last valid measurement")
            return ProcessorResult(self.last_bpm, 0.0)

        # Combine signals from all valid ROIs
        combined_signal = self._combine_roi_signals(roi_signals)

        # Apply smoothing if weights changed to prevent signal jumps
        if weights_changed and len(self.combined_buffer) > 0:
            # Smooth the transition when weights change
            last_signal = self.combined_buffer[-1]
            combined_signal = 0.7 * last_signal + 0.3 * combined_signal

        self.combined_buffer.append(combined_signal)
        self.sample_times.append(timestamp)

        # Wait for MIN_ANALYSIS_SECONDS of signal (or a full buffer at high
        # frame rates) before estimating at all
        span = self.sample_times[-1] - self.sample_times[0]
        if span < self.MIN_ANALYSIS_SECONDS and len(self.combined_buffer) < self.combined_buffer.maxlen:
            return ProcessorResult(None, 0.0)

        # Measure the actual sampling rate and resample RGB onto a uniform grid
        rgb_uniform, actual_fs = self._uniform_signal()
        green = rgb_uniform[:, 1]

        # Extract the pulse: POS colour projection (all channels) or green only
        if self.use_pos:
            try:
                pulse = pos_pulse(rgb_uniform, actual_fs)
            except Exception as e:
                print(f"POS extraction failed ({e}); falling back to green")
                pulse = green
        else:
            pulse = green

        # Clean the trace (detrend -> bandpass -> normalize)
        signal = self.preprocessor.enhance_heart_rate_signal(pulse, fs=actual_fs)

        # Heart rate estimation in the frequency domain
        freq_bpm, freq_confidence = self.hr_estimator.estimate(signal, fs=actual_fs)

        if freq_bpm is not None and freq_confidence > 0.4:
            bpm, confidence = freq_bpm, freq_confidence
            method_used = "frequency_domain"
            print(f"Using FFT method: {bpm:.1f} BPM, confidence: {confidence:.2f}")
        else:
            # FFT method failed
            bpm, confidence = None, 0.0
            method_used = "none"
            print("FFT heart rate estimation failed")

        # Confidence validation
        if bpm is not None and confidence is not None:
            min_confidence_threshold = 0.4

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

        # Warm-up gate: hold back the very first reading until several
        # high-confidence estimates agree, so a startup artifact is never shown
        if not self.reporting_started:
            relaxed = (timestamp - self.start_time) > self.WARMUP_RELAX_AFTER_S
            min_conf = 0.4 if relaxed else self.FIRST_READING_MIN_CONFIDENCE

            # Artifact check on the raw green brightness (not the POS pulse,
            # which deliberately suppresses these artifacts): while the window
            # holds outlier samples it is contaminated and even confident
            # estimates cannot be trusted yet
            med = np.median(green)
            mad = np.median(np.abs(green - med)) + 1e-6
            artifact_fraction = float(np.mean(np.abs(green - med) > self.WARMUP_ARTIFACT_MAD * mad))
            contaminated = (not relaxed) and artifact_fraction > self.WARMUP_MAX_ARTIFACT_FRACTION

            if bpm is not None and confidence >= min_conf and not contaminated:
                self.first_reading_candidates.append(bpm)
                candidates = list(self.first_reading_candidates)
                if (len(candidates) == self.FIRST_READING_CONSISTENCY_N
                        and max(candidates) - min(candidates) <= self.FIRST_READING_TOLERANCE_BPM):
                    self.reporting_started = True
                    print(f"Warm-up complete - first reading: {bpm:.1f} BPM")

            if not self.reporting_started:
                return ProcessorResult(None, 0.0)

        # Baseline-based heart rate calculation (baseline timing in sample time)
        if bpm is not None:
            time_since_start = timestamp - self.start_time if self.start_time else 0

            if not self.baseline_established and time_since_start >= self.baseline_time:
                # Establish baseline after 20 seconds
                self.baseline_hr = bpm
                self.baseline_established = True
                self.smoothed_hr = bpm
                print(f"Baseline HR established: {bpm:.1f} BPM")

            if self.baseline_established:
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
                fft_freqs, fft_power = welch(signal, fs=actual_fs, nperseg=window_size, nfft=2**10)

                # Focus on heart rate frequency range (0.5-3 Hz)
                hr_mask = (fft_freqs >= 0.5) & (fft_freqs <= 3.0)
                fft_freqs = fft_freqs[hr_mask]
                fft_power = fft_power[hr_mask]
            except Exception as e:
                print(f"FFT calculation error: {e}")
                fft_freqs, fft_power = None, None

        return ProcessorResult(filtered_bpm, confidence, fft_freqs, fft_power, method_used)

    def _uniform_signal(self):
        """Resample the buffered RGB samples onto a uniform time grid.

        Returns (rgb, actual_fs) where rgb is (N, 3). Uses the mean sampling
        rate measured from the timestamps; falls back to the nominal rate if
        the measurement is implausible (e.g. duplicate timestamps).
        """
        t = np.asarray(self.sample_times, dtype=float)
        v = np.asarray(self.combined_buffer, dtype=float)  # (N, 3)

        span = t[-1] - t[0]
        if span <= 0:
            return v, float(self.fs)

        actual_fs = (len(t) - 1) / span
        if not (self.MIN_PLAUSIBLE_FS <= actual_fs <= self.MAX_PLAUSIBLE_FS):
            self.logger.warning(
                "Implausible measured sampling rate %.1f Hz; using nominal %d Hz",
                actual_fs, self.fs)
            return v, float(self.fs)

        uniform_t = np.linspace(t[0], t[-1], len(t))
        out = np.empty_like(v)
        for c in range(v.shape[1]):
            out[:, c] = np.interp(uniform_t, t, v[:, c])
        return out, float(actual_fs)

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

        return weights_changed

    def _combine_roi_signals(self, roi_signals):
        """Combine per-ROI RGB vectors using dynamic weighted average."""
        if not roi_signals:
            return np.zeros(3)

        # Calculate weighted average using current dynamic weights
        total_weight = 0.0
        weighted_sum = np.zeros(3)

        for roi_name, signal_value in roi_signals.items():
            weight = self.roi_weights.get(roi_name, 0.0)
            weighted_sum += signal_value * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback to simple average if no weights
            return np.mean(list(roi_signals.values()), axis=0)

    def reset(self):
        """Reset the signal processor state for a fresh measurement."""
        # Clear buffers
        self.combined_buffer.clear()
        self.sample_times.clear()

        self.last_bpm = None
        self.start_time = None
        self.initialized = False

        # Reset warm-up gate
        self.reporting_started = False
        self.first_reading_candidates.clear()

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
