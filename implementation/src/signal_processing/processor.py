# import numpy as np
# from scipy import signal
# from sklearn.decomposition import FastICA
# import cv2
# from collections import deque
# import time
# import logging

# class SignalProcessor:
#     def __init__(self, sampling_rate=30):
#         """Initialize the signal processor."""
#         self.sampling_rate = sampling_rate
#         self.window_size = int(sampling_rate * 10)  # 10 seconds window
        
#         # Reduced thresholds for more lenient signal quality assessment
#         self.min_bpm_confidence = 0.5
#         self.signal_quality_threshold = 0.005
        
#         # Heart rate parameters
#         self.min_hr = 40
#         self.max_hr = 180
        
#         # Initialize filters
#         self.setup_filters()
        
#         # Signal processing parameters
#         self.window = signal.windows.hann(self.window_size)
#         self.smooth_hr = None
#         self.no_update_count = 0
#         self.max_no_update = 15  # Increased from 10
        
#         # BPM update control
#         self.last_update_time = time.time()
#         self.update_interval = 0.1
#         self.bpm_history = deque(maxlen=5)
        
#         self.baseline = None
#         self.values_history = []
#         self.debug = True
        
#         # Configure logging
#         self.logger = logging.getLogger(__name__)
        
#         # Buffer parameters
#         self.min_samples = 30  # Reduced from 60 to 30 (1 second of data)
        
#         # Initialize signal buffer
#         self.green_values = deque(maxlen=self.window_size)
        
#         # Cache for intermediate results
#         self.last_bpm = None
#         self.last_confidence = None
#         self.fft_freqs = np.fft.rfftfreq(self.window_size, d=1.0/self.sampling_rate)
#         self.valid_freq_mask = (self.fft_freqs >= self.min_hr/60) & (self.fft_freqs <= self.max_hr/60)
        
#         self.logger.info("SignalProcessor initialized with sampling rate: %d Hz", self.sampling_rate)

#     def setup_filters(self):
#         """Initialize all signal processing filters."""
#         nyquist = self.sampling_rate / 2
        
#         # Single bandpass filter for heart rate frequency range (0.75-3 Hz)
#         low = self.min_hr / 60 / nyquist
#         high = self.max_hr / 60 / nyquist
#         self.b_bandpass, self.a_bandpass = signal.butter(3, [low, high], btype='band')
        
#         # Single notch filter at 1 Hz (common noise frequency)
#         self.b_notch, self.a_notch = signal.iirnotch(1.0 / nyquist, 30.0)
        
#         # Moving average filter length
#         self.ma_length = 5

#     def preprocess_frame(self, frame, roi):
#         """Extract and preprocess the green channel from ROI."""
#         if frame is None or roi is None:
#             self.logger.warning("Invalid frame or ROI received")
#             return None
            
#         try:
#             x, y, w, h = roi
#             if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
#                 self.logger.warning("ROI coordinates out of frame bounds: (%d,%d,%d,%d)", x, y, w, h)
#                 return None
                
#             roi_frame = frame[y:y+h, x:x+w]
#             if roi_frame.size == 0:
#                 self.logger.warning("Empty ROI frame detected")
#                 return None
            
#             # Extract green channel
#             green = roi_frame[:, :, 1].astype(float)
            
#             # Apply CLAHE for contrast enhancement
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             green = clahe.apply(green.astype(np.uint8))
            
#             # Apply spatial averaging with weights
#             weights = np.ones_like(green) / green.size
#             green_mean = np.average(green, weights=weights)
            
#             self.logger.debug("Frame preprocessed - Green channel mean: %.4f", green_mean)
#             return green_mean
            
#         except Exception as e:
#             self.logger.error("Error in preprocess_frame: %s", str(e))
#             return None

#     def update_signal(self, value):
#         """Update the signal buffer with a new value."""
#         try:
#             self.green_values.append(value)
#             self.logger.debug("Signal buffer updated - Buffer size: %d", len(self.green_values))
#         except Exception as e:
#             self.logger.error("Error updating signal buffer: %s", str(e))

#     def apply_filters(self, data):
#         """Apply all filters to the signal."""
#         # Detrend and normalize
#         data = signal.detrend(data)
#         data = (data - np.mean(data)) / np.std(data)
        
#         # Apply all notch filters
#         for b, a in self.notch_filters:
#             data = signal.filtfilt(b, a, data)
        
#         # Apply bandpass filter
#         data = signal.filtfilt(self.b_bandpass, self.a_bandpass, data)
        
#         # Create window of correct size
#         window = signal.windows.hann(len(data))
        
#         # Apply window function
#         data = data * window
        
#         # Additional smoothing
#         data = signal.filtfilt(self.b_lowpass, self.a_lowpass, data)
        
#         return data

#     def extract_components(self, data):
#         """Apply ICA to extract independent components."""
#         try:
#             # Create multiple phase-shifted signals
#             shifts = [0, 1, 2, 3, 4]
#             X = np.column_stack([np.roll(data, shift) for shift in shifts])
            
#             # Apply ICA
#             ica = FastICA(n_components=len(shifts), random_state=42, max_iter=1000)
#             components = ica.fit_transform(X)
            
#             # Select the component with highest periodicity
#             periodicities = [self.estimate_periodicity(comp) for comp in components.T]
#             best_idx = np.argmax(periodicities)
            
#             # Check if the best component is good enough
#             if periodicities[best_idx] > self.signal_quality_threshold:
#                 return components[:, best_idx]
#             return data
#         except:
#             return data

#     def estimate_periodicity(self, data):
#         """Estimate the periodicity of a signal component."""
#         # Calculate autocorrelation
#         corr = np.correlate(data, data, mode='full')
#         corr = corr[len(corr)//2:]
        
#         # Find peaks in autocorrelation
#         peaks, properties = signal.find_peaks(corr, distance=10)
#         if len(peaks) < 2:
#             return 0
        
#         # Calculate peak quality metrics
#         peak_heights = corr[peaks]
#         peak_distances = np.diff(peaks)
        
#         # Assess periodicity quality
#         height_consistency = 1.0 - np.std(peak_heights) / np.mean(peak_heights)
#         distance_consistency = 1.0 - np.std(peak_distances) / np.mean(peak_distances)
        
#         return height_consistency * distance_consistency

#     def find_heart_rate(self, data):
#         """Calculate heart rate from frequency analysis."""
#         # Calculate FFT
#         fft_data = np.fft.rfft(data)
#         freqs = np.fft.rfftfreq(len(data), d=1/self.sampling_rate)
        
#         # Find peaks in the frequency domain
#         magnitude = np.abs(fft_data)
#         freq_mask = (freqs >= self.min_hr/60) & (freqs <= self.max_hr/60)
#         valid_freqs = freqs[freq_mask]
#         valid_magnitude = magnitude[freq_mask]
        
#         if len(valid_magnitude) == 0:
#             return None, 0
        
#         # Normalize magnitude
#         valid_magnitude = valid_magnitude / np.max(valid_magnitude)
        
#         # Find peaks in the magnitude spectrum with adjusted parameters
#         peaks, properties = signal.find_peaks(valid_magnitude, 
#                                             distance=8,  # Reduced for more peaks
#                                             height=0.15,
#                                             prominence=0.1)
#         if len(peaks) == 0:
#             return None, 0
        
#         # Calculate confidence based on multiple factors
#         peak_heights = valid_magnitude[peaks]
#         background = np.mean(valid_magnitude)
        
#         # Calculate signal quality metrics
#         snr = np.max(peak_heights) / background if background > 0 else 0
#         peak_ratio = np.max(peak_heights) / np.sum(peak_heights)
#         periodicity = self.estimate_periodicity(data)
        
#         # Dynamic confidence calculation
#         if self.no_update_count > 5:
#             # Give more weight to SNR when we haven't updated in a while
#             confidence = (0.4 * snr + 0.3 * peak_ratio + 0.3 * periodicity)
#         else:
#             # Normal weights
#             confidence = (0.3 * snr + 0.3 * peak_ratio + 0.4 * periodicity)
        
#         # Find the most prominent peak
#         max_idx = peaks[np.argmax(peak_heights)]
        
#         # Use quadratic interpolation for better frequency estimation
#         if 0 < max_idx < len(valid_magnitude) - 1:
#             try:
#                 true_freq = self.quadratic_peak_interp(
#                     valid_freqs[max_idx-1:max_idx+2],
#                     valid_magnitude[max_idx-1:max_idx+2]
#                 )
#             except:
#                 true_freq = valid_freqs[max_idx]
#         else:
#             true_freq = valid_freqs[max_idx]
        
#         heart_rate = true_freq * 60
        
#         if self.debug:
#             print(f"SNR: {snr:.2f}, Peak Ratio: {peak_ratio:.2f}, Periodicity: {periodicity:.2f}")
#             print(f"Frequency Resolution: {freqs[1]-freqs[0]:.3f} Hz")
#             if self.no_update_count > 0:
#                 print(f"No update count: {self.no_update_count}")
        
#         return heart_rate, confidence

#     def quadratic_peak_interp(self, freqs, magnitudes):
#         """Quadratic interpolation for peak frequency estimation."""
#         alpha = magnitudes[0]
#         beta = magnitudes[1]
#         gamma = magnitudes[2]
#         peak_pos = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
#         return freqs[1] + peak_pos * (freqs[1] - freqs[0])

#     def is_valid_bpm(self, bpm):
#         """Check if BPM value is physiologically plausible."""
#         if bpm is None:
#             return False
            
#         # Check if within physiological range
#         if not (self.min_hr <= bpm <= self.max_hr):
#             return False
            
#         # Check for sudden changes
#         if self.bpm_history:
#             last_bpm = self.bpm_history[-1]
#             max_change = min(25, 15 + self.no_update_count)
#             if abs(bpm - last_bpm) > max_change:
#                 return False
                
#         return True

#     def calculate_heart_rate(self):
#         """Calculate heart rate from the signal buffer."""
#         if len(self.green_values) < self.min_samples:
#             self.logger.info("Not enough samples for heart rate calculation")
#             return None, 0.0

#         try:
#             # Check if it's time to update
#             current_time = time.time()
#             if current_time - self.last_update_time < self.update_interval:
#                 return self.smooth_hr, 0.0

#             signal_data = np.array(list(self.green_values))
            
#             # Remove linear trend and normalize
#             signal_data = signal.detrend(signal_data)
#             signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-6)

#             # Apply all filters
#             filtered_signal = self.apply_filters(signal_data)
            
#             # Calculate FFT
#             fft_data = np.fft.rfft(filtered_signal)
#             freqs = np.fft.rfftfreq(len(filtered_signal), d=1.0/self.sampling_rate)
            
#             # Get the frequency range for heart rate
#             mask = (freqs >= self.min_hr/60) & (freqs <= self.max_hr/60)
#             valid_freqs = freqs[mask]
#             valid_magnitude = np.abs(fft_data[mask])
            
#             if len(valid_magnitude) == 0:
#                 self.logger.warning("No valid frequencies found in signal")
#                 return self.smooth_hr, 0.0
            
#             # Normalize magnitude
#             valid_magnitude = valid_magnitude / (np.max(valid_magnitude) + 1e-6)
            
#             # Find peaks with more lenient parameters
#             peaks, properties = signal.find_peaks(valid_magnitude,
#                                                 distance=5,  # Reduced from 8
#                                                 height=0.1,  # Reduced from 0.15
#                                                 prominence=0.05)  # Reduced from 0.1
            
#             if len(peaks) == 0:
#                 self.logger.warning("No peaks found in frequency spectrum")
#                 self.no_update_count += 1
#                 if self.no_update_count > self.max_no_update:
#                     return None, 0.0
#                 return self.smooth_hr, 0.0
            
#             # Find the most prominent peak
#             peak_heights = valid_magnitude[peaks]
#             max_peak_idx = peaks[np.argmax(peak_heights)]
#             peak_freq = valid_freqs[max_peak_idx]
            
#             # Calculate heart rate
#             heart_rate = peak_freq * 60
            
#             # Validate heart rate
#             if not self.is_valid_bpm(heart_rate):
#                 self.logger.warning(f"Invalid heart rate calculated: {heart_rate}")
#                 self.no_update_count += 1
#                 if self.no_update_count > self.max_no_update:
#                     return None, 0.0
#                 return self.smooth_hr, 0.0
            
#             # Update smooth heart rate with more aggressive smoothing
#             if self.smooth_hr is None:
#                 self.smooth_hr = heart_rate
#             else:
#                 # More aggressive smoothing for stability
#                 self.smooth_hr = 0.7 * self.smooth_hr + 0.3 * heart_rate
            
#             self.no_update_count = 0
#             self.last_update_time = current_time
            
#             self.logger.info(f"Heart rate calculated: {self.smooth_hr:.1f} BPM")
#             return self.smooth_hr, 0.0
            
#         except Exception as e:
#             self.logger.error(f"Error calculating heart rate: {str(e)}")
#             return self.smooth_hr, 0.0

#     def process_frame(self, frame, roi=None):
#         """Process a single frame and update the heart rate estimate."""
#         try:
#             if frame is None or roi is None:
#                 return None, 0.0

#             # Extract the green channel average from the ROI
#             x, y, w, h = roi
#             roi_frame = frame[y:y+h, x:x+w]
#             green_value = np.mean(roi_frame[:, :, 1])
            
#             # Process the raw value
#             processed_value = self.process_value(green_value)
            
#             # Add to the signal buffer
#             self.green_values.append(processed_value)

#             # Calculate heart rate if we have enough samples
#             if len(self.green_values) >= self.min_samples:
#                 current_time = time.time()
#                 if current_time - self.last_update_time < self.update_interval:
#                     return self.smooth_hr, 1.0 if self.smooth_hr is not None else 0.0

#                 # Convert buffer to numpy array
#                 signal_data = np.array(self.green_values)
                
#                 # Remove DC component and linear trend
#                 signal_data = signal_data.astype(float)
#                 signal_data = signal_data - np.mean(signal_data)
#                 signal_data = signal.detrend(signal_data)
                
#                 # Apply bandpass filter
#                 filtered_signal = signal.filtfilt(self.b_bandpass, self.a_bandpass, signal_data)
                
#                 # Apply window function
#                 windowed_signal = filtered_signal * self.window[:len(filtered_signal)]
                
#                 # Calculate FFT
#                 fft_data = np.fft.rfft(windowed_signal)
#                 freqs = np.fft.rfftfreq(len(windowed_signal), d=1.0/self.sampling_rate)
                
#                 # Get the frequency range for heart rate
#                 mask = (freqs >= self.min_hr/60) & (freqs <= self.max_hr/60)
#                 valid_freqs = freqs[mask]
#                 valid_magnitude = np.abs(fft_data[mask])
                
#                 if len(valid_magnitude) == 0:
#                     return self.smooth_hr, 0.0
                
#                 # Normalize magnitude
#                 valid_magnitude = valid_magnitude / np.max(valid_magnitude)
                
#                 # Find peaks in the magnitude spectrum
#                 peaks, properties = signal.find_peaks(valid_magnitude,
#                                                     distance=5,
#                                                     height=0.1,
#                                                     prominence=0.1)
                
#                 if len(peaks) == 0:
#                     self.no_update_count += 1
#                     if self.no_update_count > self.max_no_update:
#                         return None, 0.0
#                     return self.smooth_hr, 0.5
                
#                 # Find the most prominent peak
#                 peak_heights = valid_magnitude[peaks]
#                 max_peak_idx = peaks[np.argmax(peak_heights)]
#                 peak_freq = valid_freqs[max_peak_idx]
                
#                 # Calculate confidence metrics
#                 background = np.mean(valid_magnitude)
#                 snr = np.max(peak_heights) / background if background > 0 else 0
#                 peak_ratio = np.max(peak_heights) / np.sum(peak_heights)
                
#                 # Calculate heart rate
#                 heart_rate = peak_freq * 60
                
#                 # Validate heart rate
#                 if not self.is_valid_bpm(heart_rate):
#                     self.no_update_count += 1
#                     if self.no_update_count > self.max_no_update:
#                         return None, 0.0
#                     return self.smooth_hr, 0.5
                
#                 # Calculate confidence
#                 confidence = 0.4 * snr + 0.6 * peak_ratio
                
#                 if confidence > self.min_bpm_confidence:
#                     # Update smooth heart rate
#                     if self.smooth_hr is None:
#                         self.smooth_hr = heart_rate
#                     else:
#                         # Smooth with variable rate based on confidence
#                         alpha = min(0.7, max(0.3, confidence))
#                         self.smooth_hr = (1 - alpha) * self.smooth_hr + alpha * heart_rate
                    
#                     self.bpm_history.append(heart_rate)
#                     self.no_update_count = 0
#                     self.last_update_time = current_time
                    
#                     self.logger.info(f"Heart Rate: {self.smooth_hr:.1f} BPM (Confidence: {confidence:.2f})")
#                     return self.smooth_hr, confidence
#                 else:
#                     self.no_update_count += 1
#                     if self.no_update_count > self.max_no_update:
#                         return None, 0.0
#                     return self.smooth_hr, 0.5
            
#             return None, 0.0

#         except Exception as e:
#             self.logger.error(f"Error processing frame: {str(e)}")
#             return None, 0.0

#     def process_value(self, value):
#         """Process a raw green channel value."""
#         # Update baseline with slower adaptation
#         if self.baseline is None:
#             self.baseline = value
#         else:
#             # Use dynamic alpha based on signal stability
#             dynamic_alpha = min(0.95, max(0.05, 1.0 - np.std(self.values_history) if self.values_history else 0.1))
#             self.baseline = dynamic_alpha * self.baseline + (1 - dynamic_alpha) * value

#         # Store value in history
#         self.values_history.append(value)
#         if len(self.values_history) > self.window_size:
#             self.values_history.pop(0)

#         # Calculate rolling statistics
#         if len(self.values_history) >= 3:
#             recent_mean = np.mean(self.values_history[-10:] if len(self.values_history) >= 10 else self.values_history)
#             recent_std = np.std(self.values_history[-10:] if len(self.values_history) >= 10 else self.values_history)
#             # Use adaptive standard deviation threshold
#             std_dev = max(recent_std, 0.1 * abs(recent_mean - self.baseline))
#         else:
#             recent_mean = value
#             std_dev = 1.0

#         # Normalize with respect to recent statistics
#         normalized = (value - recent_mean) / (std_dev + 1e-6)  # Prevent division by zero
        
#         # Apply non-linear amplification
#         amplified = np.sign(normalized) * np.log1p(abs(normalized)) * 5.0
        
#         # Clip extreme values
#         amplified = np.clip(amplified, -10.0, 10.0)

#         self.logger.debug(f"Raw: {value:.4f}, Norm: {normalized:.4f}, Amp: {amplified:.4f}")
#         return amplified

#     def update_heart_rate(self, bpm):
#         # Add new heart rate to history
#         self.hr_history.append(bpm)
#         if len(self.hr_history) > self.hr_window:
#             self.hr_history.pop(0)

#         # Apply median filter to reject outliers
#         if len(self.hr_history) >= 3:
#             filtered_hr = np.median(self.hr_history)
#             # Only accept new heart rate if it's within 20% of median
#             if abs(bpm - filtered_hr) > 0.2 * filtered_hr:
#                 bpm = filtered_hr

#         # Smooth heart rate changes
#         if len(self.hr_history) >= 2:
#             prev_hr = self.hr_history[-2]
#             # Limit rate of change
#             max_change = 2.0  # Maximum allowed BPM change per update
#             if abs(bpm - prev_hr) > max_change:
#                 if bpm > prev_hr:
#                     bpm = prev_hr + max_change
#                 else:
#                     bpm = prev_hr - max_change

#         return bpm

#     def calculate_signal_quality(self):
#         """Calculate signal quality metrics."""
#         if len(self.green_values) < self.min_samples:
#             return 0.0
            
#         try:
#             # Convert buffer to numpy array
#             signal = np.array(self.green_values)
            
#             # Calculate signal-to-noise ratio with adaptive thresholds
#             signal_mean = np.mean(signal)
#             signal_std = np.std(signal)
            
#             # More lenient minimum threshold based on signal mean
#             min_std_threshold = 0.01 * abs(signal_mean)
#             if signal_std < min_std_threshold:
#                 return 0.0
                
#             # Calculate SNR using signal variation
#             background_noise = np.median(np.abs(np.diff(signal)))
#             snr = 20 * np.log10(signal_std / (background_noise + 1e-6))
#             snr_quality = min(1.0, max(0.0, (snr + 10) / 20))  # Adjusted scaling
            
#             # Calculate temporal consistency with windowed analysis
#             window_size = min(len(signal) // 4, 20)
#             windows = [signal[i:i+window_size] for i in range(0, len(signal)-window_size, window_size//2)]
#             if windows:
#                 window_stds = [np.std(w) for w in windows]
#                 temporal_consistency = 1.0 / (1.0 + 0.5 * np.std(window_stds) / (np.mean(window_stds) + 1e-6))
#             else:
#                 temporal_consistency = 0.0
            
#             # Calculate frequency domain metrics
#             window = signal.windows.hann(len(signal))
#             windowed = signal * window
#             fft = np.fft.rfft(windowed)
#             freqs = np.fft.rfftfreq(len(signal), d=1.0/self.sampling_rate)
            
#             # Focus on heart rate frequency range (0.7-4.0 Hz)
#             mask = (freqs >= 0.7) & (freqs <= 4.0)
#             hr_freqs = freqs[mask]
#             hr_mags = np.abs(fft[mask])
            
#             if len(hr_mags) == 0:
#                 return 0.0
                
#             # Calculate peak quality with improved metrics
#             peaks, properties = signal.find_peaks(hr_mags, distance=5, prominence=0.1)
#             if len(peaks) == 0:
#                 freq_quality = 0.0
#             else:
#                 # Calculate peak quality based on prominence and width
#                 prominences = properties["prominences"]
#                 peak_quality = np.max(prominences) / (np.sum(hr_mags) + 1e-6)
#                 freq_quality = min(1.0, peak_quality * 2.0)
            
#             self.logger.debug(f"Signal quality metrics - SNR: {snr_quality:.3f}, "
#                             f"Temporal: {temporal_consistency:.3f}, "
#                             f"Frequency: {freq_quality:.3f}")
            
#             # Combine metrics with dynamic weights based on frequency quality
#             if freq_quality > 0.5:  # Strong frequency content
#                 quality = max(0.2, (0.2 * snr_quality +
#                           0.3 * temporal_consistency +
#                           0.5 * freq_quality))  # Emphasize frequency quality
#             else:
#                 quality = max(0.2, (0.4 * snr_quality +
#                           0.4 * temporal_consistency +
#                           0.2 * freq_quality))  # More balanced weights
            
#             return quality
            
#         except Exception as e:
#             self.logger.error(f"Error calculating signal quality: {str(e)}")
#             return 0.0

#STEP 1: Face Detection & ROI Tracking (MediaPipe + Smoothing)

from collections import deque
import numpy as np
import logging
import time
from .preprocessing import SignalPreprocessor
from .hr_estimation import HeartRateEstimator
from .filtering import HeartRateFilter, ROIStabilityChecker
from signal_processing.preprocessing import SignalPreprocessor
from signal_processing.hr_estimation import HeartRateEstimator
from signal_processing.filtering import HeartRateFilter, ROIStabilityChecker
from signal_processing.performance import ParallelProcessor

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
        self.green_values = []#TODO: check if needed
        
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

        # Step 2: Additional PPG signal enhancement (replacing ICA)
        signal = self.enhance_ppg_signal(signal)

        # Step 5: Heart rate estimation using FFT method only
        freq_bpm, freq_confidence = self.hr_estimator.estimate(signal)
        
        # Use FFT method (which is working well based on your feedback)
        if freq_bpm is not None and freq_confidence > 0.4:
            bpm, confidence = freq_bpm, freq_confidence
            method_used = "frequency_domain"
            print(f"Using FFT method: {bpm:.1f} BPM, confidence: {confidence:.2f}")
        else:
            # FFT method failed
            bpm, confidence = None, 0.0
            method_used = "none"
            print("FFT heart rate estimation failed")

        # Step 6: Enhanced confidence validation
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

        # Step 7: Baseline-based heart rate calculation
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
                from scipy.signal import welch
                window_size = min(512, len(signal))
                fft_freqs, fft_power = welch(signal, fs=self.fs, nperseg=window_size, nfft=2**10)
                
                # Focus on heart rate frequency range (0.5-3 Hz)
                hr_mask = (fft_freqs >= 0.5) & (fft_freqs <= 3.0)
                fft_freqs = fft_freqs[hr_mask]
                fft_power = fft_power[hr_mask]
            except Exception as e:
                print(f"FFT calculation error: {e}")
                fft_freqs, fft_power = None, None

        # Include method data in return (no cardiac cycle data needed)
        method_data = {
            'method_used': method_used if 'method_used' in locals() else 'none'
        }

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

    def enhance_ppg_signal(self, signal):
        """
        Minimal PPG signal enhancement - baseline implementation for ablation study.
        Currently: No enhancement (pass-through)
        """
        # Step 1: No enhancement - just return the signal as-is
        return signal

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
            from scipy.signal import butter, filtfilt
            nyquist = self.fs / 2
            low = 0.8 / nyquist  # 48 BPM
            high = 3.0 / nyquist  # 180 BPM
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
            
            # 2. Apply adaptive smoothing based on signal quality
            signal_quality = self._assess_signal_quality(filtered)
            if signal_quality > 0.7:  # High quality signal
                # Light smoothing
                from scipy.signal import savgol_filter
                smoothed = savgol_filter(filtered, 7, 2)
            else:  # Lower quality signal
                # More aggressive smoothing
                from scipy.signal import savgol_filter
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
                from scipy.signal import savgol_filter
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
            from scipy.signal import welch
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
            hr_freqs = freqs[hr_mask]
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
        self.green_values.clear()
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






#STEP 2: Signal Preprocessing Enhancements




#STEP 3: ICA Optimization with PCA + Component Scoring



#STEP 4: Heart Rate Estimation Enhancements


#STEP 5: Signal Quality & Outlier Handling



#STEP 6: Performance Optimizations



#STEP 7: Testing & Evaluation Utilities

