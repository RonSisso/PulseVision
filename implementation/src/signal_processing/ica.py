import numpy as np
from sklearn.decomposition import FastICA, PCA
from scipy.ndimage import shift
from scipy.stats import skew
from scipy.signal import welch

class ICAExtractor:
    def __init__(self, n_components=5, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.last_best_component = None
        self.component_stability_threshold = 0.5  # Much higher threshold to prevent switching
        self.fs = 30  # Sampling rate
        self.last_hr_freq = None  # Track last heart rate frequency
        self.freq_tolerance = 0.5  # Hz tolerance for frequency consistency (more flexible)

    def extract_best_component(self, signal):
        try:
            # Create phase-shifted matrix using scipy.ndimage.shift
            shifts = [0, 1, 2, 3, 4]
            X = np.column_stack([shift(signal, s, mode='nearest') for s in shifts])

            # Apply PCA to reduce noise
            pca = PCA(n_components=self.n_components)
            X_pca = pca.fit_transform(X)

            # Apply ICA
            ica = FastICA(n_components=self.n_components, random_state=self.random_state, max_iter=1000)
            S = ica.fit_transform(X_pca)

            # Score components using improved heart rate-specific metrics
            scores = [self._score_component(comp) for comp in S.T]
            
            # Component stability check with frequency tracking
            if self.last_best_component is not None:
                current_best_score = max(scores)
                last_score = scores[self.last_best_component]
                
                # Check frequency consistency for the best new component
                best_new_idx = np.argmax(scores)
                new_component = S[:, best_new_idx]
                new_hr_freq = self._estimate_heart_rate_frequency(new_component)
                
                # Frequency consistency check
                freq_consistent = True
                if self.last_hr_freq is not None and new_hr_freq is not None:
                    freq_diff = abs(new_hr_freq - self.last_hr_freq)
                    freq_consistent = freq_diff <= self.freq_tolerance
                    print(f"ICA: Frequency check - new: {new_hr_freq:.2f}Hz, last: {self.last_hr_freq:.2f}Hz, diff: {freq_diff:.2f}Hz, consistent: {freq_consistent}")
                
                # Only switch if new component is significantly better AND frequency is consistent
                if (current_best_score > last_score + self.component_stability_threshold and freq_consistent):
                    best_idx = best_new_idx
                    print(f"ICA: Switching to component {best_idx} (score: {current_best_score:.3f} vs {last_score:.3f})")
                else:
                    best_idx = self.last_best_component
                    if not freq_consistent:
                        print(f"ICA: Keeping component {best_idx} (frequency inconsistent)")
                    else:
                        print(f"ICA: Keeping component {best_idx} (stability check passed)")
            else:
                best_idx = np.argmax(scores)
                print(f"ICA: Initial component {best_idx} selected (score: {scores[best_idx]:.3f})")
            
            # Update tracking variables
            self.last_best_component = best_idx
            selected_component = S[:, best_idx]
            self.last_hr_freq = self._estimate_heart_rate_frequency(selected_component)
            
            return selected_component
        except Exception as e:
            print(f"ICA extraction error: {e}")
            return signal

    def _score_component(self, comp):
        """Score component based on heart rate signal characteristics - heart rate agnostic."""
        try:
            # 1. Frequency domain analysis - focus on heart rate band (0.8-3.0 Hz)
            freqs, power = welch(comp, fs=self.fs, nperseg=min(256, len(comp)))
            hr_mask = (freqs >= 0.8) & (freqs <= 3.0)  # 48-180 BPM
            hr_power = np.sum(power[hr_mask])
            total_power = np.sum(power)
            hr_snr = hr_power / (total_power - hr_power + 1e-6)
            
            # 2. Temporal consistency - check for smooth transitions
            temporal_consistency = self._calculate_temporal_consistency(comp)
            
            # 3. Periodicity in heart rate band
            periodicity = self._estimate_periodicity(comp)
            
            # 4. Signal stability (lower variance is better for PPG)
            stability = 1.0 / (np.std(comp) + 1e-6)
            stability = min(stability / 1000, 1.0)  # Normalize to [0,1]
            
            # 5. Peak prominence - prefer components with clear, dominant peaks
            peak_prominence = self._calculate_peak_prominence(comp)
            
            # Weighted scoring - prioritize signal quality over specific frequency
            score = (0.3 * hr_snr + 
                    0.3 * temporal_consistency + 
                    0.2 * peak_prominence + 
                    0.1 * periodicity + 
                    0.1 * stability)
            
            return score
            
        except Exception as e:
            print(f"Component scoring error: {e}")
            return 0.0
    
    def _calculate_peak_prominence(self, comp):
        """Calculate peak prominence - prefers components with clear, dominant peaks."""
        try:
            freqs, power = welch(comp, fs=self.fs, nperseg=min(256, len(comp)))
            hr_mask = (freqs >= 0.8) & (freqs <= 3.0)  # 48-180 BPM
            
            if not np.any(hr_mask):
                return 0.0
                
            hr_power = power[hr_mask]
            
            if len(hr_power) == 0:
                return 0.0
            
            # Find the maximum peak
            max_peak = np.max(hr_power)
            mean_power = np.mean(hr_power)
            
            # Calculate prominence (how much the peak stands out)
            prominence = (max_peak - mean_power) / (mean_power + 1e-6)
            
            return min(prominence / 10, 1.0)  # Normalize to [0,1]
            
        except Exception:
            return 0.5
    
    def _calculate_temporal_consistency(self, comp):
        """Calculate temporal consistency of the component."""
        try:
            # Check for smooth transitions (avoid sudden jumps)
            diff = np.abs(np.diff(comp))
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            
            # Lower mean difference and lower variance indicate better consistency
            consistency = 1.0 / (mean_diff + std_diff + 1e-6)
            return min(consistency / 10, 1.0)  # Normalize to [0,1]
            
        except Exception:
            return 0.5

    def _estimate_periodicity(self, comp):
        corr = np.correlate(comp, comp, mode='full')
        corr = corr[len(corr)//2:]
        peaks = np.diff(np.sign(np.diff(corr))) < 0
        peak_indices = np.where(peaks)[0]
        if len(peak_indices) < 2:
            return 0
        intervals = np.diff(peak_indices)
        return 1.0 / (np.std(intervals) + 1e-6)
    
    def _estimate_heart_rate_frequency(self, comp):
        """Estimate the dominant heart rate frequency from a component."""
        try:
            freqs, power = welch(comp, fs=self.fs, nperseg=min(256, len(comp)))
            hr_mask = (freqs >= 0.8) & (freqs <= 3.0)  # 48-180 BPM
            
            if not np.any(hr_mask):
                return None
                
            hr_freqs = freqs[hr_mask]
            hr_power = power[hr_mask]
            
            if len(hr_power) == 0:
                return None
                
            # Find the peak frequency in heart rate band
            peak_idx = np.argmax(hr_power)
            return hr_freqs[peak_idx]
            
        except Exception as e:
            print(f"Frequency estimation error: {e}")
            return None