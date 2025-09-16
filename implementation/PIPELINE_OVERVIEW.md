# PulseVision - rPPG Heart Rate Detection Pipeline

## Abstract Pipeline Overview

### 1. **Input Video**
- **Source**: Webcam (default) or video file
- **Format**: BGR color space, typically 30 FPS
- **Resolution**: Variable (webcam dependent)
- **Processing**: Frame-by-frame real-time processing

### 2. **Face Detection**
- **Method**: MediaPipe Face Mesh
- **Output**: 468 facial landmarks per frame
- **Confidence**: min_detection_confidence=0.5, min_tracking_confidence=0.5
- **Tracking**: Single face detection with landmark tracking

### 3. **ROI (Region of Interest) Tracking**
- **Multiple ROIs**: Forehead, Left Cheek, Right Cheek
- **Forehead**: Upper 30% of face bounding box
- **Cheeks**: Left and right side regions of face
- **Smoothing**: Exponential Moving Average (EMA) with α=0.6
- **Stability**: ROI stability checking to detect motion artifacts

### 4. **Green Channel Extraction**
- **Method**: Extract green channel (index 1) from BGR frame
- **Processing**: Spatial averaging within each ROI
- **Output**: Single green intensity value per ROI per frame
- **Buffering**: 10-second rolling buffer per ROI (300 samples at 30 FPS)

### 5. **Multi-ROI Signal Combination**
- **Weighting**: Dynamic weights based on ROI health/stability
- **Base Weights**: Forehead=0.5, Left Cheek=0.25, Right Cheek=0.25
- **Health Tracking**: ROI stability history (last 10 checks)
- **Adaptive**: Weights adjust based on ROI quality over time
- **Combination**: Weighted average of all valid ROIs

### 6. **Signal Preprocessing**
- **Detrending**: Remove linear trends
- **Bandpass Filter**: 0.67-3.0 Hz (40-180 BPM) Butterworth filter
- **Notch Filtering**: Remove 1Hz, 2Hz, 3Hz, 4Hz noise sources
- **Normalization**: Robust normalization using median and MAD
- **Smoothing**: Savitzky-Golay filter (adaptive window size)
- **Motion Artifact Removal**: Statistical outlier detection and interpolation

### 7. **Heart Rate Estimation (FFT Method)**
- **Method**: Welch's Power Spectral Density (PSD)
- **Window**: 512 samples with 50% overlap
- **Frequency Range**: 0.5-3.0 Hz (30-180 BPM)
- **Peak Detection**: Find most prominent peak in frequency domain
- **Confidence**: Based on peak prominence, SNR, and temporal consistency
- **Validation**: Physiological range check (40-180 BPM)

### 8. **Signal Quality Assessment**
- **SNR**: Signal-to-noise ratio in heart rate frequency band
- **Peak Clarity**: How distinct the main frequency peak is
- **Temporal Consistency**: Stability of recent measurements
- **Multi-ROI Quality**: Boost for multiple contributing ROIs

### 9. **Heart Rate Filtering & Smoothing**
- **Outlier Rejection**: Z-score based outlier detection
- **Physiological Constraints**: Limit sudden changes (>20 BPM)
- **Temporal Smoothing**: Exponential smoothing with confidence weighting
- **Baseline System**: Establish baseline after 20 seconds, gradual updates

### 10. **FFT Visualization**
- **Power Spectrum**: Real-time FFT power spectrum display
- **Frequency Range**: 0.5-3.0 Hz focus
- **Reference Lines**: 1.0Hz (60 BPM), 1.2Hz (72 BPM), 1.5Hz (90 BPM)
- **Peak Highlighting**: Mark detected heart rate frequency

### 11. **GUI Display & Output**
- **Real-time Display**: Heart rate, frequency, method indicator
- **Plots**: rPPG signal, heart rate over time, FFT spectrum
- **Method Indicator**: Shows "FFT" method being used
- **Data Storage**: Save measurements to database
- **Visual Feedback**: ROI overlays on video feed

## Key Technical Details

### **Sampling & Buffering**
- **Sampling Rate**: 30 Hz (30 FPS)
- **Buffer Size**: 10 seconds (300 samples)
- **Minimum Samples**: 2 seconds (60 samples) for processing
- **Update Rate**: Every 0.1 seconds (3 frames)

### **Signal Processing Chain**
```
Raw Video → Face Detection → ROI Tracking → Green Extraction → 
Multi-ROI Combination → Preprocessing → FFT Analysis → 
Peak Detection → Confidence Assessment → Filtering → Output
```

### **Quality Control**
- **ROI Health**: Track stability of each ROI over time
- **Signal Quality**: Multi-metric assessment (SNR, peak clarity, consistency)
- **Confidence Thresholds**: Minimum 0.4 confidence for FFT method
- **Physiological Validation**: Range checks and change limits

### **Performance Optimizations**
- **Multi-ROI**: Parallel processing of multiple face regions
- **Adaptive Processing**: Adjust parameters based on signal quality
- **Efficient Buffering**: Rolling buffers with fixed memory usage
- **Real-time Processing**: Optimized for 30 FPS video processing

## Current Configuration
- **Method**: FFT-only (cardiac cycle detection removed)
- **ROIs**: Forehead + Left/Right Cheeks
- **Confidence Threshold**: 0.4 for FFT method
- **Update Frequency**: 10 Hz (every 0.1 seconds)
- **Display**: Real-time plots with FFT spectrum visualization
