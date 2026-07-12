# PulseVision - rPPG Heart Rate Detection Pipeline

Technical reference for the signal pipeline. For a from-scratch explanation aimed at
readers new to signal processing, see `HOW_HEART_RATE_IS_MEASURED.md`.

## Pipeline Overview

### 1. Input Video
- **Source**: webcam (default) or video file
- **Threading**: capture, face detection and signal processing run on a background
  worker thread (`gui/processing_worker.py`); the GUI thread only renders
- **Timestamps**: each sample is tagged with a time — video files use `frame_index / fps`,
  cameras use wall-clock time — so the true sampling rate can be measured downstream

### 2. Face Detection
- **Method**: MediaPipe Face Mesh, 468 landmarks per frame
- **Confidence**: `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`
- **ROIs**: forehead (≈70% × 26% of the face box) plus left and right cheeks
  (≈18% × 12%), positioned from the landmark bounding box
- **Smoothing**: ROI rectangles are EMA-smoothed (α = 0.6) to suppress detector jitter

### 3. Per-ROI RGB Extraction & Combination
- **Extraction**: mean R, G, B of each ROI patch (stored R, G, B for POS)
- **Stability check**: grayscale standard deviation of each patch; below an
  (ROI-size-adaptive) threshold the patch is flagged unstable
- **Health & weighting**: each ROI keeps a rolling stability history that feeds a health
  score (0–1); base weights (forehead 0.5, each cheek 0.25) are multiplied by health and
  re-normalized, so unstable regions fade out and recover automatically
- **Buffering**: the combined per-frame RGB value and its timestamp go into 10-second
  rolling buffers

### 4. Uniform Resampling
- The actual sampling rate is computed from the buffered timestamps
- The RGB channels are resampled onto a uniform time grid (per-channel linear
  interpolation) before spectral analysis
- If the measured rate is implausible (e.g. duplicate timestamps) the nominal rate is used

### 5. Pulse Extraction — POS
- **Method**: POS (Plane-Orthogonal-to-Skin, Wang et al., IEEE TBME 2017), an overlap-add
  sliding-window projection of the RGB trace onto a plane orthogonal to the skin-tone
  direction, which cancels common-mode intensity variation (lighting, motion)
- **Fallback**: `SignalProcessor(use_pos=False)` uses the green channel only

### 6. Preprocessing (`preprocessing.py`, `enhance_heart_rate_signal`)
- **Detrend**: remove DC offset and linear trend
- **Band-pass**: 4th-order Butterworth, 0.5–4.0 Hz, applied zero-phase (`filtfilt`);
  the pass-band is deliberately wider than the 0.67–3.0 Hz search band so the response is
  flat across all reportable heart rates. The filter is redesigned if the measured
  sampling rate drifts from the design rate.
- **Robust normalization**: median/MAD normalization with clipping at ±5 MAD
- **No notch filters** (mains flicker aliases outside the HR band at 30 fps) and **no
  extra smoothing passes** (Welch windowing handles residual noise)

### 7. Heart Rate Estimation (`hr_estimation.py`, `estimate`)
- **Method**: Welch PSD, `nperseg = min(512, N)`, `nfft = 4096` for fine resolution
- **Band**: 0.67–3.0 Hz (40–180 BPM)
- **Peak**: most prominent peak (prominence ≥ 15% of max band power)
- **Confidence**: `0.5 × prominence_score + 0.5 × snr_score`, where `snr_score` compares
  the dominant peak to the band's median (background) power
- The estimator only estimates; it does no temporal smoothing or change rejection

### 8. Warm-up Gate (`processor.py`)
- No estimate is produced until ≥ 5 s of signal is buffered (spectral resolution)
- The first reported reading requires 3 consecutive estimates with confidence ≥ 0.6
  agreeing within 10 BPM
- The gate is held while the raw green buffer contains motion-artifact samples; the
  confidence bar relaxes after ~17 s so a persistently mediocre signal still reports

### 9. Smoothing / Outlier Rejection (`filtering.py`, `HeartRateFilter`)
- **Single stage**: median-of-5 (rejects lone-frame outliers) followed by a
  confidence-weighted exponential moving average (α from 0.15 at low confidence to 0.6 at
  high), so confident readings track real changes quickly without chattering on noise
- **Physiological guard**: readings outside 40–180 BPM are dropped

### 10. Display & Storage
- Heart rate, frequency and the FFT spectrum drive three real-time plots (redrawn on a
  250 ms timer); ROI overlays are drawn on the video feed
- Instantaneous measurements are saved to SQLite at most once per second; a session
  summary (average BPM) is saved when the user stops

## Key Technical Details

### Sampling & Buffering
- **Nominal rate**: 30 Hz; **actual rate**: measured per buffer from timestamps
- **Buffer**: 10 seconds (300 samples at 30 fps)
- **Minimum before first estimate**: 5 seconds of signal

### Signal Processing Chain
```
Video → Face/ROI detection → per-ROI RGB → health-weighted combine →
timestamp resample → POS projection → detrend / band-pass / normalize →
Welch PSD peak → confidence → warm-up gate → median + EMA → output
```

### Quality Control
- **ROI health**: per-region stability tracking with adaptive weights
- **Confidence threshold**: 0.4 minimum to report a reading
- **Warm-up gate**: suppresses unreliable startup readings
- **Physiological validation**: 40–180 BPM range checks

## Current Configuration
- **Extraction**: POS (green-only available as a fallback)
- **ROIs**: forehead + left/right cheeks
- **Confidence threshold**: 0.4
- **Smoothing**: single median + confidence-weighted EMA stage
