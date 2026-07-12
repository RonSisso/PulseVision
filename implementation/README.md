# PulseVision - Real-Time Heart Rate Measurement from Video

PulseVision is a real-time heart rate monitoring system that uses remote photoplethysmography (rPPG) to detect heart rate from video of a person's face, without any physical contact or sensors.

## Features
- **Real-time video capture** from webcam or video files, processed on a background thread so the UI stays responsive
- **Face detection** using MediaPipe Face Mesh (468 landmarks)
- **Multi-ROI tracking** of the forehead and both cheeks with dynamic, stability-based weighting
- **POS colour-projection pulse extraction** (Plane-Orthogonal-to-Skin, Wang et al. 2017) using all three RGB channels — cancels the intensity changes from lighting and motion that green-only extraction cannot
- **Robust heart rate estimation** via Welch power-spectral-density analysis:
  - Sampling rate measured from frame timestamps (correct for any real frame rate, not assumed to be 30 fps)
  - A single, well-characterised filter chain (detrend → band-pass → robust normalization)
  - Confidence scoring from spectral peak quality
  - A warm-up gate that suppresses unreliable startup readings
- **Comprehensive GUI** with real-time visualization:
  - Live video feed with coloured ROI overlays
  - Heart rate display with physiological-range colour coding
  - Three synchronized plots: rPPG signal, heart rate trend, FFT spectrum
- **Patient management system** with SQLite database integration
- **Offline accuracy benchmark** (`src/benchmark.py`) for reproducible before/after evaluation of pipeline changes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RonSisso/PulseVision.git
cd PulseVision/implementation
```

2. Create a virtual environment (Python 3.11 required):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
implementation/
├── src/
│   ├── video/              # Video capture (camera / file)
│   ├── face_detection/     # MediaPipe face detection and ROI extraction
│   ├── signal_processing/  # POS extraction, filtering, HR estimation
│   ├── gui/                # PyQt5 GUI and the background processing worker
│   ├── database/           # Patient / measurement storage (SQLite)
│   └── benchmark.py        # Offline accuracy benchmark
├── benchmarks/             # Benchmark harness docs and saved results
├── requirements.txt        # Project dependencies
└── README.md
```

## Usage

1. **Activate the virtual environment** (see above).

2. **Run the main application**:
```bash
python src/main.py
```

3. **Run the accuracy benchmark** (no camera needed):
```bash
python src/benchmark.py --synthetic
```

## Technical Details

### Signal Processing Pipeline

Each webcam frame flows through the following stages (see `PIPELINE_OVERVIEW.md` for
the technical reference and `HOW_HEART_RATE_IS_MEASURED.md` for a from-scratch
explanation):

1. **Video capture** — frames are read on a worker thread; each carries a timestamp.
2. **Face detection** — MediaPipe Face Mesh locates 468 landmarks and three ROIs
   (forehead + both cheeks), whose positions are EMA-smoothed to reduce jitter.
3. **Per-ROI RGB extraction & weighting** — the mean R, G, B of each ROI is taken;
   ROIs are combined with weights that adapt to each region's stability.
4. **Uniform resampling** — the true sampling rate is measured from the timestamps
   and the buffered signal is resampled onto a uniform grid, so a frame rate other
   than 30 fps (or dropped frames) does not bias the result.
5. **POS pulse extraction** — the RGB trace is projected onto a plane orthogonal to
   the skin-tone direction, cancelling common-mode intensity noise (green-only
   extraction remains available as a fallback).
6. **Preprocessing** — detrend → 4th-order Butterworth band-pass (0.5–4.0 Hz,
   zero-phase) → median/MAD robust normalization with outlier clipping.
7. **Heart rate estimation** — Welch PSD; the most prominent peak in the 0.67–3.0 Hz
   (40–180 BPM) band gives the heart rate, with a confidence score from peak quality.
8. **Warm-up gate** — the first reading is withheld until several consecutive
   high-confidence estimates agree and the signal window is free of motion artifacts.
9. **Smoothing** — a single stage: median-of-5 outlier rejection followed by a
   confidence-weighted exponential moving average.
10. **Display & storage** — results drive the GUI plots and are saved to SQLite.

### Design notes

- **No notch filters.** At a ~30 fps sampling rate, mains-driven light flicker
  (100/120 Hz) aliases to ~10 Hz or DC, never into the heart-rate band; notches at
  1–3 Hz would instead sit directly on real heart rates (60–180 BPM).
- **One filter chain, not many.** Earlier revisions stacked several band-passes and
  smoothers; these attenuated real heart rates near the band edges. The current chain
  is a single band-pass whose pass-band is deliberately wider than the search band.
- **One smoothing stage.** Temporal stability is handled in a single median+EMA filter
  rather than several interacting layers.

### Technology Stack

- **Python 3.11**
- **OpenCV** — video capture and image processing
- **MediaPipe** — face detection and landmark tracking (468 points)
- **NumPy / SciPy** — signal processing
- **PyQt5** — GUI framework
- **Matplotlib** — real-time plotting

### System Requirements

- **Hardware**: standard webcam (720p+ recommended)
- **Software**: Python 3.11, OpenCV, MediaPipe, NumPy, SciPy, PyQt5
- **Processing**: modern multi-core CPU
- **Memory**: ~2 GB RAM
- **OS**: Windows, macOS, or Linux

## Accuracy & Validation

The project ships with an **offline benchmark harness** (`src/benchmark.py`) that runs
the real signal processor over reproducible test cases and reports MAE / RMSE / bias per
case. Two modes are provided:

- **Synthetic suite** (`--synthetic`) — generated signals with a known heart rate across
  45–180 BPM under clean, noisy, drifting-light, motion-spike and camera-warm-up
  conditions. On the current pipeline the synthetic suite reports a mean absolute error
  of well under 1 BPM with full coverage. These figures characterise the algorithm in
  controlled conditions; they are not a substitute for real-world validation.
- **Recorded-clip suite** (`--clips manifest.json`) — runs the full pipeline over face
  videos recorded alongside a reference device (e.g. an Apple Watch). Recording a small
  set of such clips and comparing against the reference is the recommended way to
  validate real-world accuracy; see `benchmarks/README.md` for the protocol.

An earlier prototype of the system spot-checked at 74.6 BPM detected vs 76 BPM reference
(~1.8% error) against an Apple Watch. The signal pipeline has since been substantially
revised, so the current build should be re-validated with recorded clips before any
accuracy figure is quoted.

- **Range**: 40–180 BPM (covers normal and exercise heart rates)
- **Confidence threshold**: 0.4 for a reading to be reported
