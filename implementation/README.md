# PulseVision - Real-Time Heart Rate Measurement from Video

PulseVision is a real-time heart rate monitoring system that uses remote photoplethysmography (rPPG) to detect heart rate from video of a person's face, without any physical contact or sensors.

## Features
- **Real-time video capture** from webcam or video files
- **Advanced face detection** using MediaPipe Face Mesh (468 landmarks)
- **Multi-ROI tracking** of forehead and cheek regions with dynamic weighting
- **Robust heart rate measurement** using FFT-based frequency analysis:
  - Green channel extraction for optimal blood volume detection
  - Advanced signal preprocessing and noise filtering
  - Real-time signal quality assessment
  - Confidence-based measurement validation
- **Comprehensive GUI** with real-time visualization:
  - Live video feed with colored ROI overlays
  - Real-time heart rate display with confidence indicators
  - Three synchronized plots: rPPG signal, heart rate trend, FFT spectrum
  - Method indicator showing "FFT" analysis
- **Patient management system** with database integration
- **High accuracy**: Within 2-3 BPM of reference devices (tested vs Apple Watch ECG)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RonSisso/PulseVision.git
cd PulseVision/implementation
```

2. Create a virtual environment (Python 3.11 required):
```bash
python -m venv venv_311
# On Windows:
venv_311\Scripts\activate.ps1
# On macOS/Linux:
source venv_311/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── src/
│   ├── video/              # Video capture and processing
│   ├── face_detection/     # MediaPipe face detection and landmark tracking
│   ├── signal_processing/  # FFT-based heart rate estimation and filtering
│   ├── gui/               # PyQt5 GUI with real-time visualization
│   └── database/          # Patient data management
├── tests/                 # Unit tests
├── data/                  # Sample data and resources
├── venv_311/             # Python 3.11 virtual environment
└── requirements.txt       # Project dependencies
```

## Usage

1. **Activate the virtual environment**:
```bash
# Windows:
venv_311\Scripts\activate.ps1
# macOS/Linux:
source venv_311/bin/activate
```

2. **Run the main application**:
```bash
python src/main.py
```

## Technical Details

### Signal Processing Pipeline

The rPPG heart rate measurement follows an 11-step pipeline:

1. **Input Video** - Capture video from webcam or file
2. **Face Detection** - Use MediaPipe to detect 468 facial landmarks
3. **ROI Tracking** - Track forehead and cheek regions with stability monitoring
4. **Green Channel Extraction** - Extract green color intensity for optimal blood volume detection
5. **Multi-ROI Signal Combination** - Combine signals from multiple regions with dynamic weighting
6. **Signal Preprocessing** - Filter and clean the signal:
   - CLAHE enhancement for lighting normalization
   - Motion artifact removal using MAD-based outlier detection
   - Bandpass filtering (0.67-3.0 Hz for 40-180 BPM)
   - Notch filtering for power line interference
7. **Heart Rate Estimation** - Use FFT (Fast Fourier Transform) to find heart rate frequency:
   - Welch's Power Spectral Density analysis
   - Peak detection in heart rate band (0.5-3.0 Hz)
   - Confidence calculation based on peak quality and signal quality
8. **Signal Quality Assessment** - Evaluate signal reliability:
   - SNR (Signal-to-Noise Ratio) analysis
   - Peak clarity assessment
   - Temporal consistency checking
9. **Heart Rate Filtering** - Smooth and validate results:
   - Physiological range validation (40-180 BPM)
   - Outlier rejection using Z-score analysis
   - Temporal smoothing with moving average
10. **FFT Visualization** - Display frequency spectrum for analysis
11. **GUI Display** - Show real-time results with three synchronized plots

### Performance Features

- **Real-time processing**: 30 FPS video with 10 Hz heart rate updates
- **High accuracy**: Within 2-3 BPM of reference devices (tested vs Apple Watch ECG)
- **Robust signal processing**: Handles motion artifacts and lighting variations
- **Multi-ROI redundancy**: System continues working even if some regions fail
- **Confidence-based validation**: Only displays high-confidence measurements
- **Baseline establishment**: 20-second baseline for stable measurements

## Development

### Technology Stack

- **Python 3.11**: Main programming language
- **OpenCV**: Video capture and image processing
- **MediaPipe**: Face detection and landmark tracking (468 points)
- **NumPy/SciPy**: Signal processing and mathematical operations
- **PyQt5**: GUI framework for user interface
- **Matplotlib**: Real-time plotting and visualization

### Key Components

1. **Video Input**: OpenCV-based video capture with 30 FPS processing
2. **Face Detection**: MediaPipe Face Mesh for robust facial landmark tracking
3. **Signal Processing**: 
   - Multi-ROI tracking with dynamic weighting
   - Green channel extraction for optimal blood volume detection
   - Advanced preprocessing with CLAHE and motion artifact removal
   - FFT-based frequency analysis for heart rate estimation
   - Real-time signal quality assessment
4. **GUI**: PyQt5-based interface with three synchronized real-time plots
5. **Database**: Patient management system for storing measurements

### System Requirements

- **Hardware**: Standard webcam (720p+ recommended)
- **Software**: Python 3.11, OpenCV, MediaPipe, NumPy, SciPy, PyQt5
- **Processing**: Modern CPU (multi-core recommended)
- **Memory**: ~2GB RAM for real-time processing
- **OS**: Windows, macOS, or Linux

## Accuracy & Validation

- **Reference validation**: Tested against Apple Watch ECG measurements
- **Real-world accuracy**: 74.6 BPM detected vs 76 BPM true (1.8% error)
- **Confidence threshold**: 0.4+ for reliable measurements
- **Range**: 40-180 BPM (covers normal and exercise heart rates)
- **Stability**: 20-second baseline establishment for consistent readings

