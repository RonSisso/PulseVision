PulseVision

PulseVision is a system that extracts heart rate from normal video input using remote photoplethysmography (rPPG) techniques, eliminating the need for physical sensors. It enables accurate, non-contact heart rate monitoring in real time, leveraging advanced computer vision and signal processing methods.



Features


Face & ROI Detection – Detects the face and extracts regions of interest (forehead and cheeks) for optimal rPPG signal capture.

Signal Preprocessing & Enhancement – Applies smoothing, notch filtering, bandpass filtering, and normalization to remove noise and improve signal quality.

Heart Rate Estimation – Uses frequency-domain analysis (Welch’s PSD, FFT) and peak detection with stability checks to estimate heart rate in beats per minute.

Robust Filtering – Outlier rejection and smoothing (z-score, median) ensure stable results under motion and lighting changes.

User-Friendly Operation – Real-time GUI display for heart rate monitoring, with database integration for patient history.



System Requirements


Operating System: Windows 10

Python Version: 3.11.0

Hardware:

Webcam (≥30 FPS recommended)

Distance from camera: 30–60 cm

Good ambient lighting (works in indoor and outdoor environments)

Dependencies: Provided in requirements.txt



Installation


Clone the repository:
git clone https://github.com/RonSisso/PulseVision.git
cd pulsevision/implementation


Create and activate a virtual environment:

python -m venv venv

venv\Scripts\activate.ps1  # On Windows

source venv/bin/activate # On Linux


Install requirements:
pip install -r requirements.txt



Login Credentials (Initial Setup)

When you first run the system, use the following default credentials to log in as an administrator:

Username: admin

Password: admin123



Usage


Run the system with:
python src/main.py

Workflow

Start the application.

Log in / select patient.

Choose video input source (camera or video file).

System detects face and extracts ROI.

Real-time heart rate estimation is displayed on the GUI.



System Pipeline


The PulseVision pipeline:

Input Video – Capture from webcam or file.

Face Detection & ROI Tracking – Locate face, extract forehead and cheeks.

Green Channel Extraction – Use the green channel intensity as raw rPPG signal.

Signal Preprocessing – Smoothing, flicker removal, robust normalization.

Signal Enhancement – Bandpass filtering, adaptive smoothing, baseline removal.

Heart Rate Estimation – Welch’s PSD, FFT, peak scoring, stability checks.

Outlier Filtering – Temporal filtering and z-score outlier rejection.

GUI Output – Heart rate values and plots shown in real time.



Results


Achieved accuracy up to ±5 BPM compared to ground truth across ~20 subjects (ages 20–65).

Robust performance under varied lighting conditions (indoor/outdoor, artificial flicker).

Occasionally up to ±10 BPM deviation under sudden movement or severe lighting changes.



Challenges & Ablation


Lighting Flicker – Common artificial lighting introduces 1–2 Hz noise; solved with notch filtering.

Motion Artifacts – Head movements reduce signal quality; mitigated with ROI smoothing and stability checks.

ROI Selection – Tested forehead-only vs. multi-ROI (forehead + cheeks); multi-ROI improved robustness.

Signal Quality Variability – Explored different smoothing and bandpass parameters during ablation studies.



Future Work


Improve Robustness – Handle extreme lighting and rapid motion more effectively.

Additional Parameters – Extend to SpO₂ and other vital sign estimation.

Mobile Optimization – Adapt system for mobile devices or embedded systems.

Scalability – Larger-scale testing with diverse populations.



Authors


Nadav Reubens

Ron Sisso

Software Engineering Degree – Braude College








