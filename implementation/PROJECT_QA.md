# PulseVision - rPPG Heart Rate Detection Project Q&A

## **Project Overview**

### Q: What is PulseVision?
**A:** PulseVision is a real-time heart rate monitoring system that uses remote photoplethysmography (rPPG) to detect heart rate from video of a person's face, without any physical contact or sensors.

### Q: What is rPPG?
**A:** Remote Photoplethysmography (rPPG) is a non-contact method that uses video cameras to detect subtle color changes in the skin caused by blood volume changes during the cardiac cycle. When the heart pumps blood, it causes slight color variations in the face that can be captured and analyzed.

### Q: What are the advantages of rPPG over traditional methods?
**A:** 
- **Non-contact**: No need for physical sensors or electrodes
- **Convenient**: Works with standard webcams
- **Continuous monitoring**: Real-time heart rate detection
- **Cost-effective**: Uses existing camera hardware
- **Comfortable**: No skin irritation or discomfort

---

## **Technical Pipeline Questions**

### Q: How does the system work step by step?
**A:** The pipeline follows these steps:
1. **Input Video** - Capture video on a background worker thread, tagging each frame with a timestamp
2. **Face Detection** - Use MediaPipe to detect 468 facial landmarks
3. **ROI Tracking** - Track the forehead and both cheeks with stability-based weighting
4. **Per-ROI RGB Extraction** - Take the mean red, green and blue of each region
5. **Uniform Resampling** - Measure the true sampling rate from the timestamps and resample
6. **POS Pulse Extraction** - Combine the RGB channels to isolate the pulse from lighting/motion
7. **Signal Preprocessing** - Detrend, band-pass filter, robust-normalize
8. **Heart Rate Estimation** - Welch PSD; most prominent peak in the heart-rate band
9. **Warm-up Gate & Smoothing** - Withhold unreliable startup readings; median + confidence-weighted EMA
10. **GUI Display & Storage** - Show real-time results and save to the database

### Q: Which colour channels do you use, and why?
**A:** The heartbeat is **strongest in green** (hemoglobin absorbs green light most
strongly), but it is present in red and blue too. Crucially, lighting and motion change all
three channels *together*, while the pulse changes them in a fixed, unequal ratio. So rather
than using green alone, the system uses **all three channels** and combines them with the
**POS** method (next question) to separate the pulse from lighting/motion noise. Green-only
extraction is retained as a fallback for comparison.

### Q: What is POS and why did you add it?
**A:** POS (Plane-Orthogonal-to-Skin, Wang et al. 2017) is a standard rPPG technique. It
treats each moment as a point in Red-Green-Blue space and projects it onto the plane
perpendicular to the skin-tone/intensity direction. Lighting and motion changes move all
channels together (the intensity direction) and are cancelled; the pulse, which moves the
channels in a different direction, survives. This makes the measurement far more robust to
lighting and head motion than green-only extraction — the error mode that dominates real
webcam use. In the project's offline benchmark, POS cut the error on motion-heavy cases
roughly five-fold versus green-only.

### Q: Why use multiple ROIs (forehead and cheeks)?
**A:** Multiple ROIs provide several benefits:
- **Redundancy**: If one region fails, others can compensate
- **Better signal quality**: Different regions may have varying signal quality
- **Motion robustness**: If the person moves, different regions may remain stable
- **Dynamic weighting**: System automatically adjusts weights based on ROI quality

### Q: How does the face detection work?
**A:** We use MediaPipe Face Mesh which:
- Detects 468 facial landmarks in real-time
- Provides robust tracking even with head movement
- Works with various lighting conditions
- Has built-in confidence scoring for reliability

### Q: What is the FFT method and why use it?
**A:** FFT (Fast Fourier Transform) converts the time-domain signal to frequency domain:
- **Purpose**: Find the dominant frequency corresponding to heart rate
- **Range**: Analyzes 0.5-3.0 Hz (30-180 BPM)
- **Advantage**: Robust to noise and artifacts
- **Accuracy**: Provides reliable heart rate estimation

---

## **Signal Processing Questions**

### Q: How do you handle noise and artifacts?
**A:** In order of impact:
- **POS colour projection**: cancels lighting/motion changes (the largest error source) by
  combining the RGB channels
- **Multi-ROI combination**: averaging three stability-weighted regions reduces local artifacts
- **Band-pass filtering**: a single 0.5–4.0 Hz Butterworth band-pass keeps only heart-rate
  frequencies (its pass-band is wider than the search band so real heart rates aren't weakened)
- **Robust (median/MAD) normalization**: keeps one motion spike from distorting the signal's scale
- **Welch averaging**: averaging overlapping spectra suppresses random noise in the frequency domain

Note there are deliberately **no notch filters**: at ~30 fps, mains flicker aliases outside
the heart-rate band, and a notch at 1–3 Hz would delete real heart rates (60–180 BPM).

### Q: How do you ensure signal quality?
**A:** Quality assessment includes:
- **SNR**: Signal-to-noise ratio in heart rate band
- **Peak clarity**: How distinct the main frequency peak is
- **Temporal consistency**: Stability of recent measurements
- **ROI health**: Track stability of each region over time
- **Confidence scoring**: Overall reliability assessment

### Q: What happens if the signal quality is poor?
**A:** The system has several safeguards:
- **Dynamic ROI weighting**: reduce the weight of poor-quality regions automatically
- **Confidence threshold**: only readings above 0.4 confidence are reported
- **Warm-up gate**: the first reading is withheld until several consistent, high-confidence
  estimates agree and the signal is free of motion artifacts
- **Median + EMA smoothing**: a lone bad estimate is stepped over by the median; the display
  moves faster only when confidence is high
- **Physiological range check**: values outside 40–180 BPM are rejected

---

## **Performance & Accuracy Questions**

### Q: What is the accuracy of the system?
**A:** Accuracy is characterised with a reproducible **offline benchmark** (`src/benchmark.py`):
- **Synthetic suite**: over generated signals with a known heart rate (45–180 BPM, under
  noise, drift, motion spikes and camera warm-up), the current pipeline reports a mean
  absolute error well under 1 BPM with full coverage. This measures the *algorithm* in
  controlled conditions.
- **Real-world validation**: the same harness can run over face videos recorded next to a
  reference device (`--clips`). An earlier prototype spot-checked at 74.6 vs 76 BPM (~1.8%)
  against an Apple Watch; because the pipeline has since been substantially revised, the
  current build should be re-validated with recorded clips before a real-world figure is quoted.
- **Range**: 40–180 BPM; **confidence threshold**: 0.4 to report a reading.

### Q: What is the processing speed?
**A:** Real-time performance:
- **Frame rate**: 30 FPS video processing
- **Update rate**: 10 Hz heart rate updates (every 0.1 seconds)
- **Latency**: <100ms from frame capture to heart rate display
- **Buffer size**: 10-second rolling window for analysis

### Q: What are the system requirements?
**A:** 
- **Hardware**: Standard webcam (720p+ recommended)
- **Software**: Python 3.11+, OpenCV, MediaPipe, NumPy, SciPy
- **Processing**: Modern CPU (multi-core recommended)
- **Memory**: ~2GB RAM for real-time processing
- **OS**: Windows, macOS, or Linux

---

## **Implementation Questions**

### Q: What programming languages and libraries are used?
**A:** 
- **Python 3.11**: Main programming language
- **OpenCV**: Video capture and image processing
- **MediaPipe**: Face detection and landmark tracking
- **NumPy/SciPy**: Signal processing and mathematical operations
- **PyQt5**: GUI framework for user interface
- **Matplotlib**: Real-time plotting and visualization

### Q: How is the GUI structured?
**A:** The interface includes:
- **Video display**: Live camera feed with ROI overlays
- **Real-time metrics**: Heart rate, frequency, method indicator
- **Three plots**: rPPG signal, heart rate over time, FFT spectrum
- **Controls**: Start/stop measurement, reset, patient management
- **Database integration**: Save measurements and patient data

### Q: How do you handle different lighting conditions?
**A:** 
- **POS colour projection**: lighting changes move all colour channels together and are
  cancelled by the projection — this is the main defence
- **Detrending**: removes slow brightness drift (auto-exposure, a passing cloud)
- **Robust (median/MAD) normalization**: resists single large brightness excursions
- **Multi-ROI weighting**: down-weights regions that fall into shadow

---

## **Challenges & Solutions Questions**

### Q: What are the main challenges in rPPG?
**A:** 
- **Motion artifacts**: Head movement, talking, facial expressions
- **Lighting variations**: Changes in ambient light
- **Skin tone differences**: Varying signal strength across individuals
- **Noise sources**: Electronic interference, camera noise
- **Real-time processing**: Computational efficiency requirements

### Q: How do you handle motion artifacts?
**A:** 
- **ROI stability checking**: Detect when regions become unstable
- **Dynamic weighting**: Reduce weight of moving regions
- **Temporal smoothing**: Smooth out sudden changes
- **Multi-ROI redundancy**: Use stable regions when others fail
- **Outlier detection**: Filter out motion-induced spikes

### Q: What if the person moves or talks?
**A:** 
- **Robust tracking**: MediaPipe handles moderate head movement
- **Multiple ROIs**: System continues with stable regions
- **Quality assessment**: Automatically detects degraded signal
- **Graceful degradation**: Maintains last good measurement
- **Recovery**: Quickly adapts when stable conditions return

---

## **Comparison & Validation Questions**

### Q: How do you validate the accuracy?
**A:** 
- **Reference device**: Apple Watch ECG for ground truth
- **Controlled testing**: Known heart rate scenarios
- **Real-world testing**: Various lighting and movement conditions
- **Statistical analysis**: Error rates and confidence intervals
- **Continuous monitoring**: Long-term stability assessment

### Q: How does this compare to other heart rate monitoring methods?
**A:** 
- **vs. Chest straps**: More comfortable, no skin contact
- **vs. Smartwatches**: No hardware required, works with any camera
- **vs. Pulse oximeters**: Non-contact, continuous monitoring
- **vs. ECG**: Less accurate but more convenient
- **vs. Other rPPG systems**: Optimized for real-time performance

### Q: What are the limitations?
**A:** 
- **Validation**: real-world accuracy still needs confirmation with recorded reference clips
- **Lighting dependency**: Requires adequate lighting
- **Motion sensitivity**: Significant movement can affect accuracy
- **Individual variation**: May work better for some people than others
- **Processing requirements**: Needs reasonable computational power

---

## **Future Development Questions**

### Q: What improvements could be made?
**A:** 
- **Machine learning**: neural rPPG models (e.g. learned pulse extraction)
- **Skin-tone-adaptive processing**: tune extraction per individual
- **3D face modeling**: Better ROI selection
- **Mobile optimization**: Smartphone app development
- **Medical validation**: Clinical testing and certification

### Q: What are potential applications?
**A:** 
- **Telemedicine**: Remote patient monitoring
- **Fitness tracking**: Exercise heart rate monitoring
- **Stress monitoring**: Workplace wellness programs
- **Accessibility**: Heart rate monitoring for people with disabilities
- **Research**: Large-scale heart rate studies

### Q: How could this be commercialized?
**A:** 
- **Healthcare**: Integration with telemedicine platforms
- **Fitness**: Smart gym equipment integration
- **Automotive**: Driver fatigue monitoring
- **Security**: Stress detection in security applications
- **Consumer**: Smart home health monitoring

---

## **Technical Deep Dive Questions**

### Q: Why did you choose FFT over other methods?
**A:** 
- **Proven reliability**: FFT is well-established for frequency analysis
- **Noise robustness**: Works well with noisy signals
- **Real-time performance**: Efficient computation
- **Interpretability**: Easy to visualize and debug
- **Accuracy**: Provides good results for heart rate detection

### Q: How do you handle the sampling rate and buffer size?
**A:** 
- **Measured sampling rate**: the true rate is computed from frame timestamps (not assumed
  to be 30 fps), and the signal is resampled onto a uniform grid — so any real frame rate reads correctly
- **10-second buffer**: balances latency against frequency resolution
- **Rolling window**: fixed-size buffers keep memory constant and processing real-time
- **Minimum before a first reading**: 5 seconds of signal, for adequate frequency resolution

### Q: What is the confidence scoring system?
**A:** 
- **Formula**: `0.5 × peak_prominence_score + 0.5 × snr_score`, where the SNR score compares
  the dominant spectral peak to the band's median (background) power
- **Meaning**: high only when a single sharp peak dominates the heart-rate band
- **Threshold**: 0.4 minimum to report a reading; the warm-up gate requires ≥ 0.6 for the first reading
- **Use**: also sets how fast the smoothing follows a change (confident readings move faster)

---

## **Demo & Presentation Questions**

### Q: Can you show a live demonstration?
**A:** Yes, the system can demonstrate:
- **Real-time processing**: Live video feed with ROI overlays
- **Heart rate detection**: Current BPM with confidence indicator
- **Signal visualization**: Three real-time plots
- **Method indicator**: Shows "FFT" method being used
- **Quality feedback**: Visual indication of signal quality

### Q: What should I expect to see during the demo?
**A:** 
- **Video feed**: Your face with colored ROI rectangles
- **Heart rate**: Real-time BPM display (typically 60-100 BPM at rest)
- **Frequency**: Corresponding frequency in Hz
- **Plots**: Signal waveform, heart rate trend, FFT spectrum
- **Stability**: System should maintain consistent readings

### Q: How long does it take to get a stable reading?
**A:** 
- **Signal build-up**: at least 5 seconds of video are buffered before any estimate
- **First reading**: about 7 seconds in — deliberately withheld until it is trustworthy
  (three consistent high-confidence estimates, no motion artifact in the window)
- **Under heavy motion at startup**: the first reading may wait up to ~17 seconds
- **Steady tracking**: the reading then follows real heart-rate changes within a few seconds

---

## **Troubleshooting Questions**

### Q: What if the system doesn't detect my face?
**A:** 
- **Lighting**: Ensure adequate, even lighting
- **Position**: Face the camera directly
- **Distance**: Stay 1-3 feet from camera
- **Background**: Avoid cluttered backgrounds
- **Camera**: Check camera permissions and functionality

### Q: What if the heart rate seems inaccurate?
**A:** 
- **Stay still**: Minimize head movement
- **Good lighting**: Ensure even, bright lighting
- **Wait**: Allow 20-30 seconds for stabilization
- **Check confidence**: Look for high confidence values
- **Compare**: Use reference device for validation

### Q: What if the system is slow or laggy?
**A:** 
- **Close other applications**: Free up CPU resources
- **Check camera settings**: Lower resolution if needed
- **Update drivers**: Ensure latest camera drivers
- **Restart application**: Clear any memory issues
- **System requirements**: Verify adequate hardware

---

## **Conclusion Questions**

### Q: What are the key achievements of this project?
**A:** 
- **Real-time performance**: background-threaded processing with a responsive GUI
- **Literature-standard signal chain**: POS colour extraction plus a clean, well-characterised
  filter and estimation pipeline
- **Reproducible evaluation**: an offline benchmark that measures accuracy on every change
- **Robust startup and motion handling**: a warm-up gate and stability-weighted multi-ROI design
- **Complete system**: end-to-end rPPG with patient management and storage

### Q: What did you learn from this project?
**A:** 
- **Signal processing**: Advanced techniques for biomedical signals
- **Computer vision**: Face detection and tracking methods
- **Real-time systems**: Performance optimization and buffering
- **GUI development**: PyQt5 and real-time visualization
- **Project management**: Full-stack development and testing

### Q: What would you do differently next time?
**A:** 
- **Machine learning**: Explore AI-based signal enhancement
- **Mobile development**: Create smartphone app version
- **Clinical validation**: More extensive testing with medical devices
- **Performance optimization**: Further computational improvements
- **User experience**: Enhanced interface and feedback systems
