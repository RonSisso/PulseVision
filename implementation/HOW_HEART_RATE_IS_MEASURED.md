# How PulseVision Measures Heart Rate From a Video — Explained From Zero

This document explains, step by step, exactly how the code in this project turns a plain
webcam video into a heart rate number. It assumes **no prior knowledge** of signal
processing, and every step is taken directly from the actual source code (file names and
key numbers are cited throughout).

---

## Part 0 — The Physical Principle: Why a Camera Can See Your Heartbeat

Every time your heart beats, it pushes a wave of blood through your arteries and into the
tiny blood vessels (capillaries) under the skin of your face. For a fraction of a second,
there is slightly **more blood** under the skin than a moment before.

Blood absorbs light. More blood under the skin = slightly **more light absorbed** =
the skin looks **very slightly darker** to a camera. When the wave passes, the skin
becomes very slightly brighter again.

So the brightness of your facial skin actually flickers in rhythm with your heartbeat.
The flicker is invisible to the human eye — the brightness changes by less than 1% — but
it is present in the numbers of a digital camera image. This technique is called
**remote photoplethysmography (rPPG)**: "photo" = light, "plethysmography" = measuring
volume change, "remote" = without touching the person.

**One more key fact:** blood absorbs **green** light much more strongly than red or blue
light (hemoglobin, the molecule that carries oxygen in blood, has its strongest light
absorption around the green wavelength of ~530 nanometers). This means the heartbeat
flicker is **strongest in the green channel** — but it is still present in red and blue.

That last point matters. Lighting changes and head movements also change the brightness of
the skin — usually by a lot more than the heartbeat does — but they change **all three
colour channels together**. The heartbeat, by contrast, changes the colours in a specific,
fixed ratio (strong in green, weaker in red and blue). PulseVision uses this difference to
separate the real pulse from lighting and motion noise (Part 4).

The whole job of the software is therefore:

> Find the face → find good skin areas → record the average red, green and blue brightness
> of that skin 30 times per second → combine the colours to isolate the pulse → clean the
> remaining noise → find the rhythm hidden in it → convert the rhythm into
> beats-per-minute (BPM).

---

## Part 1 — Capturing the Video (`src/video/capture.py`, `src/gui/processing_worker.py`)

A webcam produces a stream of **frames** — still images, roughly 30 per second (30 FPS,
frames per second). Each frame is a grid of pixels, and each pixel stores three numbers
between 0 and 255: how much Blue, Green, and Red it contains (OpenCV stores them in
B-G-R order).

The `VideoCapture` class opens the webcam and delivers these frames one at a time. All of
this — reading frames, finding the face, and the signal processing — runs on a **background
worker thread** (`ProcessingWorker`), so the on-screen interface stays smooth no matter how
much computation each frame needs.

**Every frame is tagged with a timestamp.** Rather than *assuming* 30 frames arrive every
second, the system records *when* each frame actually arrived and later measures the true
rate from those timestamps. This matters because the sampling rate (`fs`, samples per
second) is the ruler used to convert an oscillation speed into BPM — if the real rate is 25
fps but the code assumes 30, every heart rate comes out ~20% too high. Measuring the rate
means a slow camera, a dropped frame, or a video file recorded at 24 or 60 fps all read
correctly.

---

## Part 2 — Finding the Face and Choosing Skin Regions (`src/face_detection/mediapipe_detector.py`)

### 2.1 Face landmark detection

Each frame is passed to **MediaPipe Face Mesh** (a Google library) which finds a face and
returns **468 landmark points** on it — precise positions along the eyebrows, nose,
jawline, etc. The detector is configured for one face at a time (`max_num_faces=1`) with a
50% minimum detection confidence.

### 2.2 Computing the face bounding box

The code takes all 468 landmark coordinates and finds the leftmost, rightmost, topmost,
and bottommost points. That rectangle is the face's **bounding box**, and its width and
height are used to place the skin regions proportionally — so the regions scale correctly
whether the person is near or far from the camera.

### 2.3 The three Regions of Interest (ROIs)

A **Region of Interest (ROI)** is just a rectangle of pixels we choose to analyze. The
code extracts three, using proportions of the face bounding box:

| ROI | Width | Height | Position |
|---|---|---|---|
| **Forehead** | 70% of face width | 26% of face height | Centered horizontally, near the top of the face |
| **Left cheek** | 18% of face width | 12% of face height | 10% in from the left edge, slightly above face center |
| **Right cheek** | 18% of face width | 12% of face height | 10% in from the right edge, same height |

Why these areas? They are flat, well-lit patches of bare skin with lots of capillaries,
and they are rarely covered by hair, glasses, or facial hair. Why **three** regions
instead of one? If one region is momentarily ruined (a shadow, a hand, hair falling), the
other two still carry the signal.

### 2.4 Smoothing the rectangle positions (EMA)

Face detection jitters — the detected rectangle jumps around by a few pixels every frame
even when the person is still. If the rectangle jumps, the pixels inside it change, which
creates fake brightness changes that pollute our measurement.

The fix is an **Exponential Moving Average (EMA)**: instead of using the newly detected
position directly, each coordinate is blended with its previous position:

```
smoothed_position = 0.6 × previous_position + 0.4 × new_position
```

(`ema_alpha = 0.6` in the code.) The rectangle therefore glides smoothly instead of jumping.

---

## Part 3 — Turning Pixels Into Signals (`src/signal_processing/processor.py`, method `process_multiple_rois`)

### 3.1 Three colour numbers per region per frame

For each of the three ROIs, the code cuts that rectangle out of the frame and computes the
**average red, average green, and average blue** of every pixel inside it. Averaging
hundreds or thousands of pixels cancels out random per-pixel camera noise, leaving the tiny
shared colour change we care about.

Result: three colour values per region per frame. We keep **all three colours** (not just
green) because the next stage needs colour information to tell the pulse apart from lighting
and motion.

### 3.2 Checking whether each region is trustworthy (`filtering.py`, class `ROIStabilityChecker`)

Before trusting a region, the code checks that it actually contains textured skin and not
something wrong (a blurred smear, a wall behind the head, a solid occlusion). The check is
simple: convert the patch to grayscale and compute the **standard deviation** of its pixel
values — a measure of how much variety there is. Real skin under real lighting has texture;
a patch that is nearly one flat colour (standard deviation below a small threshold, relaxed
for the smaller cheek patches) is flagged as **unstable**.

### 3.3 Health scores and dynamic weighting

Each ROI keeps a record of its last 10 stability checks. From this the code maintains a
**health score** from 0.0 (always failing) to 1.0 (always stable), updated gently each
frame (only 5% of the new result is mixed in, so the score can't whipsaw).

The three per-frame colour values are merged into **one colour value** using a weighted
average. The starting weights are: forehead 50%, each cheek 25% — but each weight is
multiplied by that region's current health and then re-normalized. A cheek that keeps
failing its stability check fades out of the average automatically; if it recovers, it
fades back in. (If the weights shift by more than 5% between frames, the newly combined
value is blended with the previous one so the re-weighting itself never creates a jump.)

### 3.4 The rolling buffer and the true sampling rate

Each combined RGB value — and the timestamp of its frame — is appended to **buffers** that
hold the last 10 seconds (about **300 samples** at 30 FPS; `deque(maxlen=fs*10)`). When the
buffer is full, the oldest value drops off.

Before spectral analysis, the code measures the **actual sampling rate** from the stored
timestamps (samples ÷ elapsed time) and **resamples** the buffer onto an evenly-spaced time
grid. From here on, the analysis works on a clean, uniformly-sampled recording whose sample
rate is known to be correct.

**One warm-up rule:** no heart rate is computed until at least **5 seconds** of signal has
accumulated. A shorter recording cannot resolve frequencies finely enough — the first
guesses tend to lock onto noise or onto the pulse's echo at double the true rate (its
"second harmonic"). Waiting for 5 seconds is the single most important defence against a
wrong first number.

---

## Part 4 — Isolating the Pulse From Colour: POS (`src/signal_processing/pos.py`)

Here is the core problem green-only measurement cannot solve: the green brightness rises and
falls with the heartbeat, but it *also* rises and falls when a cloud passes the window, when
auto-exposure adjusts, or when the head tilts into shadow. A green-only signal cannot tell
"blood pulse" from "the light changed" — and the light changes are usually far larger.

The way out is **colour**. Picture each moment as a point in a 3-D space whose axes are Red,
Green and Blue.

- A **lighting or motion change** makes all three channels brighter or darker together, so
  it moves the point along the "intensity" direction (roughly equal R, G, B).
- The **heartbeat** changes the channels in a fixed, unequal ratio (strong green, weaker
  red and blue), so it moves the point along a *different* direction.

Because the two effects point in different directions, they can be separated. **POS
(Plane-Orthogonal-to-Skin**, Wang et al., IEEE TBME 2017) projects the RGB signal onto the
plane that is **perpendicular to the skin-tone/intensity direction**. Anything that was pure
intensity change collapses to nothing; what survives is the part of the colour motion that
*isn't* just brightness — the pulse. The code does this over short sliding windows (~1.6
seconds each) and adds the results together, so it keeps adapting as the average skin tone
slowly drifts.

The output is a single one-dimensional **pulse signal** that is far less sensitive to
lighting and motion than green alone. (Green-only extraction is still available as a
fallback via `SignalProcessor(use_pos=False)`, which is useful for comparison experiments.)

---

## Part 5 — Cleaning the Signal (`src/signal_processing/preprocessing.py`, `enhance_heart_rate_signal`)

The pulse signal from POS is good, but still carries some residual slow drift and
high-frequency noise. A useful mental model: any recorded signal can be thought of as a
**mixture of oscillations at different speeds**. Speed is measured in **Hertz (Hz)** —
cycles per second. A resting heartbeat of 60 BPM is 1 beat per second = **1 Hz**. The
humanly possible range of 40–180 BPM is **0.67–3.0 Hz**. Anything much slower or faster
than that cannot be a heartbeat and can be removed.

The cleaning is deliberately short — just three operations, in this order:

**Step 1 — Detrending.** Subtract the overall average level and any slow straight-line
drift. This removes residual slow lighting changes; we never cared about absolute
brightness, only about how it oscillates.

**Step 2 — Band-pass filtering.** A **filter** keeps some oscillation speeds and suppresses
others; a **band-pass** keeps only a chosen band. The code uses a 4th-order **Butterworth**
band-pass (a classic, ripple-free design) from **0.5 to 4.0 Hz**, applied with `filtfilt`,
which runs the filter forward and then backward so it introduces **no time shift**
(**zero-phase** filtering). Note the band (0.5–4.0 Hz) is deliberately *wider* than the
heart-rate search band (0.67–3.0 Hz) used later: keeping the filter's edges outside the
search band means the response is essentially flat across every heart rate we might report,
so a real pulse at 45 or 175 BPM is not accidentally weakened.

**Step 3 — Robust normalization (MAD).** Rescale the signal to a standard range. Instead of
the usual mean and standard deviation (which one big motion spike can distort), the code
uses the **median** (the middle value when sorted — unaffected by extremes) and the **MAD
(Median Absolute Deviation)** — the median distance of points from the median. Each value
becomes `(value − median) / MAD`, and anything beyond ±5 MAD units is clipped, so one
violent jerk cannot corrupt the scaling of the whole signal.

**What the cleaning deliberately does *not* do**, and why:

- **No notch filters.** A notch deletes one narrow frequency. Earlier versions notched
  1, 2 and 3 Hz to remove "flicker" — but 1 Hz *is* 60 BPM, 2 Hz *is* 120 BPM, 3 Hz *is*
  180 BPM, so those notches were deleting real heart rates. And at a 30 fps sampling rate,
  genuine mains flicker (100/120 Hz) folds down ("aliases") to ~10 Hz or DC — nowhere near
  the heart-rate band. The band-pass already removes it.
- **No stacked smoothers.** Repeated smoothing passes act as a low-pass that quietly
  attenuates fast heart rates (a Savitzky-Golay smoother over ~15 samples starts rolling off
  around 2 Hz / 120 BPM). The single band-pass plus the averaging built into the spectrum
  step (Part 6) is enough.

The result is a clean, regular wave — essentially only the heartbeat oscillation remains.

---

## Part 6 — Finding the Heart Rate (`src/signal_processing/hr_estimation.py`, method `estimate`)

### 6.1 From wave to frequency: the idea of a spectrum

We now need to answer: *how fast is this wave oscillating?* The tool is the **Fourier
transform (FFT)**, which reports **how much energy the signal contains at every oscillation
speed**. The output is a **power spectrum**: a graph with frequency (Hz) on the x-axis and
energy on the y-axis. A signal that oscillates strongly at 1.2 Hz shows a tall **peak** at
1.2 Hz.

### 6.2 Welch's method — a more reliable spectrum

Computing one FFT over the whole 10-second buffer is fragile: a single burst of noise can
create a false peak. The code instead uses **Welch's method**: cut the signal into several
overlapping segments (up to 512 samples each), compute each segment's spectrum, and
**average** them. Real oscillations (the heartbeat) appear in every segment and survive;
random noise differs between segments and averages away. The signal is zero-padded to 4096
points (`nfft=4096`) so the frequency axis is fine-grained and the peak position — and thus
the BPM — can be read precisely.

### 6.3 Picking the right peak

The spectrum is cropped to the physiologically possible band **0.67–3.0 Hz (40–180 BPM)**.
Within it, the code finds every peak whose **prominence** (how far it rises above its
surroundings, not just its absolute height) is at least 15% of the maximum, and selects the
**most prominent** one. Its frequency is the heart rate:

```
BPM = peak_frequency_in_Hz × 60
```

Example: the tallest peak sits at 1.24 Hz → 1.24 × 60 = **74.4 BPM**.

### 6.4 How sure are we? The confidence score

Every estimate comes with a **confidence** between 0 and 1:

```
confidence = 0.5 × prominence_score + 0.5 × snr_score
```

- **prominence_score** measures the winning peak's prominence against the average power in
  the band — a sharp, lonely peak scores high.
- **snr_score** compares the peak's height to the band's **median** power (its typical
  "background" level) — the peak clearly towering over the background scores high.

Both are high only when a single sharp peak dominates the band; a flat or messy spectrum
scores low. The estimator's only job is to produce this `(BPM, confidence)` pair — it does
**no** smoothing or trend-following itself (that lives in one place, Part 7).

### 6.5 The warm-up gate: never show a bad first number

A wrong first reading is worse than a slightly delayed one, so before the very first number
is shown, `processor.py` requires several things to line up:

- at least **5 seconds** of signal (Part 3.4);
- **3 consecutive estimates** that each clear a higher confidence bar (≥ 0.6) and agree with
  each other within **10 BPM** — a lone noise/harmonic peak can't sneak through;
- the raw green buffer must be **free of motion-artifact samples** during that window (a
  motion spike can otherwise produce a confident-looking but wrong lock);
- and, as always, any reading below **0.4** confidence or outside **40–180 BPM** is
  discarded.

If a good signal never quite clears the bar, the confidence requirement relaxes after ~17
seconds so a persistently mediocre signal still eventually produces a reading. In practice
the first number appears about 7 seconds in — and it is trustworthy.

---

## Part 7 — Stabilizing the Displayed Number (`src/signal_processing/filtering.py`, `HeartRateFilter`)

Raw per-frame estimates still wobble a little. A **single** stage turns them into a steady
on-screen number:

1. **Median-of-5.** The last five accepted estimates are kept; the running value is based on
   their **median** (middle value). A single bad frame — say one estimate of 130 among
   72, 72, 73, 71 — is simply stepped over by the median. But if the heart rate *genuinely*
   changes, several estimates in a row move, and the median follows within a couple of
   frames. So this rejects lone spikes without blocking real change.

2. **Confidence-weighted smoothing (EMA).** The median then feeds an **exponential moving
   average** whose responsiveness scales with confidence: a confident reading moves the
   displayed number faster (blend factor up to 0.6), a shaky one barely nudges it (down to
   0.15). Confident real changes are tracked promptly; noisy frames don't make the number
   chatter.

That is the whole stabilizer — one median plus one confidence-weighted average, with a
single, understandable behaviour. (Earlier versions stacked a 20-second "baseline anchor",
a separate exponential smoother, a z-score rejector and a change cap; because all of those
updated every frame at 30 fps their real effect was small, and they mainly made the
behaviour hard to reason about. They were replaced by this single stage with no measurable
loss of stability.)

### What the user sees

The final number is displayed and classified:

| Status | Range |
|---|---|
| Bradycardia (slow) | below 60 BPM |
| **Normal** | 60–100 BPM |
| Tachycardia (fast) | above 100 BPM |

Instantaneous readings are saved to the SQLite database (at most once per second); when the
user presses Stop after a long-enough session, a summary with the average BPM is saved too.

---

## Part 8 — The Whole Journey on One Page

```
 A frame arrives from the webcam (≈30 times per second), tagged with a timestamp
   │        — all processing runs on a background worker thread
   ▼
 MediaPipe finds 468 face landmarks
   │
   ▼
 Three skin rectangles: forehead (70%×26% of face), two cheeks (18%×12%)
   — positions smoothed with EMA (60% previous / 40% new)
   │
   ▼
 Average RED, GREEN, BLUE of each rectangle
   │
   ▼
 Stability check per region (grayscale texture) → health score → dynamic weights
   │
   ▼
 One combined RGB value per frame → 10-second buffers (RGB + timestamps)
   │
   ▼
 Measure the true sampling rate from the timestamps → resample to a uniform grid
   │
   ▼
 POS: project RGB onto the plane orthogonal to skin tone → 1-D pulse signal
   (cancels lighting/motion; keeps the heartbeat)
   │
   ▼
 CLEANING: detrend → band-pass 0.5–4.0 Hz (zero-phase) → median/MAD normalize + clip
   │
   ▼
 Welch power spectrum (up to 512-sample segments, averaged, zero-padded to 4096)
   │
   ▼
 Most prominent peak in 0.67–3.0 Hz  →  BPM = peak Hz × 60
   │
   ▼
 Confidence = 0.5 × peak prominence + 0.5 × peak-vs-background   (must exceed 0.4)
 Warm-up gate: ≥5 s of signal, 3 consistent high-confidence estimates, no motion artifact
   │
   ▼
 Stabilize: median-of-5  →  confidence-weighted EMA (faster when confident)
   │
   ▼
 BPM on screen, colour-coded (green / orange / red), saved to SQLite
```

---

## Part 9 — Glossary (every technical term used above)

| Term | Plain meaning |
|---|---|
| **rPPG** | Measuring blood-volume pulses from a camera at a distance, via skin colour changes |
| **Frame / FPS** | One still image of the video / how many images per second (here ≈30) |
| **Sampling rate (fs)** | How many measurements per second we take; here **measured** from timestamps, not assumed |
| **Timestamp** | The recorded time of a frame, used to measure the true sampling rate |
| **Resampling** | Re-spacing samples onto an evenly-spaced time grid before analysis |
| **ROI** | Region of Interest — a rectangle of pixels chosen for analysis |
| **RGB channels** | The red, green and blue components of a pixel's colour |
| **Common-mode change** | A change that affects all colour channels together (e.g. lighting, motion) |
| **POS** | Plane-Orthogonal-to-Skin — combines the RGB channels to keep the pulse and cancel intensity noise |
| **Signal** | A list of numbers ordered in time |
| **Hz (Hertz)** | Cycles per second. 1 Hz oscillation = 60 BPM heartbeat |
| **EMA** | Exponential Moving Average — new value blended with the previous one for smooth motion |
| **Standard deviation** | A measure of how spread out a set of numbers is |
| **Detrending** | Removing the average level and slow straight-line drift from a signal |
| **Filter** | A computation that keeps some oscillation speeds and suppresses others |
| **Band-pass filter** | Keeps only a chosen frequency band (here 0.5–4.0 Hz) |
| **Butterworth** | A classic filter design with a smooth, ripple-free response |
| **filtfilt / zero-phase** | Running a filter forward then backward so it causes no time shift |
| **Median** | The middle value of a sorted list; ignores extreme values |
| **MAD** | Median Absolute Deviation — an outlier-proof version of standard deviation |
| **FFT / Fourier transform** | Math that reports how much of each oscillation speed a signal contains |
| **Power spectrum** | The result of the FFT: energy plotted against frequency |
| **Welch's method** | Averaging spectra of overlapping segments for a noise-resistant spectrum |
| **Peak prominence** | How far a spectral peak rises above its surroundings |
| **Second harmonic** | A signal's echo at twice its true frequency (why 72 BPM can be misread as ~144) |
| **Confidence** | 0–1 score of how trustworthy a single BPM estimate is |
| **Warm-up gate** | The rule that withholds the first reading until it is trustworthy |
| **Bradycardia / Tachycardia** | Medically slow (<60) / fast (>100) heart rate |

---

*Generated from direct inspection of the PulseVision source code:
`gui/processing_worker.py`, `face_detection/mediapipe_detector.py`,
`signal_processing/processor.py`, `signal_processing/pos.py`,
`signal_processing/preprocessing.py`, `signal_processing/hr_estimation.py`,
`signal_processing/filtering.py`, `video/capture.py`.*
