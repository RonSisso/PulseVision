# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working directory

All development happens in `implementation/` (this directory): `src/`, `tests/`,
`pyproject.toml`, `requirements*.txt`, and the `venv/` all live here. Run every command
below from `implementation/`.

The **git root is the parent** (`PulseVision/`), which also holds the capstone report
binaries (`.docx/.pdf/.pptx`) and `.github/workflows/ci.yml`. Do **not** `git add -A` from
the git root — it sweeps in untracked editor folders (`.idea/`) and large binaries. Stage
explicit paths under `implementation/` instead. CRLF warnings from git on Windows are
expected and harmless.

## Commands

The interpreter is the project venv (`venv/Scripts/python.exe` on Windows, `venv/bin/python`
elsewhere); the examples assume it is activated.

```bash
# Install
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # ruff, black, pytest

# Run the app (PyQt5 GUI; opens a login window, default user admin / admin123)
python src/main.py

# Accuracy benchmark — the primary tool for validating any signal-processing change
python src/benchmark.py --synthetic                         # built-in synthetic suite
python src/benchmark.py --synthetic --method green          # A/B: green-only vs POS (default pos)
python src/benchmark.py --synthetic --json benchmarks/results/x.json --label "my change"
python src/benchmark.py --clips benchmarks/clips.json       # real recorded reference clips

# Tests
pytest                                   # full suite (~7 s, no camera/MediaPipe needed)
pytest tests/test_pos.py                 # one file
pytest tests/test_filtering.py::test_ema_tracks_sustained_step   # one test
pytest -k pos                            # by keyword

# Lint / format (CI enforces the first two)
ruff check .
black --check .                          # drop --check to auto-format
```

Lint/format are **scoped** (via `pyproject.toml`) to `src/signal_processing/`,
`src/benchmark.py`, and `tests/`. The GUI and database layers are intentionally excluded;
don't reformat them as a side effect. CI (`.github/workflows/ci.yml`) runs ruff + black +
pytest on push/PR using only numpy/scipy/opencv-headless — never the GUI/MediaPipe stack.

Tests rely on `pythonpath = ["src"]` (pyproject) plus `conftest.py`, so they import the
packages directly, e.g. `from signal_processing.pos import pos_pulse`.

## Architecture

PulseVision measures heart rate from webcam video via remote photoplethysmography (rPPG).
The end-to-end signal path spans several files and is the thing worth understanding first:

```
video/capture.py            → raw frames (+ timestamps)
face_detection/
  mediapipe_detector.py     → 468 landmarks → 3 ROIs (forehead + 2 cheeks), EMA-smoothed
signal_processing/
  processor.py              → THE ORCHESTRATOR (per-frame entry point)
    · per-ROI mean R,G,B → stability-weighted combine (filtering.ROIStabilityChecker)
    · measure the true sampling rate from timestamps, resample to a uniform grid
    · pos.py                → POS colour projection → 1-D pulse (green-only is a fallback)
    · preprocessing.py      → detrend → one Butterworth band-pass 0.5–4.0 Hz → median/MAD
    · hr_estimation.py      → Welch PSD peak-pick → (bpm, confidence)
    · warm-up gate (in processor.py) withholds the first reading until it is trustworthy
    · filtering.HeartRateFilter → median-of-5 + confidence-weighted EMA (the ONLY smoother)
```

`SignalProcessor.process_frame(frame, rois, timestamp)` is the single per-frame entry point
and returns a `ProcessorResult` dataclass (`bpm, confidence, fft_freqs, fft_power, method`).
It owns the rolling buffers, ROI health/weighting, the warm-up gate, and calls the
sub-modules above in order. Read `process_multiple_rois` to see the whole flow.

**Threading (GUI):** `gui/processing_worker.py` (`ProcessingWorker`, a `QThread`) owns the
capture + detector + processor for one measurement run and emits `frame_ready(frame)` and
`update_ready(MeasurementUpdate)` signals. `gui/main_window.py` only renders — it must never
run capture/detection/DSP on the GUI thread. Plots redraw on a timer; instantaneous DB
writes are throttled to ~1/s through a single shared `Database` connection.

**Benchmark-driven methodology:** `benchmark.py` + `signal_processing/evaluation.py` drive
the *real* `SignalProcessor` over synthetic frames (or recorded clips), feeding explicit
per-sample timestamps, and report MAE/RMSE/bias per case. This is how the pipeline is
validated — treat it as the test bench for the DSP. `benchmarks/results/*.json` are the saved
before/after history of prior changes.

**GUI/data flow:** `main.py` → `LoginWindow` → `HomeWindow` → `MainWindow` (measurement).
`database/db.py` is SQLite (users, patients, per-reading `measurements`, and
`measurement_sessions` averages); it creates a default `admin`/`admin123` user on first run.

## Design invariants (do not silently undo these)

These decisions were made deliberately and are validated by the benchmark. Changing them
without re-running `python src/benchmark.py --synthetic` and comparing is how regressions
get in:

- **Sampling rate is measured, never assumed.** Each sample is timestamped; the true `fs` is
  computed and the signal resampled. Don't hard-code 30 fps into new DSP.
- **No notch filters.** At ~30 fps, mains flicker aliases outside the heart-rate band, while
  notches at 1–3 Hz would sit exactly on real heart rates (60/120/180 BPM) and delete them.
- **Preprocessing stays minimal** (detrend → one band-pass → MAD normalize). Stacking more
  band-passes or smoothers attenuates real heart rates near the band edges.
- **POS is the default extraction**, green-only is the `use_pos=False` fallback. POS combines
  RGB to cancel lighting/motion; keep the fallback for A/B comparison.
- **There is exactly one smoothing stage** (`HeartRateFilter`: median + confidence-weighted
  EMA). Do not reintroduce the old baseline-anchor / z-score / change-cap layers — they added
  stickiness and a confidence-interaction bug without improving stability.
- **The warm-up gate withholds the first reading (~7 s) by design.** It prevents the startup
  harmonic-lock that showed ~140 BPM in the first seconds. A delayed-but-correct first number
  is intended behaviour, not a bug to "speed up".
- **`SYNTHETIC_SUITE`/`STEP_SUITE` cases are append-only** — each case's RNG seed is derived
  from its position, so inserting or reordering silently changes historical results.
