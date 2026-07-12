# PulseVision Accuracy Benchmarks

Offline evaluation of the rPPG pipeline. Every signal-processing change should
be justified by a before/after run of this benchmark.

## Synthetic suite (no recordings needed)

Runs the real `SignalProcessor` over generated signals with a known BPM
(45–150 BPM, clean / noisy / drifting-light conditions):

```bash
python src/benchmark.py --synthetic
python src/benchmark.py --synthetic --json benchmarks/results/my_change.json --label "my change"
```

Compare the `mae` fields of two results JSONs to see whether a change helped.

## Real clip suite

1. Record 45–60 second face videos (webcam or phone, ~30 fps, steady even
   lighting, sit still) while wearing a reference device (e.g. Apple Watch).
   Note the reference BPM shown mid-recording.
2. Put the clips in `benchmarks/clips/` (video files are git-ignored).
3. Describe them in a manifest (see `clips.example.json`):

```json
{
  "clips": [
    { "path": "clips/rest_daylight.mp4", "true_bpm": 76, "label": "rest, daylight" }
  ]
}
```

4. Run:

```bash
python src/benchmark.py --clips benchmarks/clips.json
```

Clip paths are resolved relative to the manifest file. The clip's own FPS is
read from the file and used as the pipeline sampling rate.

## Metrics

- **MAE / RMSE / bias** — computed over the converged window (the first ~22 s are
  skipped to let the pipeline settle; for shorter clips, the second half).
- **Cover** — fraction of frames in that window with a valid BPM output.
- **First** — seconds until the first valid reading.
- **1stErr** — error of the first reported reading (startup reliability).
- **Settle** — for step-response cases, seconds after the rate change until the
  estimate reaches and stays within tolerance of the new rate.

The processor consumes explicit per-sample timestamps (frame_index / fps), so
its timing behavior (2 s init delay, 20 s baseline) is deterministic and runs
faster than real time while behaving exactly like the live app. The processor
also measures the true sampling rate from those timestamps, so clips with any
frame rate are evaluated correctly.
