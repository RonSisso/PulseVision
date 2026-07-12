"""Offline accuracy benchmark for the PulseVision rPPG pipeline.

Runs the real SignalProcessor over either synthetic signals with a known BPM
or recorded video clips with a reference BPM, and reports MAE/RMSE/bias per
case. Use it to get a before/after number for every pipeline change.

Usage (from the implementation/ directory):

    python src/benchmark.py --synthetic
    python src/benchmark.py --synthetic --json benchmarks/results/my_run.json --label "step3-no-notch"
    python src/benchmark.py --clips benchmarks/clips.json

Each frame is fed with an explicit timestamp (frame_index / fps), so offline
runs reproduce the live app's timing behavior (init delay, baseline
establishment) deterministically and at full speed.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from signal_processing.processor import SignalProcessor
from signal_processing.evaluation import SyntheticSignalGenerator, mae, rmse, bias

# Synthetic frame geometry: a 160x160 "face" with the same three ROIs the
# live detector produces (forehead + two cheeks).
FRAME_SIZE = 160
SYNTH_ROIS = {
    'forehead': (55, 15, 50, 30),
    'left_cheek': (30, 80, 30, 20),
    'right_cheek': (100, 80, 30, 20),
}
TEXTURE_STD = 6.0    # spatial skin texture so ROIStabilityChecker passes
DEFAULT_DURATION = 35.0
SETTLE_SECONDS = 22.0  # skip the pipeline's convergence window before scoring

# Step-response cases: heart rate jumps mid-recording (e.g. standing up). The
# step is placed well after any baseline-establishment period so it measures
# how fast the smoothing chain tracks a real change.
STEP_DURATION = 50.0
STEP_TIME = 25.0
STEP_SETTLE_TOL = 5.0  # BPM band around the new rate that counts as "settled"
STEP_SUITE = [(72, 108, 'clean'), (72, 108, 'noisy')]

SYNTHETIC_CONDITIONS = {
    'clean': dict(noise_level=0.3, drift_amplitude=0.0),
    'noisy': dict(noise_level=1.5, drift_amplitude=0.0),
    'drift': dict(noise_level=0.3, drift_amplitude=3.0),
    'spikes': dict(noise_level=0.8, spike_rate_per_s=0.5, spike_amplitude=8.0),
    'warmup': dict(noise_level=0.8, settle_amplitude=30.0, settle_tau_s=1.5),
    # Realistic webcam start: pulse with 2nd-harmonic content + AE settling
    'harmonic': dict(noise_level=0.8, harmonic_ratio=0.5,
                     settle_amplitude=30.0, settle_tau_s=1.5),
}

# Note: append new cases at the end only — each case's RNG seed is derived
# from its position, so inserting would silently change existing results.
SYNTHETIC_SUITE = (
    [(bpm, 'clean') for bpm in (45, 60, 72, 90, 120, 150)]
    + [(bpm, 'noisy') for bpm in (45, 60, 72, 90, 120, 150)]
    + [(72, 'drift')]
    + [(72, 'spikes'), (120, 'spikes')]
    + [(72, 'warmup')]
    + [(72, 'harmonic')]
)


def _run_processor(frame_roi_iter, sampling_rate, use_pos=True):
    """Drive a fresh SignalProcessor over (frame, rois) pairs.

    Each frame is stamped with frame_index / sampling_rate. Returns a list of
    (t_seconds, bpm_or_None, confidence) per frame. The processor's per-frame
    prints are suppressed to keep output readable.
    """
    sp = SignalProcessor(sampling_rate=sampling_rate, use_pos=use_pos)
    outputs = []
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        for i, (frame, rois) in enumerate(frame_roi_iter):
            t = i / sampling_rate
            result = sp.process_frame(frame, rois, timestamp=t)
            outputs.append((t, result.bpm, result.confidence))
    return outputs


def _summarize(name, outputs, true_bpm, eval_start):
    """Compute accuracy metrics over the post-convergence window."""
    window = [(t, b) for t, b, _ in outputs if t >= eval_start]
    estimates = [b for _, b in window if b is not None]
    first = next(((t, b) for t, b, _ in outputs if b is not None), None)

    return {
        'case': name,
        'true_bpm': float(true_bpm),
        'mean_est': float(np.mean(estimates)) if estimates else None,
        'mae': mae(estimates, true_bpm) if estimates else None,
        'rmse': rmse(estimates, true_bpm) if estimates else None,
        'bias': bias(estimates, true_bpm) if estimates else None,
        'coverage': (len(estimates) / len(window)) if window else 0.0,
        'first_reading_s': first[0] if first else None,
        'first_reading_bpm': first[1] if first else None,
        'first_reading_error': abs(first[1] - true_bpm) if first else None,
        'n_frames': len(outputs),
        'eval_start_s': eval_start,
    }


# ----------------------------------------------------------------------
# Synthetic mode
# ----------------------------------------------------------------------

def _make_frame(rgb, rng):
    """Build one synthetic uint8 BGR frame from per-channel mean R, G, B values."""
    texture = rng.normal(0.0, TEXTURE_STD, (FRAME_SIZE, FRAME_SIZE)).astype(np.float32)
    r, g, b = rgb
    frame = np.empty((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)
    frame[:, :, 0] = b + texture   # OpenCV channel order is B, G, R
    frame[:, :, 1] = g + texture
    frame[:, :, 2] = r + texture
    return np.clip(frame, 0, 255).astype(np.uint8)


def run_synthetic_case(bpm, condition, duration, sampling_rate=30, seed=0, use_pos=True):
    params = SYNTHETIC_CONDITIONS[condition]
    generator = SyntheticSignalGenerator(
        bpm=bpm, sampling_rate=sampling_rate, amplitude=1.5, seed=seed, **params
    )
    trace = generator.generate_rgb(duration)  # (N, 3)
    texture_rng = np.random.default_rng(seed + 1)

    def frames():
        for rgb in trace:
            yield _make_frame(rgb, texture_rng), SYNTH_ROIS

    outputs = _run_processor(frames(), sampling_rate, use_pos=use_pos)
    name = f"synthetic {bpm:>3d} BPM {condition}"
    return _summarize(name, outputs, bpm, eval_start=min(SETTLE_SECONDS, duration * 0.6))


def run_synthetic_suite(duration, use_pos=True):
    results = []
    for i, (bpm, condition) in enumerate(SYNTHETIC_SUITE):
        print(f"  running {bpm} BPM / {condition} ...", flush=True)
        results.append(run_synthetic_case(bpm, condition, duration, seed=100 + i, use_pos=use_pos))
    return results


def run_step_case(pre_bpm, post_bpm, condition, sampling_rate=30, seed=0, use_pos=True):
    """Heart rate steps pre_bpm -> post_bpm at STEP_TIME; measure tracking."""
    params = SYNTHETIC_CONDITIONS[condition]
    generator = SyntheticSignalGenerator(
        bpm=pre_bpm, sampling_rate=sampling_rate, amplitude=1.5, seed=seed,
        bpm_end=post_bpm, step_time_s=STEP_TIME, **params
    )
    trace = generator.generate_rgb(STEP_DURATION)
    texture_rng = np.random.default_rng(seed + 1)

    def frames():
        for rgb in trace:
            yield _make_frame(rgb, texture_rng), SYNTH_ROIS

    outputs = _run_processor(frames(), sampling_rate, use_pos=use_pos)

    # Settling time: seconds after the step until the estimate enters the
    # tolerance band around the new rate and never leaves it again.
    post = [(t, b) for t, b, _ in outputs if b is not None and t >= STEP_TIME]
    settling = None
    for i, (t, _) in enumerate(post):
        if all(abs(b - post_bpm) <= STEP_SETTLE_TOL for _, b in post[i:]):
            settling = round(t - STEP_TIME, 1)
            break

    # Steady-state accuracy over the final 8 s (well after the step)
    final = [b for t, b in post if t >= STEP_DURATION - 8]
    return {
        'case': f"step {pre_bpm}->{post_bpm} {condition}",
        'true_bpm': float(post_bpm),
        'mean_est': float(np.mean(final)) if final else None,
        'mae': mae(final, post_bpm) if final else None,
        'rmse': rmse(final, post_bpm) if final else None,
        'bias': bias(final, post_bpm) if final else None,
        'coverage': 1.0,
        'first_reading_s': None,
        'first_reading_bpm': None,
        'first_reading_error': None,
        'settling_s': settling,
        'n_frames': len(outputs),
        'eval_start_s': STEP_TIME,
    }


def run_step_suite(use_pos=True):
    results = []
    for i, (pre, post, condition) in enumerate(STEP_SUITE):
        print(f"  running step {pre}->{post} / {condition} ...", flush=True)
        results.append(run_step_case(pre, post, condition, seed=300 + i, use_pos=use_pos))
    return results


# ----------------------------------------------------------------------
# Video clip mode
# ----------------------------------------------------------------------

def run_clip(path, true_bpm, name=None, use_pos=True):
    import cv2
    from face_detection.mediapipe_detector import FaceDetector

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        print(f"  WARNING: unreliable FPS ({fps}) for {path}, assuming 30")
        fps = 30.0

    detector = FaceDetector()

    def frames():
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            landmarks = detector.detect_face(frame)
            rois = detector.get_all_rois(frame, landmarks) if landmarks is not None else None
            yield frame, rois

    outputs = _run_processor(frames(), sampling_rate=int(round(fps)), use_pos=use_pos)
    cap.release()

    if not outputs:
        raise IOError(f"No frames decoded from {path}")

    duration = outputs[-1][0]
    eval_start = SETTLE_SECONDS if duration >= 32 else duration * 0.5
    case_name = name or os.path.basename(path)
    summary = _summarize(f"clip {case_name}", outputs, true_bpm, eval_start)
    summary['fps'] = fps
    summary['duration_s'] = round(duration, 1)
    return summary


def run_clip_suite(manifest_path, use_pos=True):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    results = []
    for entry in manifest['clips']:
        path = entry['path']
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        print(f"  running {entry['path']} (true {entry['true_bpm']} BPM) ...", flush=True)
        results.append(run_clip(path, entry['true_bpm'], name=entry.get('label'), use_pos=use_pos))
    return results


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def _fmt(value, spec):
    return format(value, spec) if value is not None else '   --'

def print_report(results):
    header = (f"{'Case':<28} {'True':>6} {'Mean est':>9} {'MAE':>7} "
              f"{'Bias':>7} {'Cover':>6} {'First':>6} {'1stErr':>7} {'Settle':>7}")
    print()
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['case']:<28} {r['true_bpm']:>6.1f} {_fmt(r['mean_est'], '9.1f')} "
              f"{_fmt(r['mae'], '7.2f')} {_fmt(r['bias'], '+7.2f')} "
              f"{r['coverage']:>5.0%} {_fmt(r['first_reading_s'], '6.1f')} "
              f"{_fmt(r.get('first_reading_error'), '7.2f')} "
              f"{_fmt(r.get('settling_s'), '7.1f')}")

    scored = [r for r in results if r['mae'] is not None]
    failed = [r for r in results if r['mae'] is None]
    print('-' * len(header))
    if scored:
        overall = float(np.mean([r['mae'] for r in scored]))
        good = sum(1 for r in scored if r['mae'] <= 3.0)
        print(f"Overall MAE: {overall:.2f} BPM over {len(scored)} cases "
              f"| within 3 BPM: {good}/{len(results)}")
    if failed:
        print(f"NO OUTPUT in eval window: {len(failed)} case(s): "
              + ', '.join(r['case'] for r in failed))
    print()


def git_revision():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=SRC_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return 'unknown'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--synthetic', action='store_true',
                        help='run the built-in synthetic suite (no videos needed)')
    parser.add_argument('--clips', metavar='MANIFEST',
                        help='run recorded clips listed in a JSON manifest')
    parser.add_argument('--json', metavar='PATH', help='write results to a JSON file')
    parser.add_argument('--label', default=None, help='label stored with the results')
    parser.add_argument('--method', choices=('pos', 'green'), default='pos',
                        help='pulse extraction method (default: pos)')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION,
                        help='synthetic case duration in seconds (default %(default)s)')
    args = parser.parse_args()

    if not args.synthetic and not args.clips:
        parser.error('choose at least one of --synthetic / --clips')

    use_pos = (args.method == 'pos')
    results = []
    if args.synthetic:
        print(f"Synthetic suite ({len(SYNTHETIC_SUITE)} steady + {len(STEP_SUITE)} step "
              f"cases, fs=30, method={args.method}) ...")
        results.extend(run_synthetic_suite(args.duration, use_pos=use_pos))
        results.extend(run_step_suite(use_pos=use_pos))
    if args.clips:
        print(f"Clip suite from {args.clips} (method={args.method}) ...")
        results.extend(run_clip_suite(args.clips, use_pos=use_pos))

    print_report(results)

    if args.json:
        payload = {
            'label': args.label or datetime.now().strftime('%Y-%m-%d %H:%M'),
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'git_revision': git_revision(),
            'method': args.method,
            'results': results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"Results written to {args.json}")


if __name__ == '__main__':
    main()
