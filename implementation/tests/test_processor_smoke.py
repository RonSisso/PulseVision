"""End-to-end smoke test: synthetic frames -> SignalProcessor -> heart rate.

Exercises the full DSP path (RGB extraction, ROI combination, timestamp
resampling, POS, preprocessing, estimation, warm-up gate, smoothing) without a
camera or MediaPipe, by feeding generated frames straight into the processor.
"""

from benchmark import run_synthetic_case


def test_processor_estimates_known_bpm_pos():
    result = run_synthetic_case(72, "clean", duration=16, seed=0, use_pos=True)
    assert result["mae"] is not None, "no heart rate was reported"
    assert result["mae"] < 3.0
    assert result["coverage"] > 0.5


def test_processor_estimates_known_bpm_green_fallback():
    result = run_synthetic_case(72, "clean", duration=16, seed=0, use_pos=False)
    assert result["mae"] is not None
    assert result["mae"] < 3.0
