"""Tests for the benchmark metrics and synthetic signal generator."""

import numpy as np
from scipy.signal import detrend

from signal_processing.evaluation import (
    SyntheticSignalGenerator,
    bias,
    mae,
    rmse,
)


def test_metric_values():
    est = [70.0, 74.0]  # errors of -2 and +2 around 72
    assert mae(est, 72.0) == 2.0
    assert rmse(est, 72.0) == 2.0
    assert bias(est, 72.0) == 0.0


def test_metrics_handle_empty():
    assert np.isnan(mae([], 72.0))


def test_generate_rgb_shape_and_channel_structure():
    gen = SyntheticSignalGenerator(bpm=72, sampling_rate=30, amplitude=1.5, noise_level=0.0, seed=1)
    rgb = gen.generate_rgb(10)
    assert rgb.shape == (300, 3)
    # Pulse is strongest in green: green AC amplitude > red > blue.
    ac = [np.std(detrend(rgb[:, c])) for c in range(3)]
    assert ac[1] > ac[0] > ac[2]


def test_generate_rgb_frequency_step():
    gen = SyntheticSignalGenerator(
        bpm=72, bpm_end=108, step_time_s=10, sampling_rate=30, noise_level=0.0, seed=2
    )
    rgb = gen.generate_rgb(20)
    assert rgb.shape == (600, 3)
    assert np.all(np.isfinite(rgb))
