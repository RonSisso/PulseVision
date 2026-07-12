"""Tests for the single-stage HeartRateFilter (median + confidence-weighted EMA)."""

from signal_processing.filtering import HeartRateFilter


def test_median_rejects_lone_spike():
    f = HeartRateFilter()
    for _ in range(4):
        f.update(72.0, 0.9)
    out = f.update(150.0, 0.9)  # single in-range outlier
    assert out < 90.0  # median of [72,72,72,72,150] steps over the spike


def test_ema_tracks_sustained_step():
    f = HeartRateFilter()
    for _ in range(10):
        f.update(72.0, 0.9)
    for _ in range(30):
        out = f.update(108.0, 0.9)
    assert out > 104.0  # confident sustained change is tracked


def test_high_confidence_tracks_faster_than_low():
    fast, slow = HeartRateFilter(), HeartRateFilter()
    fast.update(72.0, 0.9)
    slow.update(72.0, 0.1)
    fast_out = fast.update(100.0, 1.0)
    slow_out = slow.update(100.0, 0.0)
    assert fast_out > slow_out  # higher confidence moves toward the new value faster


def test_out_of_range_rejected():
    f = HeartRateFilter()
    for _ in range(3):
        f.update(72.0, 0.9)
    out = f.update(300.0, 0.9)  # non-physiological
    assert abs(out - 72.0) < 1.0  # unchanged; the bad value is ignored


def test_none_before_first_update():
    f = HeartRateFilter()
    assert f.update(None, 0.5) is None


def test_reset_clears_state():
    f = HeartRateFilter()
    for _ in range(5):
        f.update(72.0, 0.9)
    f.reset()
    assert f.smoothed is None
    assert len(f.recent) == 0
