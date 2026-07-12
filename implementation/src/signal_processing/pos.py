"""POS (Plane-Orthogonal-to-Skin) rPPG extraction.

Reference: Wang, den Brinker, Stuijk & de Haan, "Algorithmic Principles of
Remote PPG", IEEE Trans. Biomedical Engineering, 2017.

Green-channel-only extraction throws away the colour information that
distinguishes a real pulse from intensity changes (lighting, motion): those
move all three channels together, while the pulse has a specific R:G:B
signature. POS projects the RGB trace onto a plane orthogonal to the skin-tone
direction, cancelling the common-mode intensity component and keeping the
pulse. It runs as an overlap-add sliding window so it adapts to slow skin-tone
drift over a measurement.
"""

import numpy as np

# Projection matrix from the paper (channel order R, G, B):
#   S1 =        G - B
#   S2 = -2R +  G + B
_P = np.array([[0.0, 1.0, -1.0],
               [-2.0, 1.0, 1.0]])


def pos_pulse(rgb, fs, window_seconds=1.6):
    """Return a 1-D pulse signal from an (N, 3) RGB trace (columns R, G, B).

    `window_seconds` is the sliding-window length; 1.6 s is the paper's value
    and comfortably spans one cardiac cycle down to ~40 BPM.
    """
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must be an (N, 3) array in R, G, B order")

    n = rgb.shape[0]
    length = int(round(window_seconds * fs))

    # Not enough data for a sliding window: one global projection.
    if length < 2 or n < length:
        return _project(rgb)

    # Overlap-add the per-window projections (paper's Algorithm 1).
    pulse = np.zeros(n)
    for start in range(0, n - length + 1):
        pulse[start:start + length] += _project(rgb[start:start + length])
    return pulse


def _project(window):
    """POS projection of one window: temporal-normalize, project, tune, center."""
    mean = np.mean(window, axis=0)
    mean = np.where(np.abs(mean) < 1e-8, 1.0, mean)   # guard flat channels
    normalized = window / mean                        # temporal normalization

    projected = normalized @ _P.T                     # (L, 2)
    s1, s2 = projected[:, 0], projected[:, 1]

    # Tune the two projections so their combination cancels intensity noise.
    std2 = np.std(s2)
    alpha = (np.std(s1) / std2) if std2 > 1e-8 else 0.0
    h = s1 + alpha * s2

    return h - np.mean(h)                             # zero-mean for overlap-add
