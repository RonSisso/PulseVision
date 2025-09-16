import os
import json
import numpy as np

class OfflineEvaluator:
    def __init__(self, truth_path):
        self.truth_path = truth_path
        self.ground_truth = self._load_truth()

    def _load_truth(self):
        if not os.path.exists(self.truth_path):
            return {}
        with open(self.truth_path, 'r') as f:
            return json.load(f)

    def evaluate(self, frame_idx, predicted_bpm):
        true_bpm = self.ground_truth.get(str(frame_idx))
        if true_bpm is None:
            return None
        error = abs(predicted_bpm - true_bpm)
        return error

class SyntheticSignalGenerator:
    def __init__(self, bpm=75, noise_level=0.05, sampling_rate=30):
        self.bpm = bpm
        self.noise = noise_level
        self.fs = sampling_rate

    def generate(self, duration_sec):
        t = np.linspace(0, duration_sec, int(self.fs * duration_sec))
        signal = np.sin(2 * np.pi * (self.bpm / 60) * t)
        noise = np.random.normal(0, self.noise, size=signal.shape)
        return signal + noise