"""Background measurement worker.

Runs the capture -> face detection -> signal processing loop in its own
thread so the GUI thread only renders. The worker owns the video source,
detector, and processor for the lifetime of one measurement run; every run
starts with fresh state.

Timestamps: video files use video time (frame_index / fps) so playback speed
never affects the measured heart rate; cameras use wall-clock time, which
captures the real (possibly jittery) sampling rate.
"""

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from video.capture import VideoCapture
from face_detection.mediapipe_detector import FaceDetector
from signal_processing.processor import SignalProcessor

ROI_COLORS = {
    'forehead': (0, 255, 0),    # Green
    'left_cheek': (255, 0, 0),  # Blue
    'right_cheek': (0, 0, 255)  # Red
}


@dataclass
class MeasurementUpdate:
    """One frame's worth of measurement data for the GUI."""
    t: float                    # seconds since measurement start (sample time)
    raw_value: float | None     # mean green level across ROIs (signal plot)
    roi_found: bool
    roi_lost: bool              # face was lost since the last update
    bpm: float | None
    confidence: float
    fft_freqs: object = None
    fft_power: object = None
    method: str = 'none'


class ProcessingWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)   # annotated BGR frame for display
    update_ready = pyqtSignal(object)      # MeasurementUpdate
    stopped = pyqtSignal(str)              # 'end_of_stream' | 'requested' | 'error: ...'

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source
        self._stop_requested = False
        self.logger = logging.getLogger(__name__)

    def request_stop(self):
        """Ask the worker loop to exit; safe to call from any thread."""
        self._stop_requested = True

    def run(self):
        reason = 'end_of_stream'
        capture = VideoCapture()
        try:
            if not capture.start(self.source):
                self.stopped.emit('error: failed to open video source')
                return

            detector = FaceDetector()
            processor = SignalProcessor()

            is_file = isinstance(self.source, str)
            fps = capture.get_fps()
            if not fps or fps <= 0 or fps > 120:
                fps = 30.0
            # Pace file playback to real time; cameras pace themselves.
            frame_interval = (1.0 / fps) if is_file else None

            frame_idx = 0
            wall_start = time.time()

            while not self._stop_requested:
                loop_start = time.time()
                frame = capture.read_frame()
                if frame is None:
                    break

                timestamp = (frame_idx / fps) if is_file else (time.time() - wall_start)
                frame_idx += 1

                landmarks = detector.detect_face(frame)
                rois = detector.get_all_rois(frame, landmarks) if landmarks is not None else None
                roi_found = rois is not None and any(rois.values())

                result = processor.process_frame(frame, rois if roi_found else None,
                                                 timestamp=timestamp)
                roi_lost = processor.was_roi_lost()

                raw_value = None
                if roi_found:
                    raw_value = self._mean_roi_green(frame, rois)
                    self._draw_rois(frame, rois)

                self.frame_ready.emit(frame)
                self.update_ready.emit(MeasurementUpdate(
                    t=timestamp,
                    raw_value=raw_value,
                    roi_found=roi_found,
                    roi_lost=roi_lost,
                    bpm=result.bpm,
                    confidence=result.confidence,
                    fft_freqs=result.fft_freqs,
                    fft_power=result.fft_power,
                    method=result.method,
                ))

                if frame_interval is not None:
                    remaining = frame_interval - (time.time() - loop_start)
                    if remaining > 0:
                        time.sleep(remaining)

            if self._stop_requested:
                reason = 'requested'

        except Exception as e:
            self.logger.exception("Processing worker crashed")
            reason = f'error: {e}'
        finally:
            capture.stop()

        self.stopped.emit(reason)

    @staticmethod
    def _mean_roi_green(frame, rois):
        """Plain average green level over all valid ROIs (for the raw plot)."""
        greens = []
        for roi in rois.values():
            if roi is None:
                continue
            x, y, w, h = roi
            greens.append(np.mean(frame[y:y + h, x:x + w, 1]))
        return float(np.mean(greens)) if greens else None

    @staticmethod
    def _draw_rois(frame, rois):
        for roi_name, roi in rois.items():
            if roi is None:
                continue
            x, y, w, h = roi
            color = ROI_COLORS.get(roi_name, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, roi_name.replace('_', ' ').title(),
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
