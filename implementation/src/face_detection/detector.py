import cv2
import numpy as np
import logging

class FaceDetector:
    def __init__(self, debug=False):
        """Initialize face detector."""
        self.logger = logging.getLogger('PulseVision.FaceDetector')
        self.debug = debug

        # Load pre-trained Haar cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # ROI tracking settings
        self.prev_roi = None
        self.roi_padding = 0.3
        self.detection_interval = 30
        self.frame_count = 0
        self.track_quality = 0.7

        # Tracker
        self.tracker = None
        self.tracker_initialized = False
        self.init_tracker()

        self.logger.info("Face detector initialized with ROI padding: %.2f", self.roi_padding)

    def init_tracker(self):
        """Initialize tracker with fallback options."""
        tracker_types = ['CSRT', 'KCF', 'MOSSE', 'MIL', 'BOOSTING']
        for tracker_type in tracker_types:
            try:
                if hasattr(cv2, 'legacy'):
                    self.tracker = getattr(cv2.legacy, f'Tracker{tracker_type}_create')()
                else:
                    self.tracker = getattr(cv2, f'Tracker{tracker_type}_create')()
                self.logger.info(f"Initialized {tracker_type} tracker")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize {tracker_type} tracker: {str(e)}")

        self.logger.warning("All trackers failed. Falling back to optical flow only.")

    def detect_face(self, frame):
        if frame is None:
            return None

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                self.tracker_initialized = False
                return largest_face
        except Exception as e:
            self.logger.error("Face detection error: %s", str(e))

        return None

    def get_forehead_roi(self, frame, face):
        if frame is None or face is None:
            return None

        try:
            x, y, w, h = face
            forehead_height = int(h * 0.33)
            padding_x = int(w * self.roi_padding)
            padding_y = int(forehead_height * self.roi_padding)

            roi_x = max(0, x - padding_x)
            roi_y = max(0, y - padding_y)
            roi_w = min(w + 2 * padding_x, frame.shape[1] - roi_x)
            roi_h = min(forehead_height + 2 * padding_y, frame.shape[0] - roi_y)

            if roi_w <= 0 or roi_h <= 0:
                return None

            return (roi_x, roi_y, roi_w, roi_h)

        except Exception as e:
            self.logger.error("Forehead ROI extraction error: %s", str(e))
            return None

    def track_roi(self, frame, prev_frame, prev_roi):
        if frame is None or prev_frame is None or prev_roi is None:
            return None

        try:
            self.frame_count += 1

            # Detect face every detection_interval
            if self.frame_count == 1 or self.frame_count % self.detection_interval == 0:
                detected_face = self.detect_face(frame)
                if detected_face is not None:
                    roi = self.get_forehead_roi(frame, detected_face)
                    self.init_tracker()
                    if self.tracker and roi:
                        self.tracker.init(frame, tuple(roi))
                        self.tracker_initialized = True
                        return True, roi
                return None

            if self.tracker and not self.tracker_initialized:
                self.tracker.init(frame, tuple(prev_roi))
                self.tracker_initialized = True

            if self.tracker and self.tracker_initialized:
                success, bbox = self.tracker.update(frame)
                if success:
                    roi = [int(v) for v in bbox]
                    if self.debug:
                        x, y, w, h = roi
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    return True, roi

            # Optical flow fallback
            x, y, w, h = prev_roi
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            points = cv2.goodFeaturesToTrack(prev_gray[y:y+h, x:x+w], maxCorners=20, qualityLevel=0.01, minDistance=5)
            if points is not None:
                points = points + [x, y]
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None)

                if status.sum() > len(status) * self.track_quality:
                    movement = new_points[status == 1] - points[status == 1]
                    dx = int(np.median(movement[:, 0]))
                    dy = int(np.median(movement[:, 1]))
                    new_x = max(0, min(frame.shape[1] - w, x + dx))
                    new_y = max(0, min(frame.shape[0] - h, y + dy))
                    return True, [new_x, new_y, w, h]

        except Exception as e:
            self.logger.error("ROI tracking error: %s", str(e))

        return None

    def reset(self):
        """Reset the face detector state for a fresh measurement."""
        self.prev_roi = None
        self.frame_count = 0
        self.tracker = None
        self.tracker_initialized = False
        self.init_tracker()
        self.logger.debug("Haar Cascade face detector reset")
