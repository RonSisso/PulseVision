import cv2
import numpy as np
import os
import logging
import time

class VideoCapture:
    def __init__(self):
        """Initialize video capture."""
        # Set up logging
        self.logger = logging.getLogger('PulseVision.VideoCapture')
        
        self.cap = None
        self.source = 0  # Default to first webcam
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 0
        self.resolution = (0, 0)
        self.last_frame_time = None
        self.min_frame_interval = 1.0 / 60  # Maximum 60 FPS
        self.logger.info("VideoCapture initialized")

    def start(self, source=None):
        """Start video capture from specified source."""
        if source is not None:
            self.source = source
            
        # Stop any existing capture
        self.stop()
        
        self.logger.info("Attempting to open video source: %s", self.source)
        
        # Handle file paths
        if isinstance(self.source, str):
            if not os.path.exists(self.source):
                self.logger.error("Video file not found: %s", self.source)
                raise FileNotFoundError(f"Video file not found: {self.source}")
        
        try:
            # For webcam, try different backends
            if isinstance(self.source, int):
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                for backend in backends:
                    self.logger.info(f"Trying camera backend: {backend}")
                    self.cap = cv2.VideoCapture(self.source + backend)
                    if self.cap.isOpened():
                        self.logger.info(f"Successfully opened camera with backend: {backend}")
                        break
            else:
                # For video files, use default backend
                self.cap = cv2.VideoCapture(str(self.source))
            
            if not self.cap.isOpened():
                if isinstance(self.source, int):
                    self.logger.error("Failed to open webcam %d", self.source)
                    return False
                else:
                    self.logger.error("Failed to open video file: %s", self.source)
                    return False
            
            # Get and log video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution = (width, height)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info("Video properties - FPS: %.2f, Resolution: %dx%d", self.fps, width, height)
            
            if self.total_frames > 0:
                self.logger.info("Total frames: %d", self.total_frames)
            
            # Read test frame to verify capture is working
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.logger.error("Failed to read test frame")
                self.cap.release()
                self.cap = None
                return False
            
            # Reset frame counter and timing
            self.frame_count = 0
            self.last_frame_time = None
            
            self.logger.info("Video capture initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error("Error initializing video capture: %s", str(e))
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

    def stop(self):
        """Stop video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.logger.info("Video capture stopped after %d frames", self.frame_count)
            self.frame_count = 0
            self.last_frame_time = None

    def read_frame(self):
        """Read a frame from the video source with frame rate control."""
        if self.cap is None or not self.cap.isOpened():
            self.logger.warning("Attempting to read from invalid capture")
            return None
        
        try:
            # Frame rate control
            current_time = time.time()
            if self.last_frame_time is not None:
                elapsed = current_time - self.last_frame_time
                if elapsed < self.min_frame_interval:
                    time.sleep(self.min_frame_interval - elapsed)
            
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                self.logger.warning(f"Failed to read frame {self.frame_count + 1}")
                if isinstance(self.source, str):
                    # For video files, we've reached the end
                    self.logger.info("End of video file reached")
                    return None
                else:
                    # For webcam, try to recover
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.source)
                    if not self.cap.isOpened():
                        return None
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        return None
            
            self.frame_count += 1
            self.last_frame_time = current_time
            
            # Verify frame integrity
            if frame.size == 0 or not np.all(np.isfinite(frame)):
                self.logger.warning("Invalid frame detected")
                return None
                
            self.logger.debug("Read frame %d", self.frame_count)
            return frame
            
        except Exception as e:
            self.logger.error("Error reading frame: %s", str(e))
            return None

    def is_opened(self):
        """Check if video capture is opened."""
        try:
            is_open = self.cap is not None and self.cap.isOpened()
            if not is_open:
                self.logger.warning("Video capture is not open")
            return is_open
        except Exception as e:
            self.logger.error("Error checking video capture state: %s", str(e))
            return False

    def release(self):
        """Release video capture resources."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.logger.info(f"Video capture stopped after {self.frame_count} frames")
            self.cap = None
            
        except Exception as e:
            self.logger.error(f"Error releasing video capture: {str(e)}")

    def __del__(self):
        """Cleanup resources."""
        self.stop()

    def get_fps(self):
        """Get video FPS."""
        return self.fps

    def get_resolution(self):
        """Get video resolution."""
        return self.resolution

    def get_frame_count(self):
        """Get current frame count."""
        return self.frame_count

    def get_total_frames(self):
        """Get total number of frames (0 for live feed)."""
        return self.total_frames