import os
import cv2
import numpy as np
import time
import logging
import signal
from collections import deque
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .base_window import BaseWindow
from signal_processing.processor import SignalProcessor
from face_detection.mediapipe_detector import FaceDetector
from video.capture import VideoCapture


class MainWindow(BaseWindow):
    def __init__(self):
        # Initialize with back button but no power off button
        super().__init__(show_back_button=True, show_power_off=False)
        
        # Initialize state flags
        self.is_running = False
        
        # Window size for signal display (in seconds)
        self.window_duration = 10
        
        # Signal quality indicators
        self.quality_threshold = 0.005
        self.min_samples_for_quality = 30
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Data buffers for plotting
        max_samples = int(self.window_duration * 30)  # 30 fps * 10 seconds
        self.signal_times = deque(maxlen=max_samples)
        self.signal_values = deque(maxlen=max_samples)
        self.raw_values = deque(maxlen=max_samples)
        self.hr_times = deque(maxlen=max_samples)
        self.hr_values = deque(maxlen=max_samples)
        self.fft_freqs = deque(maxlen=512)  # FFT frequency bins
        self.fft_power = deque(maxlen=512)  # FFT power spectrum
        self.cardiac_cycle_data = deque(maxlen=100)  # Cardiac cycle data
        self.start_time = None
        
        # Initialize components in correct order
        self.setup_plots()
        self.setup_video()
        self.setup_processor()
        self.setup_detector()
        self.init_ui()
        
        self.current_rois = None  # Changed to support multiple ROIs
        self.prev_frame = None
        self.last_frame_time = None
        
        # Initialize quality indicator
        self.setup_quality_indicator()
        
        # Measurement session tracking for average calculation
        self.session_hr_values = []  # Store HR values for current session
        self.session_start_time = None  # When current session started
        self.session_measurement_started = False  # Whether we have valid HR data

    def init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Left side - Video feed and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Video feed label with fixed size (responsive to screen size)
        self.video_label = QLabel()
        self.video_width = self.scaled(480)  # Fixed base width that scales
        self.video_height = self.scaled(360)  # Fixed base height that scales (4:3 ratio)
        self.video_label.setFixedSize(self.video_width, self.video_height)
        self.video_label.setScaledContents(False)  # Don't scale contents automatically
        self.video_label.setAlignment(Qt.AlignCenter)  # Center the content
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                background-color: #f0f0f0;
            }
        """)
        left_layout.addWidget(self.video_label)
        
        # Controls widget with background color and padding
        controls_widget = QWidget()
        controls_widget.setStyleSheet(f"""
            QWidget {{
                background-color: #E8EEF7;
                border-radius: {self.scaled(10)}px;
                padding: {self.scaled(10)}px;
            }}
            QComboBox {{
                background-color: white;
                border: 2px solid #D0D9E7;
                border-radius: {self.scaled(5)}px;
                padding: {self.scaled(8)}px;
                min-height: {self.scaled(35)}px;
                font-size: {self.scaled(14)}px;
            }}
            QPushButton {{
                background-color: #4A90E2;
                color: white;
                border: none;
                border-radius: {self.scaled(5)}px;
                padding: {self.scaled(10)}px {self.scaled(20)}px;
                font-size: {self.scaled(self.base_font_size)}px;
                min-height: {self.scaled(35)}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #357ABD;
            }}
            QPushButton:disabled {{
                background-color: #CCCCCC;
            }}
        """)
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setSpacing(self.scaled(15))  # Add some space between controls
        
        # Patient selection
        patient_label = QLabel('Patient:')
        patient_label.setStyleSheet('font-weight: bold;')
        controls_layout.addWidget(patient_label)
        
        self.patient_combo = QComboBox()
        self.patient_combo.addItem('Select Patient...')
        self.patient_combo.currentIndexChanged.connect(self.on_patient_selected)
        self.load_patients()
        controls_layout.addWidget(self.patient_combo)
        
        # Refresh patients button
        self.refresh_patients_btn = QPushButton('Refresh')
        self.refresh_patients_btn.clicked.connect(self.load_patients)
        self.refresh_patients_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: {self.scaled(5)}px;
                padding: {self.scaled(8)}px {self.scaled(15)}px;
                font-size: {self.scaled(12)}px;
                min-height: {self.scaled(30)}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #218838;
            }}
        """)
        controls_layout.addWidget(self.refresh_patients_btn)
        
        # Input type selection
        self.video_combo = QComboBox()
        self.video_combo.addItems(['Camera', 'Video File'])
        self.video_combo.currentIndexChanged.connect(self.on_input_type_changed)
        controls_layout.addWidget(self.video_combo)
        
        # Open file button (always visible but conditionally enabled)
        self.open_file_btn = QPushButton('Open File')
        self.open_file_btn.clicked.connect(self.open_video_file)
        self.open_file_btn.setEnabled(False)  # Initially disabled
        controls_layout.addWidget(self.open_file_btn)
        
        # Start/Stop button
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.toggle_video)
        controls_layout.addWidget(self.start_button)
        
        # Reset button
        self.reset_button = QPushButton('Reset Data')
        self.reset_button.clicked.connect(self.reset_all_data)
        self.reset_button.setEnabled(False)  # Initially disabled since no data
        self.reset_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: {self.scaled(5)}px;
                padding: {self.scaled(10)}px {self.scaled(20)}px;
                font-size: {self.scaled(self.base_font_size)}px;
                min-height: {self.scaled(35)}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #C0392B;
            }}
            QPushButton:disabled {{
                background-color: #CCCCCC;
            }}
        """)
        controls_layout.addWidget(self.reset_button)
        
        left_layout.addWidget(controls_widget)
        main_layout.addWidget(left_panel)
        
        # Right side - Measurements and plots
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Measurements display
        measurements_widget = QWidget()
        measurements_widget.setStyleSheet(f"""
            QWidget {{
                background-color: #E8EEF7;
                border-radius: {self.scaled(10)}px;
                padding: {self.scaled(20)}px;
            }}
            QLabel {{
                font-size: {self.scaled(self.base_font_size)}px;
            }}
        """)
        measurements_layout = QVBoxLayout(measurements_widget)
        
        # Patient information display
        patient_info_layout = QHBoxLayout()
        patient_info_label = QLabel('Current Patient:')
        patient_info_label.setStyleSheet('font-weight: bold;')
        self.patient_info_value = QLabel('None selected')
        self.patient_info_value.setStyleSheet('color: #666;')
        patient_info_layout.addWidget(patient_info_label)
        patient_info_layout.addWidget(self.patient_info_value)
        measurements_layout.addLayout(patient_info_layout)
        
        # Heart rate display with colored background
        hr_layout = QHBoxLayout()
        hr_label = QLabel('Heart Rate:')
        hr_label.setStyleSheet('font-weight: bold;')
        
        # Create a container widget for heart rate with colored background
        self.hr_container = QWidget()
        self.hr_container.setStyleSheet(f"""
            QWidget {{
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: {self.scaled(8)}px;
                padding: {self.scaled(12)}px;
                margin: {self.scaled(2)}px;
            }}
        """)
        hr_container_layout = QVBoxLayout(self.hr_container)
        hr_container_layout.setContentsMargins(self.scaled(12), self.scaled(12), self.scaled(12), self.scaled(12))
        hr_container_layout.setSpacing(self.scaled(6))
        
        # Heart rate value label
        self.heart_rate_value = QLabel('-- BPM')
        self.heart_rate_value.setStyleSheet(f"""
            font-weight: bold;
            font-size: {self.scaled(18)}px;
            color: #333;
        """)
        self.heart_rate_value.setAlignment(Qt.AlignCenter)
        hr_container_layout.addWidget(self.heart_rate_value)
        
        # Status message label (always visible with fixed height)
        self.hr_alert_label = QLabel('ROI Not Found')
        self.hr_alert_label.setStyleSheet(f"""
            font-weight: bold;
            font-size: {self.scaled(14)}px;
            color: #666;
            background-color: transparent;
        """)
        self.hr_alert_label.setAlignment(Qt.AlignCenter)
        
        hr_container_layout.addWidget(self.hr_alert_label)
        
        hr_layout.addWidget(hr_label)
        hr_layout.addWidget(self.hr_container)
        measurements_layout.addLayout(hr_layout)
        
        # Signal quality indicator
        quality_layout = QHBoxLayout()
        quality_label = QLabel('Frequency:')
        quality_label.setStyleSheet('font-weight: bold;')
        self.freq_value = QLabel('-- Hz')
        self.freq_value.setStyleSheet('font-weight: bold;')
        quality_layout.addWidget(quality_label)
        quality_layout.addWidget(self.freq_value)
        
        # Method indicator
        method_label = QLabel('Method:')
        method_label.setStyleSheet('font-weight: bold;')
        self.method_value = QLabel('--')
        self.method_value.setStyleSheet('font-weight: bold; color: #1976D2;')
        quality_layout.addWidget(method_label)
        quality_layout.addWidget(self.method_value)
        
        measurements_layout.addLayout(quality_layout)
        
        right_layout.addWidget(measurements_widget)
        
        # Initialize matplotlib figures and canvases
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        # Create signal plot
        plot_dpi = max(80, min(120, int(100 * self.scale_factor)))
        self.signal_figure = plt.figure(figsize=(8, 3), dpi=plot_dpi)
        self.signal_canvas = FigureCanvas(self.signal_figure)
        self.signal_canvas.setMinimumHeight(self.scaled(200))
        self.signal_ax = self.signal_figure.add_subplot(111)
        self.signal_ax.set_title('Raw Signal')
        self.signal_ax.set_xlabel('Time (s)')
        self.signal_ax.set_ylabel('Amplitude')
        self.signal_ax.grid(True)
        self.signal_figure.tight_layout()
        
        # Create heart rate plot
        self.hr_figure = plt.figure(figsize=(8, 3), dpi=plot_dpi)
        self.hr_canvas = FigureCanvas(self.hr_figure)
        self.hr_canvas.setMinimumHeight(self.scaled(200))
        self.hr_ax = self.hr_figure.add_subplot(111)
        self.hr_ax.set_title('Heart Rate Over Time')
        self.hr_ax.set_xlabel('Time (s)')
        self.hr_ax.set_ylabel('BPM')
        self.hr_ax.grid(True)
        self.hr_figure.tight_layout()
        
        # Create FFT plot
        self.fft_figure = plt.figure(figsize=(8, 3), dpi=plot_dpi)
        self.fft_canvas = FigureCanvas(self.fft_figure)
        self.fft_canvas.setMinimumHeight(self.scaled(200))
        self.fft_ax = self.fft_figure.add_subplot(111)
        self.fft_ax.set_title('FFT Power Spectrum')
        self.fft_ax.set_xlabel('Frequency (Hz)')
        self.fft_ax.set_ylabel('Power')
        self.fft_ax.grid(True)
        self.fft_figure.tight_layout()
        
        # Add plots to layout
        right_layout.addWidget(self.signal_canvas)
        right_layout.addWidget(self.hr_canvas)
        right_layout.addWidget(self.fft_canvas)
        
        # Initialize data storage
        from collections import deque
        self.raw_values = deque(maxlen=1000)
        self.signal_times = deque(maxlen=1000)
        self.hr_values = deque(maxlen=1000)
        self.hr_times = deque(maxlen=1000)
        self.signal_values = deque(maxlen=1000)
        self.fft_freqs = deque(maxlen=512)
        self.fft_power = deque(maxlen=512)
        self.cardiac_cycle_data = deque(maxlen=100)
        self.start_time = None
        
        main_layout.addWidget(right_panel)
        
        # Set stretch factors
        main_layout.setStretch(0, 1)  # Video feed takes 50% of width
        main_layout.setStretch(1, 1)  # Measurements take 50% of width
        
        # Add the main widget to the content layout
        self.content_layout.addWidget(main_widget)
        
        # Set up video capture and processing components
        self.setup_video()
        self.setup_processor()
        self.setup_detector()
        
        self.logger.info("Video setup complete")

    def setup_video(self):
        """Initialize video capture and processing components."""
        try:
            # Initialize video capture
            self.video_capture = VideoCapture()
            self.logger.info("VideoCapture initialized")

            # Initialize face detector and signal processor
            self.face_detector = FaceDetector()
            self.signal_processor = SignalProcessor()

            # Set up video timer for frame updates
            self.video_timer = QTimer()
            self.video_timer.timeout.connect(self.update_frame)
            
            # Set up plot timer for less frequent updates
            self.plot_timer = QTimer()
            self.plot_timer.timeout.connect(self.update_plots)
            
            # Initialize state variables
            self.is_running = False
            self.current_frame = None
            self.current_roi = None
            self.current_bpm = None
            
            # Double buffering for smooth display
            self.display_buffer = None
            self.buffer_ready = False
            
            self.logger.info("Video setup complete")
            
        except Exception as e:
            self.logger.error(f"Error in video setup: {str(e)}")
            raise

    def setup_processor(self):
        """Initialize the signal processor."""
        self.signal_processor = SignalProcessor()

    def setup_detector(self):
        """Initialize the face detector."""
        self.face_detector = FaceDetector()

    def setup_plots(self):
        """Set up the signal and heart rate plots."""
        pass  # Plot setup is now handled in init_ui

    def open_video_file(self):
        """Open a video file dialog."""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Open Video File",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
            )
            if file_name:
                self.video_file = file_name
                self.start_button.setEnabled(True)
                self.logger.info(f"Video file selected: {file_name}")
        except Exception as e:
            self.logger.error(f"Error opening video file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to open video file: {str(e)}")

    def on_input_type_changed(self, index):
        """Handle change of input type selection."""
        try:
            input_type = self.video_combo.currentText()
            if input_type == "Camera":
                self.start_button.setEnabled(True)
                self.open_file_btn.setEnabled(False)
            else:
                self.start_button.setEnabled(False)  # Disable until file is selected
                self.open_file_btn.setEnabled(True)
            
            # Stop current video if running
            if self.is_running:
                self.stop_video_processing()
            
        except Exception as e:
            self.logger.error(f"Error changing input type: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to change input type: {str(e)}")

    def toggle_video(self):
        """Start or stop video processing."""
        if not self.is_running:
            if self.video_combo.currentText() == 'Camera':
                try:
                    self.logger.info("Initializing webcam...")
                    self.start_video_processing()
                    self.logger.info("Webcam initialized successfully")
                except Exception as e:
                    self.logger.error(f"Webcam initialization error: {str(e)}")
                    QMessageBox.critical(self, "Error", 
                        "Could not initialize webcam. Please check if:\n"
                        "1. The webcam is properly connected\n"
                        "2. No other application is using the webcam\n"
                        "3. You have necessary permissions\n\n"
                        f"Error: {str(e)}")
                    self.reset_video_state()
            else:
                # For video file input
                if not hasattr(self, 'video_file') or not self.video_file:
                    QMessageBox.warning(self, "Error", "Please select a video file first")
                    return
                try:
                    self.start_video_processing()
                except Exception as e:
                    self.logger.error(f"Video file initialization error: {str(e)}")
                    QMessageBox.critical(self, "Error", 
                        f"Could not open video file: {str(e)}")
                    self.reset_video_state()
        else:
            self.stop_video_processing()

    def start_video_processing(self):
        """Start video capture and processing."""
        try:
            if not self.is_running:
                # Check if a patient is selected
                patient_id = self.get_selected_patient_id()
                if not patient_id:
                    QMessageBox.warning(self, "Patient Required", "Please select a patient before starting the measurement.")
                    return
                
                # Get selected input type
                input_type = self.video_combo.currentText()
                source = 0 if input_type == "Camera" else self.video_file
                
                # Try to start video capture
                if not self.video_capture.start(source):
                    error_msg = "Failed to open webcam" if input_type == "Camera" else f"Failed to open video file: {self.video_file}"
                    QMessageBox.warning(self, "Error", error_msg)
                    self.reset_video_state()
                    return
                
                # Start video processing
                self.video_timer.start(33)  # ~30 FPS
                self.plot_timer.start(100)  # Update plots every 100ms
                self.is_running = True
                self.start_button.setText("Stop")
                self.start_time = time.time()
                
                # Initialize measurement session tracking
                self.session_hr_values.clear()
                self.session_start_time = time.time()
                self.session_measurement_started = False
                
                # Reset data buffers for new measurement
                self.signal_times.clear()
                self.signal_values.clear()
                self.hr_times.clear()
                self.hr_values.clear()
                self.fft_freqs.clear()
                self.fft_power.clear()
                self.cardiac_cycle_data.clear()
                
                # Disable reset button when starting new measurement
                self.reset_button.setEnabled(False)
                
                self.logger.info("Video processing started successfully")
                
        except Exception as e:
            self.logger.error(f"Error starting video processing: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to start video processing: {str(e)}")
            self.reset_video_state()

    def stop_video_processing(self):
        """Stop video capture and processing."""
        try:
            if self.is_running:
                # Calculate and save session average before stopping
                self.calculate_and_save_session_average()
                
                self.video_timer.stop()
                self.plot_timer.stop()
                self.video_capture.stop()
                self.is_running = False
                self.start_button.setText("Start")
                self.current_roi = None
                self.current_frame = None
                self.display_buffer = None
                self.buffer_ready = False
                
                # Reset all data buffers
                has_data = len(self.raw_values) > 0
                self.signal_times.clear()
                self.signal_values.clear()
                self.raw_values.clear()
                self.hr_times.clear()
                self.hr_values.clear()
                self.fft_freqs.clear()
                self.fft_power.clear()
                self.cardiac_cycle_data.clear()
                self.start_time = None
                
                # Reset session tracking
                self.session_hr_values.clear()
                self.session_start_time = None
                self.session_measurement_started = False
                
                # Reset signal processor for fresh measurement
                if hasattr(self, 'signal_processor'):
                    self.signal_processor.reset()
                
                # Reset face detector for fresh measurement
                if hasattr(self, 'face_detector'):
                    self.face_detector.reset()
                
                # Reset display values
                self.heart_rate_value.setText('-- BPM')
                self.freq_value.setText('-- Hz')
                self.method_value.setText('--')
                
                # Reset heart rate container to default styling
                self.reset_hr_container_style()
                
                # Enable reset button if we had data before clearing
                if has_data:
                    self.reset_button.setEnabled(True)
                else:
                    self.reset_button.setEnabled(False)
                
                self.logger.info("Video processing stopped and data reset")
        except Exception as e:
            self.logger.error(f"Error stopping video processing: {str(e)}")

    def reset_video_state(self):
        """Reset video processing state."""
        self.current_roi = None
        self.prev_frame = None
        self.start_time = None
        
        # Reset all data buffers
        has_data = len(self.raw_values) > 0
        self.signal_times.clear()
        self.signal_values.clear()
        self.raw_values.clear()
        self.hr_times.clear()
        self.hr_values.clear()
        self.fft_freqs.clear()
        self.fft_power.clear()
        
        # Reset session tracking
        self.session_hr_values.clear()
        self.session_start_time = None
        self.session_measurement_started = False
        
        # Reset signal processor for fresh measurement
        if hasattr(self, 'signal_processor'):
            self.signal_processor.reset()
        
        # Reset face detector for fresh measurement
        if hasattr(self, 'face_detector'):
            self.face_detector.reset()
        
        # Reset display values
        self.heart_rate_value.setText('-- BPM')
        self.freq_value.setText('-- Hz')
        self.method_value.setText('--')
        
        # Reset heart rate container to default styling
        self.reset_hr_container_style()
        
        # Enable reset button if we had data before clearing
        if has_data:
            self.reset_button.setEnabled(True)
        else:
            self.reset_button.setEnabled(False)
        
        # Clear plots if they exist
        if hasattr(self, 'signal_ax') and self.signal_ax:
            self.signal_ax.clear()
            self.signal_canvas.draw()
        if hasattr(self, 'hr_ax') and self.hr_ax:
            self.hr_ax.clear()
            self.hr_canvas.draw()
        if hasattr(self, 'fft_ax') and self.fft_ax:
            self.fft_ax.clear()
            self.fft_canvas.draw()

    def setup_quality_indicator(self):
        """Set up the signal quality indicator."""
        self.quality_indicator = QLabel()
        self.quality_indicator.setStyleSheet("""
            QLabel {
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
            }
        """)
        self.update_quality_indicator(0.0)

    def update_quality_indicator(self, quality):
        """Update the signal quality indicator color and text."""
        if quality < self.quality_threshold:
            color = "#FF4444"  # Red
            text = "Poor Signal"
        elif quality < self.quality_threshold * 2:
            color = "#FFAA44"  # Orange
            text = "Fair Signal"
        else:
            color = "#44FF44"  # Green
            text = "Good Signal"
        
        self.quality_indicator.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: black;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
            }}
        """)
        self.quality_indicator.setText(text)

    def update_frame(self):
        """Update video frame and process for heart rate."""
        try:
            if not self.is_running:
                return

            # Read frame
            frame = self.video_capture.read_frame()
            if frame is None:
                self.logger.warning("Failed to read frame")
                self.stop_video_processing()
                return

            # Store current frame for processing
            self.current_frame = frame.copy()

            # Detect face and get all ROIs (forehead + cheeks)
            landmarks = self.face_detector.detect_face(frame)
            if landmarks is not None:
                self.current_rois = self.face_detector.get_all_rois(frame, landmarks)
            else:
                self.current_rois = None

            # Process frame in background
            self.process_frame(frame)
            
            # Display the processed frame
            self.display_frame(frame)
            
        except Exception as e:
            self.logger.error(f"Error in frame update: {str(e)}")

    def process_frame(self, frame):
        """Process a single frame of video."""
        try:
            if frame is None:
                return
            
            # Get current time relative to start
            if self.start_time is None:
                self.start_time = time.time()
            current_time = time.time() - self.start_time
            
            # Always call signal processor to check for ROI loss
            # Check if ROI was lost and clear displays
            if self.signal_processor.was_roi_lost():
                print("Clearing displays due to ROI loss")
                self.heart_rate_value.setText("-- BPM")
                self.freq_value.setText("-- Hz")
                self.method_value.setText("--")
                # Reset heart rate container to default styling
                self.reset_hr_container_style()
                # Don't clear plot data - keep history for graphs
            
            # Process frame if we have valid ROIs
            if self.current_rois is not None and any(self.current_rois.values()):
                # Show measuring status
                self.update_hr_status_message("Measuring...", "#4A90E2")
                
                # Calculate combined green channel mean from all ROIs
                total_green_mean = 0.0
                valid_rois = 0
                
                for roi_name, roi in self.current_rois.items():
                    if roi is not None:
                        x, y, w, h = roi
                        roi_frame = frame[y:y+h, x:x+w]
                        green_mean = np.mean(roi_frame[:, :, 1])
                        total_green_mean += green_mean
                        valid_rois += 1
                
                if valid_rois > 0:
                    combined_green_mean = total_green_mean / valid_rois
                    
                    # Store raw value
                    self.raw_values.append(combined_green_mean)
                    self.signal_times.append(current_time)
                    
                    # Enable reset button only when stopped and we have data
                    if not self.is_running and len(self.raw_values) > 0 and not self.reset_button.isEnabled():
                        self.reset_button.setEnabled(True)
                
                # Process signal to get heart rate using all ROIs
                result = self.signal_processor.process_frame(frame, self.current_rois)
                if len(result) == 5:
                    bpm, confidence, fft_freqs, fft_power, method_data = result
                elif len(result) == 4:
                    bpm, confidence, fft_freqs, fft_power = result
                    method_data = {}
                else:
                    bpm, confidence = result
                    fft_freqs, fft_power = None, None
                    method_data = {}
                
                if bpm is not None:
                    # Update heart rate display and save to database
                    self.update_hr_display(bpm)
                    
                    # Update FFT data if available
                    if fft_freqs is not None and fft_power is not None:
                        self.fft_freqs.extend(fft_freqs)
                        self.fft_power.extend(fft_power)
                    
                    # Update method data if available
                    if method_data:
                        self.cardiac_cycle_data.append(method_data)
                    
                    # Update frequency display
                    freq = bpm / 60.0
                    self.freq_value.setText(f"{freq:.2f} Hz")
                    
                    # Update method display (always FFT now)
                    if method_data and 'method_used' in method_data:
                        method = method_data['method_used']
                        if method == 'frequency_domain':
                            self.method_value.setText('FFT')
                            self.method_value.setStyleSheet('font-weight: bold; color: #2E7D32;')  # Green for FFT
                        else:
                            self.method_value.setText('FFT')
                            self.method_value.setStyleSheet('font-weight: bold; color: #FF9800;')  # Orange for failed
                    else:
                        self.method_value.setText('FFT')
                        self.method_value.setStyleSheet('font-weight: bold; color: #2E7D32;')  # Default to FFT
                
                # Draw all ROIs on frame with different colors
                colors = {
                    'forehead': (0, 255, 0),    # Green
                    'left_cheek': (255, 0, 0),  # Blue
                    'right_cheek': (0, 0, 255)  # Red
                }
                
                for roi_name, roi in self.current_rois.items():
                    if roi is not None:
                        x, y, w, h = roi
                        color = colors.get(roi_name, (0, 255, 0))
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        # Add label
                        cv2.putText(frame, roi_name.replace('_', ' ').title(), 
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # No ROIs detected - show ROI not found message
                self.update_hr_status_message("ROI Not Found", "#666")
                # Call signal processor with None ROIs to trigger reset
                self.signal_processor.process_frame(frame, None)
            
            # Update plots
            if len(self.raw_values) > 0:
                self.update_plots()
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")

    def display_frame(self, frame):
        """Display the frame in the video label with proper aspect ratio handling."""
        try:
            if frame is None:
                return
            
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get fixed label dimensions
            label_width = self.video_width
            label_height = self.video_height
            frame_height, frame_width = rgb_frame.shape[:2]
            
            # Calculate aspect ratio preserving dimensions
            frame_aspect = frame_width / frame_height
            label_aspect = label_width / label_height
            
            # Calculate the size to fit the frame within the label while preserving aspect ratio
            if frame_aspect > label_aspect:
                # Frame is wider than label - fit to width
                new_width = label_width
                new_height = int(label_width / frame_aspect)
            else:
                # Frame is taller than label - fit to height
                new_height = label_height
                new_width = int(label_height * frame_aspect)
            
            # Resize frame to calculated dimensions
            resized_frame = cv2.resize(rgb_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create a canvas with the label size and fill with background color
            canvas = np.full((label_height, label_width, 3), 240, dtype=np.uint8)  # Light gray background
            
            # Calculate position to center the resized frame
            x_offset = (label_width - new_width) // 2
            y_offset = (label_height - new_height) // 2
            
            # Place the resized frame on the canvas
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
            
            # Convert to QImage
            bytes_per_line = label_width * 3
            q_image = QImage(canvas.data, label_width, label_height, 
                            bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap and set to label
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap)
            
            # Force immediate update
            self.video_label.update()
            
        except Exception as e:
            self.logger.error(f"Error displaying frame: {str(e)}")

    def update_hr_display(self, bpm):
        """Update heart rate display and save to database if patient is selected."""
        try:
            self.heart_rate_value.setText(f"{bpm:.1f} BPM")
            
            # Update heart rate container styling based on BPM value
            self.update_hr_container_style(bpm)
            
            # Track heart rate for session average calculation
            if not self.session_measurement_started:
                self.session_measurement_started = True
                self.logger.info("Heart rate measurement started for session average")
            
            self.session_hr_values.append(bpm)
            
            # Update plot data with relative time
            if self.start_time is None:
                self.start_time = time.time()
            current_time = time.time() - self.start_time
            
            self.hr_times.append(current_time)
            self.hr_values.append(bpm)
            
            # Save measurement to database if a patient is selected
            patient_id = self.get_selected_patient_id()
            if patient_id:
                try:
                    from database.db import Database
                    db = Database()
                    
                    # Determine status based on heart rate value
                    if 60 <= bpm <= 100:
                        status = "Normal"
                    elif bpm < 60:
                        status = "Bradycardia"
                    else:
                        status = "Tachycardia"
                    
                    # Save measurement to database
                    if db.add_measurement(patient_id, int(bpm), status):
                        self.logger.info(f"Saved measurement for patient {patient_id}: {bpm:.1f} BPM ({status})")
                    else:
                        self.logger.warning(f"Failed to save measurement for patient {patient_id}")
                        
                except Exception as db_error:
                    self.logger.error(f"Database error saving measurement: {str(db_error)}")
            
        except Exception as e:
            self.logger.error("Error updating HR display: %s", str(e))

    def update_hr_container_style(self, bpm):
        """Update the heart rate container styling based on BPM value."""
        try:
            # Define heart rate ranges and colors
            if bpm < 40:
                # Very low heart rate - Red alert
                bg_color = "#FF4444"  # Red
                border_color = "#CC0000"  # Darker red
                text_color = "#FFFFFF"  # White text
                alert_text = "CRITICAL: Very Low HR"
            elif bpm < 50:
                # Low heart rate - Orange warning
                bg_color = "#FF8800"  # Orange
                border_color = "#CC6600"  # Darker orange
                text_color = "#FFFFFF"  # White text
                alert_text = "WARNING: Low HR"
            elif bpm <= 120:
                # Normal heart rate - Green
                bg_color = "#44FF44"  # Green
                border_color = "#00CC00"  # Darker green
                text_color = "#000000"  # Black text
                alert_text = "NORMAL"
            else:
                # High heart rate - Red alert
                bg_color = "#FF4444"  # Red
                border_color = "#CC0000"  # Darker red
                text_color = "#FFFFFF"  # White text
                alert_text = "ALERT: High HR"
            
            # Update container styling
            self.hr_container.setStyleSheet(f"""
                QWidget {{
                    background-color: {bg_color};
                    border: 2px solid {border_color};
                    border-radius: {self.scaled(8)}px;
                    padding: {self.scaled(12)}px;
                    margin: {self.scaled(2)}px;
                }}
            """)
            
            # Update heart rate value text color
            self.heart_rate_value.setStyleSheet(f"""
                font-weight: bold;
                font-size: {self.scaled(18)}px;
                color: {text_color};
            """)
            
            # Always show alert message with appropriate content
            self.hr_alert_label.setText(alert_text)
            self.hr_alert_label.setStyleSheet(f"""
                font-weight: bold;
                font-size: {self.scaled(14)}px;
                color: {text_color};
                background-color: transparent;
            """)
                
        except Exception as e:
            self.logger.error(f"Error updating HR container style: {str(e)}")

    def reset_hr_container_style(self):
        """Reset the heart rate container to default styling."""
        try:
            # Reset to default neutral styling
            self.hr_container.setStyleSheet(f"""
                QWidget {{
                    background-color: #f0f0f0;
                    border: 2px solid #ccc;
                    border-radius: {self.scaled(8)}px;
                    padding: {self.scaled(12)}px;
                    margin: {self.scaled(2)}px;
                }}
            """)
            
            # Reset heart rate value text color to default
            self.heart_rate_value.setStyleSheet(f"""
                font-weight: bold;
                font-size: {self.scaled(18)}px;
                color: #333;
            """)
            
            # Show ROI not found message
            self.hr_alert_label.setText("ROI Not Found")
            self.hr_alert_label.setStyleSheet(f"""
                font-weight: bold;
                font-size: {self.scaled(14)}px;
                color: #666;
                background-color: transparent;
            """)
            
        except Exception as e:
            self.logger.error(f"Error resetting HR container style: {str(e)}")

    def update_hr_status_message(self, message, color="#666"):
        """Update the heart rate status message without changing the container styling."""
        try:
            self.hr_alert_label.setText(message)
            self.hr_alert_label.setStyleSheet(f"""
                font-weight: bold;
                font-size: {self.scaled(14)}px;
                color: {color};
                background-color: transparent;
            """)
        except Exception as e:
            self.logger.error(f"Error updating HR status message: {str(e)}")

    def calculate_and_save_session_average(self):
        """Calculate accurate average heart rate and save measurement session to database."""
        try:
            if not self.session_measurement_started or len(self.session_hr_values) == 0:
                self.logger.info("No heart rate data available for session average")
                return
            
            # Calculate measurement duration
            if self.session_start_time:
                measurement_duration = time.time() - self.session_start_time
            else:
                measurement_duration = 0
            
            # Minimum requirements for a valid measurement session
            MIN_DURATION_SECONDS = 30  # At least 30 seconds
            MIN_MEASUREMENTS = 15      # At least 15 heart rate readings
            
            if measurement_duration < MIN_DURATION_SECONDS:
                self.logger.info(f"Measurement session too short ({measurement_duration:.1f}s < {MIN_DURATION_SECONDS}s). Not saving session.")
                return
            
            if len(self.session_hr_values) < MIN_MEASUREMENTS:
                self.logger.info(f"Too few measurements ({len(self.session_hr_values)} < {MIN_MEASUREMENTS}). Not saving session.")
                return
            
            # Calculate accurate average heart rate
            avg_hr = sum(self.session_hr_values) / len(self.session_hr_values)
            
            # Get patient information
            patient_id = self.get_selected_patient_id()
            if not patient_id:
                self.logger.warning("No patient selected for session average")
                return
            
            # Save measurement session to database
            from database.db import Database
            db = Database()
            
            # Get current date and time
            from datetime import datetime
            current_datetime = datetime.now()
            
            # Determine status based on average heart rate
            if 60 <= avg_hr <= 100:
                status = "Normal"
            elif avg_hr < 60:
                status = "Bradycardia"
            else:
                status = "Tachycardia"
            
            # Save session measurement
            success = db.add_measurement_session(
                patient_id=patient_id,
                avg_heart_rate=round(avg_hr, 1),
                status=status,
                measurement_date=current_datetime.date(),
                measurement_time=current_datetime.time(),
                duration_seconds=round(measurement_duration, 1),
                total_measurements=len(self.session_hr_values)
            )
            
            if success:
                self.logger.info(f"Session average saved: {avg_hr:.1f} BPM over {measurement_duration:.1f}s with {len(self.session_hr_values)} measurements")
            else:
                self.logger.error("Failed to save session average to database")
                
        except Exception as e:
            self.logger.error(f"Error calculating and saving session average: {str(e)}")

    def update_plots(self):
        """Update signal, heart rate, and FFT plots."""
        try:
            # Clear previous plots
            self.signal_ax.clear()
            self.hr_ax.clear()
            self.fft_ax.clear()
            
            # Plot raw signal if we have data
            if len(self.raw_values) > 0:
                # Convert lists to numpy arrays for easier manipulation
                times = np.array(list(self.signal_times))
                values = np.array(list(self.raw_values))
                
                # Plot the raw signal
                self.signal_ax.plot(times, values, 'g-', linewidth=1, label='rPPG Signal')
                
                # No cardiac cycle markers needed since we're using FFT only
                
                # Set axis labels and title
                self.signal_ax.set_title('rPPG Signal')
                self.signal_ax.set_xlabel('Time (s)')
                self.signal_ax.set_ylabel('Amplitude')
                
                # Set x-axis limits to show last 10 seconds
                if len(times) > 0:
                    x_max = times[-1]
                    x_min = max(0, x_max - self.window_duration)
                    self.signal_ax.set_xlim(x_min, x_max)
                
                # Auto-scale y-axis
                self.signal_ax.set_ylim(np.min(values) - 1, np.max(values) + 1)
                
                # Add grid and legend
                self.signal_ax.grid(True, alpha=0.3)
                self.signal_ax.legend(loc='upper right', fontsize=8)
            
            # Plot heart rate if we have data
            if len(self.hr_values) > 0:
                # Convert lists to numpy arrays
                hr_times = np.array(list(self.hr_times))
                hr_values = np.array(list(self.hr_values))
                
                # Plot the heart rate
                self.hr_ax.plot(hr_times, hr_values, 'r-', linewidth=1)
                
                # Set axis labels and title
                self.hr_ax.set_title('Heart Rate Over Time')
                self.hr_ax.set_xlabel('Time (s)')
                self.hr_ax.set_ylabel('BPM')
                
                # Set x-axis limits to show last 10 seconds
                if len(hr_times) > 0:
                    x_max = hr_times[-1]
                    x_min = max(0, x_max - self.window_duration)
                    self.hr_ax.set_xlim(x_min, x_max)
                
                # Set y-axis limits for heart rate with some padding
                min_hr = max(40, np.min(hr_values) - 10)
                max_hr = min(180, np.max(hr_values) + 10)
                self.hr_ax.set_ylim(min_hr, max_hr)
                
                # Add grid
                self.hr_ax.grid(True, alpha=0.3)
            
            # Plot FFT if we have data
            if len(self.fft_freqs) > 0 and len(self.fft_power) > 0:
                # Convert lists to numpy arrays
                fft_freqs = np.array(list(self.fft_freqs))
                fft_power = np.array(list(self.fft_power))
                
                # Plot the FFT power spectrum
                self.fft_ax.plot(fft_freqs, fft_power, 'b-', linewidth=1)
                
                # Set axis labels and title
                self.fft_ax.set_title('FFT Power Spectrum')
                self.fft_ax.set_xlabel('Frequency (Hz)')
                self.fft_ax.set_ylabel('Power')
                
                # Set x-axis limits for heart rate range
                self.fft_ax.set_xlim(0.5, 3.0)
                
                # Auto-scale y-axis
                if len(fft_power) > 0:
                    self.fft_ax.set_ylim(0, np.max(fft_power) * 1.1)
                
                # Add grid
                self.fft_ax.grid(True, alpha=0.3)
                
                # Add vertical lines for common heart rate frequencies
                self.fft_ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='60 BPM')
                self.fft_ax.axvline(x=1.2, color='g', linestyle='--', alpha=0.5, label='72 BPM')
                self.fft_ax.axvline(x=1.5, color='orange', linestyle='--', alpha=0.5, label='90 BPM')
            
            # Adjust layout to prevent overlapping
            self.signal_figure.tight_layout()
            self.hr_figure.tight_layout()
            self.fft_figure.tight_layout()
            
            # Redraw all canvases
            self.signal_canvas.draw()
            self.hr_canvas.draw()
            self.fft_canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating plots: {str(e)}")

    def reset_all_data(self):
        """Reset all data (raw values, heart rate, signal, plots, and display)."""
        self.logger.info("Resetting all data...")
        self.raw_values.clear()
        self.signal_times.clear()
        self.hr_values.clear()
        self.hr_times.clear()
        self.signal_values.clear()
        self.start_time = None
        
        # Reset session tracking
        self.session_hr_values.clear()
        self.session_start_time = None
        self.session_measurement_started = False
        
        # Reset signal processor for fresh measurement
        if hasattr(self, 'signal_processor'):
            self.signal_processor.reset()
        
        # Reset face detector for fresh measurement
        if hasattr(self, 'face_detector'):
            self.face_detector.reset()
        
        # Clear plots
        if hasattr(self, 'signal_ax') and self.signal_ax:
            self.signal_ax.clear()
            self.signal_canvas.draw()
        if hasattr(self, 'hr_ax') and self.hr_ax:
            self.hr_ax.clear()
            self.hr_canvas.draw()
        
        # Reset display values
        self.heart_rate_value.setText('-- BPM')
        self.freq_value.setText('-- Hz')
        self.method_value.setText('--')
        
        # Reset heart rate container to default styling
        self.reset_hr_container_style()
        
        # Disable reset button since no data is available after reset
        self.reset_button.setEnabled(False)
        
        self.logger.info("All data reset.")

    def load_patients(self):
        """Load patients from database and populate the combo box."""
        try:
            from database.db import Database
            db = Database()
            
            # Get current user role to filter patients if needed
            user_role = getattr(self, 'user_role', None)
            
            # Get patients from database
            if user_role == 'Administrator':
                # Administrators can see all patients
                patients = db.get_all_patients()
            else:
                # Other users see only their assigned patients
                # For now, we'll show all patients since we don't have the current username
                # In a real implementation, you'd pass the username from login
                patients = db.get_all_patients()
            
            # Temporarily disconnect the signal to prevent triggering on_patient_selected during loading
            self.patient_combo.currentIndexChanged.disconnect()
            
            # Clear existing items except the first "Select Patient..." item
            self.patient_combo.clear()
            self.patient_combo.addItem('Select Patient...')
            
            # Add patients to combo box
            for patient in patients:
                display_text = f"{patient['firstName']} {patient['lastName']} (ID: {patient['id']})"
                self.patient_combo.addItem(display_text, patient['id'])
            
            # Reconnect the signal
            self.patient_combo.currentIndexChanged.connect(self.on_patient_selected)
            
            self.logger.info(f"Loaded {len(patients)} patients into combo box")
            
        except Exception as e:
            self.logger.error(f"Error loading patients: {str(e)}")
            # Add a fallback item
            self.patient_combo.clear()
            self.patient_combo.addItem('Select Patient...')
            self.patient_combo.addItem('Error loading patients')
            # Reconnect the signal in case of error
            try:
                self.patient_combo.currentIndexChanged.connect(self.on_patient_selected)
            except:
                pass

    def get_selected_patient_id(self):
        """Get the ID of the currently selected patient."""
        current_index = self.patient_combo.currentIndex()
        if current_index > 0:  # Skip the "Select Patient..." item
            return self.patient_combo.itemData(current_index)
        return None

    def on_patient_selected(self):
        """Handle patient selection change."""
        # Safety check to ensure UI elements exist
        if not hasattr(self, 'patient_info_value'):
            return
            
        patient_id = self.get_selected_patient_id()
        if patient_id:
            try:
                from database.db import Database
                db = Database()
                patient = db.get_patient(patient_id)
                if patient:
                    display_text = f"{patient['firstName']} {patient['lastName']} (ID: {patient['id']})"
                    self.patient_info_value.setText(display_text)
                    self.patient_info_value.setStyleSheet('color: #333; font-weight: bold;')
                else:
                    self.patient_info_value.setText('Patient not found')
                    self.patient_info_value.setStyleSheet('color: #e74c3c;')
            except Exception as e:
                self.logger.error(f"Error loading patient info: {str(e)}")
                self.patient_info_value.setText('Error loading patient info')
                self.patient_info_value.setStyleSheet('color: #e74c3c;')
        else:
            self.patient_info_value.setText('None selected')
            self.patient_info_value.setStyleSheet('color: #666;')

    def go_back(self):
        """Return to home window."""
        from .home_window import HomeWindow
        # Use the stored user role if available
        user_role = getattr(self, 'user_role', None)
        self.home_window = HomeWindow(user_role=user_role)
        self.home_window.show()
        self.close()

    def closeEvent(self, event):
        """Handle window close event."""
        try:
            self.stop_video_processing()
            event.accept()
        except Exception as e:
            self.logger.error(f"Error in close event: {str(e)}")
            event.accept() 