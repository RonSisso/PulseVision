from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QStyle, QApplication, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPixmap, QIcon
import os

class BaseWindow(QMainWindow):
    def __init__(self, show_back_button=True, show_power_off=True):
        super().__init__()
        self.show_back_button = show_back_button
        self.show_power_off = show_power_off
        self.setup_base_ui()

    def setup_base_ui(self):
        """Set up the base UI elements common to all windows."""
        # Set window to fullscreen
        self.showFullScreen()
        
        # Get screen dimensions and DPI
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        self.screen_width = screen_size.width()
        self.screen_height = screen_size.height()
        
        # Calculate scaling factors for responsive design
        self.dpi = screen.logicalDotsPerInch()
        self.scale_factor = self.calculate_scale_factor()
        
        # Define base sizes that will be scaled
        self.base_button_height = 60
        self.base_font_size = 16
        self.base_title_font_size = 32
        self.base_padding = 20
        self.base_margin = 10
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("background-color: #f3f6ff;")
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create header bar
        header = QWidget()
        header.setFixedHeight(self.scaled(80))
        header.setStyleSheet("""
            QWidget {
                background-color: #E8EEF7;
                border-bottom: 2px solid #D0D9E7;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(self.scaled(self.base_padding), 0, 
                                       self.scaled(self.base_padding), 0)
        
        # Add PulseVision title (left-aligned)
        title = QLabel('PulseVision')
        title.setFont(QFont('Arial', self.scaled(self.base_title_font_size), QFont.Bold))
        title.setStyleSheet('color: #1A237E;')
        header_layout.addWidget(title)
        
        # Add stretch to push buttons to the right
        header_layout.addStretch()
        
        # Add back button if needed
        if self.show_back_button:
            back_button = QPushButton()
            button_size = self.scaled(50)
            back_button.setFixedSize(button_size, button_size)
            back_button.setCursor(Qt.PointingHandCursor)
            
            # Use back button image
            back_icon = QPixmap(os.path.join('assets', 'back_button.png'))
            back_button.setIcon(QIcon(back_icon))
            back_button.setIconSize(back_button.size())
            back_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: none;
                }}
                QPushButton:hover {{
                    background-color: rgba(0, 0, 0, 0.1);
                    border-radius: {self.scaled(5)}px;
                }}
            """)
            back_button.clicked.connect(self.go_back)
            header_layout.addWidget(back_button)
        
        # Add power off button if needed
        if self.show_power_off:
            power_button = QPushButton()
            button_size = self.scaled(50)
            power_button.setFixedSize(button_size, button_size)
            power_button.setCursor(Qt.PointingHandCursor)
            
            # Use Qt's built-in power off icon
            power_button.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
            power_button.setIconSize(power_button.size())
            power_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: none;
                }}
                QPushButton:hover {{
                    background-color: rgba(255, 0, 0, 0.1);
                    border-radius: {self.scaled(5)}px;
                }}
            """)
            power_button.clicked.connect(self.power_off)
            header_layout.addWidget(power_button)
        
        # Add header to main layout
        self.main_layout.addWidget(header)
        
        # Create content widget for derived classes
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        content_margin = self.scaled(40)
        self.content_layout.setContentsMargins(content_margin, content_margin, 
                                             content_margin, content_margin)
        self.main_layout.addWidget(self.content_widget)

    def calculate_scale_factor(self):
        """Calculate an appropriate scale factor based on screen size and DPI."""
        # Base reference: 1920x1080 at 96 DPI
        base_width = 1920
        base_height = 1080
        base_dpi = 96
        
        # Calculate scale based on screen dimensions (with limits)
        width_scale = min(self.screen_width / base_width, 2.0)  # Max 2x scaling
        height_scale = min(self.screen_height / base_height, 2.0)  # Max 2x scaling
        size_scale = min(width_scale, height_scale)
        
        # Calculate DPI scale (with limits)
        dpi_scale = min(self.dpi / base_dpi, 1.5)  # Max 1.5x DPI scaling
        
        # Combine scales with weighting (favor size over DPI)
        final_scale = (size_scale * 0.7) + (dpi_scale * 0.3)
        
        # Ensure minimum scale of 0.8 and maximum of 1.8
        return max(0.8, min(final_scale, 1.8))

    def scaled(self, value):
        """Scale a value according to the calculated scale factor."""
        return int(value * self.scale_factor)

    def go_back(self):
        """Virtual method to be implemented by derived classes."""
        pass

    def power_off(self):
        """Close the application."""
        QApplication.quit() 