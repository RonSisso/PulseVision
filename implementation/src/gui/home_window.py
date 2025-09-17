import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from utils.path_utils import get_assets_path
from .main_window import MainWindow
from .add_patient_window import AddPatientWindow
from .add_user_window import AddUserWindow
from .measurement_history_window import MeasurementHistoryWindow
from .base_window import BaseWindow

class HomeWindow(BaseWindow):
    def __init__(self, user_role=None):
        # Initialize with no back button but with power off button
        super().__init__(show_back_button=False, show_power_off=True)
        self.user_role = user_role
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        # Remove default content margins to allow image to touch borders
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Determine layout based on screen aspect ratio
        screen_aspect_ratio = self.screen_width / self.screen_height
        use_vertical_layout = screen_aspect_ratio < 1.6  # Use vertical layout for narrow screens
        
        if use_vertical_layout:
            # Vertical layout for smaller/narrower screens
            main_content = QVBoxLayout()
            self.setup_vertical_layout(main_content)
        else:
            # Horizontal layout for wider screens
            main_content = QHBoxLayout()
            main_content.setContentsMargins(0, 0, 0, 0)  # No margins
            main_content.setSpacing(0)  # No spacing between panels
            self.setup_horizontal_layout(main_content)
        
        # Add main content to the window
        self.content_layout.addLayout(main_content)

    def setup_horizontal_layout(self, main_content):
        """Setup horizontal layout for wider screens."""
        # Left side - Menu
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(self.scaled(30), self.scaled(40), self.scaled(20), self.scaled(40))
        
        # Menu title
        title_label = QLabel('Menu')
        title_label.setFont(QFont('Arial', self.scaled(48)))
        title_label.setStyleSheet('color: #2C3E50;')
        left_layout.addWidget(title_label)
        
        # Add some space after title
        left_layout.addSpacing(self.scaled(40))
        
        # Add stretch to push buttons to center-lower area
        left_layout.addStretch(1)
        
        # Create buttons for horizontal layout
        self.create_menu_buttons(left_layout, horizontal=True)
        
        # Add stretch at bottom (less than top to keep buttons in lower-center)
        left_layout.addStretch(2)
        
        # Right side - Menu Image (full size, touching borders)
        right_panel = QWidget()
        right_panel.setContentsMargins(0, 0, 0, 0)
        
        # Create a layout that fills the entire panel
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        right_layout.setSpacing(0)  # No spacing
        
        # Add menu image that fills the entire right panel
        menu_image = QLabel()
        menu_image.setContentsMargins(0, 0, 0, 0)
        menu_pixmap = QPixmap(get_assets_path('menu.png'))
        
        # Calculate exact available space (full right panel)
        available_width = int(self.screen_width * 0.7)  # 70% of screen width
        available_height = self.screen_height - self.scaled(80)  # Screen height minus header
        
        # Scale image to fit the available space while showing the full image
        scaled_pixmap = menu_pixmap.scaled(available_width, available_height,
                                         Qt.KeepAspectRatio,  # This will show full image
                                         Qt.SmoothTransformation)
        menu_image.setPixmap(scaled_pixmap)
        menu_image.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        menu_image.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)
        
        # Make the label fill the entire layout
        right_layout.addWidget(menu_image)
        
        # Add panels to main content
        main_content.addWidget(left_panel)
        main_content.addWidget(right_panel)
        
        # Set stretch factors - give more space to the image
        main_content.setStretch(0, 3)  # Left panel takes 30% of width
        main_content.setStretch(1, 7)  # Right panel takes 70% of width

    def setup_vertical_layout(self, main_content):
        """Setup vertical layout for smaller/narrower screens."""
        # Top section - Menu title and image (smaller)
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)
        
        # Menu title (left side)
        title_label = QLabel('Menu')
        title_label.setFont(QFont('Arial', self.scaled(36)))  # Smaller title for vertical layout
        title_label.setStyleSheet('color: #2C3E50;')
        top_layout.addWidget(title_label)
        
        # Add stretch
        top_layout.addStretch()
        
        # Small menu image (right side)
        menu_image = QLabel()
        menu_pixmap = QPixmap(get_assets_path('menu.png'))
        # Smaller image for vertical layout
        max_image_width = self.scaled(150)
        max_image_height = self.scaled(150)
        scaled_pixmap = menu_pixmap.scaled(max_image_width, max_image_height,
                                         Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
        menu_image.setPixmap(scaled_pixmap)
        menu_image.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(menu_image)
        
        # Bottom section - Buttons (centered)
        buttons_section = QWidget()
        buttons_outer_layout = QHBoxLayout(buttons_section)
        
        # Add stretch on left
        buttons_outer_layout.addStretch()
        
        # Create centered button container
        buttons_container = QWidget()
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, self.scaled(20), 0, 0)
        
        # Create buttons for vertical layout
        self.create_menu_buttons(buttons_layout, horizontal=False)
        
        # Add stretch to push buttons up
        buttons_layout.addStretch()
        
        # Add button container to outer layout
        buttons_outer_layout.addWidget(buttons_container)
        
        # Add stretch on right
        buttons_outer_layout.addStretch()
        
        # Add sections to main content
        main_content.addWidget(top_section)
        main_content.addWidget(buttons_section)
        
        # Set stretch factors - give more space to buttons
        main_content.setStretch(0, 1)  # Top section takes less space
        main_content.setStretch(1, 3)  # Buttons section takes more space

    def create_menu_buttons(self, layout, horizontal=True):
        """Create menu buttons with appropriate styling for the layout type."""
        # Adjust button styling based on layout
        if horizontal:
            button_width_factor = 1.0
            font_size = self.scaled(18)
            button_height_extra = 10
        else:
            # For vertical layout, make buttons wider and adjust font
            button_width_factor = 1.2
            font_size = self.scaled(20)
            button_height_extra = 15
        
        # Add buttons with consistent styling
        button_style = f"""
            QPushButton {{
                background-color: #5B6B7C;
                color: white;
                border: none;
                border-radius: {self.scaled(15)}px;
                padding: {self.scaled(12)}px {self.scaled(25)}px;
                font-size: {font_size}px;
                text-align: center;
                margin: {self.scaled(self.base_margin)}px 0;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #4A5A6C;
            }}
        """
        
        # Calculate button height with extra space for text
        button_height = self.scaled(self.base_button_height + button_height_extra)
        
        # Calculate button width for vertical layout
        if not horizontal:
            # For vertical layout, use a percentage of screen width but with limits
            button_width = min(self.scaled(400), int(self.screen_width * 0.6))
        
        # Add New Patient button
        add_patient_btn = QPushButton('Add New Patient')
        add_patient_btn.setFixedHeight(button_height)
        if not horizontal:
            add_patient_btn.setFixedWidth(button_width)
        add_patient_btn.setStyleSheet(button_style)
        add_patient_btn.clicked.connect(self.show_add_patient)
        layout.addWidget(add_patient_btn)
        
        # Add User button (only for administrators)
        if self.user_role == 'Administrator':
            add_user_btn = QPushButton('Add User')
            add_user_btn.setFixedHeight(button_height)
            if not horizontal:
                add_user_btn.setFixedWidth(button_width)
            add_user_btn.setStyleSheet(button_style)
            add_user_btn.clicked.connect(self.show_add_user)
            layout.addWidget(add_user_btn)
        
        # Patient Control Panel button
        control_panel_btn = QPushButton('Patient Control Panel')
        control_panel_btn.setFixedHeight(button_height)
        if not horizontal:
            control_panel_btn.setFixedWidth(button_width)
        control_panel_btn.setStyleSheet(button_style)
        control_panel_btn.clicked.connect(self.show_control_panel)
        layout.addWidget(control_panel_btn)
        
        # Measurement History button
        history_btn = QPushButton('Measurement History')
        history_btn.setFixedHeight(button_height)
        if not horizontal:
            history_btn.setFixedWidth(button_width)
        history_btn.setStyleSheet(button_style)
        history_btn.clicked.connect(self.show_measurement_history)
        layout.addWidget(history_btn)
        
        # Logout button
        logout_btn = QPushButton('Logout')
        logout_btn.setFixedHeight(button_height)
        if not horizontal:
            logout_btn.setFixedWidth(button_width)
        logout_btn.setStyleSheet(button_style)
        logout_btn.clicked.connect(self.logout)
        layout.addWidget(logout_btn)

    def show_add_patient(self):
        """Show the add patient window."""
        self.add_patient_window = AddPatientWindow(user_role=self.user_role)
        self.add_patient_window.show()
        self.hide()

    def show_add_user(self):
        """Show the add user window."""
        self.add_user_window = AddUserWindow(user_role=self.user_role)
        self.add_user_window.show()
        self.hide()

    def show_control_panel(self):
        """Show the patient control panel (main window)."""
        self.main_window = MainWindow()
        # Store the user role in the main window for later use
        self.main_window.user_role = self.user_role
        self.main_window.show()
        self.hide()

    def show_measurement_history(self):
        """Show the measurement history window."""
        self.measurement_history_window = MeasurementHistoryWindow(user_role=self.user_role)
        self.measurement_history_window.show()
        self.hide()

    def logout(self):
        """Return to login window."""
        from .login_window import LoginWindow
        self.login_window = LoginWindow()
        self.login_window.show()
        self.close()

    def closeEvent(self, event):
        """Handle window close event."""
        event.accept() 