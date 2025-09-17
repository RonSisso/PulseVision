import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QLineEdit, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from utils.path_utils import get_assets_path
from database.db import Database
from .base_window import BaseWindow

class AddUserWindow(BaseWindow):
    def __init__(self, user_role='Administrator'):
        # Initialize with back button but no power off button
        super().__init__(show_back_button=True, show_power_off=False)
        self.user_role = user_role
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        # Create main content widget
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Left side - Image (half screen width)
        image_label = QLabel()
        pixmap = QPixmap(get_assets_path('add_patient.png'))
        # Scale image to half screen width
        image_width = int(self.screen_width * 0.5)
        image_height = self.screen_height - self.scaled(80)  # Screen height minus header
        image_label.setPixmap(pixmap.scaled(image_width, image_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        main_layout.addWidget(image_label)
        
        # Right side - Form
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setSpacing(15)  # Reduced spacing
        form_layout.setContentsMargins(30, 30, 30, 30)  # Reduced margins
        
        # Title
        title_label = QLabel('Add User')
        title_label.setFont(QFont('Arial', 28))  # Smaller title
        form_layout.addWidget(title_label)
        
        # Input fields style
        input_style = """
            QLineEdit {
                border: 2px solid #E0E0E0;
                border-radius: 20px;
                padding: 12px;
                font-size: 14px;
                margin: 8px 0;
            }
            QLineEdit:focus {
                border-color: #6C63FF;
            }
        """
        
        combo_style = """
            QComboBox {
                border: 2px solid #E0E0E0;
                border-radius: 20px;
                padding: 12px;
                font-size: 14px;
                margin: 8px 0;
                background-color: white;
            }
            QComboBox:focus {
                border-color: #6C63FF;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #666;
                margin-right: 15px;
            }
        """
        
        # Username
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText('Enter username')
        self.username_input.setStyleSheet(input_style)
        form_layout.addWidget(self.username_input)
        
        # Password
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText('Enter password')
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet(input_style)
        form_layout.addWidget(self.password_input)
        
        # Confirm Password
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setPlaceholderText('Confirm password')
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        self.confirm_password_input.setStyleSheet(input_style)
        form_layout.addWidget(self.confirm_password_input)
        
        # Role selection
        self.role_combo = QComboBox()
        self.role_combo.addItems(['Doctor', 'Administrator'])
        self.role_combo.setStyleSheet(combo_style)
        form_layout.addWidget(self.role_combo)
        
        # Add User button
        button_style = """
            QPushButton {
                background-color: #6C63FF;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 12px;
                font-size: 16px;
                margin-top: 15px;
            }
            QPushButton:hover {
                background-color: #5B52FF;
            }
        """
        
        add_user_btn = QPushButton('Add User')
        add_user_btn.setStyleSheet(button_style)
        add_user_btn.clicked.connect(self.add_user)
        form_layout.addWidget(add_user_btn)
        
        # Back button
        back_btn = QPushButton('←')
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #333;
                border: none;
                font-size: 24px;
                padding: 10px;
                position: absolute;
                top: 10px;
                left: 10px;
            }
            QPushButton:hover {
                color: #6C63FF;
            }
        """)
        back_btn.clicked.connect(self.go_back)
        form_layout.addWidget(back_btn)
        
        # Add stretch to push everything to the top
        form_layout.addStretch()
        
        main_layout.addWidget(form_widget)
        
        # Add the main widget to the content layout
        self.content_layout.addWidget(main_widget)

    def add_user(self):
        """Add a new user to the database."""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm_password = self.confirm_password_input.text()
        role = self.role_combo.currentText()
        
        # Validation
        if not username or not password or not confirm_password:
            QMessageBox.warning(self, 'Error', 'Please fill in all fields')
            return
        
        if password != confirm_password:
            QMessageBox.warning(self, 'Error', 'Passwords do not match')
            return
        
        if len(password) < 6:
            QMessageBox.warning(self, 'Error', 'Password must be at least 6 characters long')
            return
        
        # Add user to database
        db = Database()
        
        if db.add_user(username, password, role):
            QMessageBox.information(self, 'Success', f'User "{username}" added successfully as {role}')
            # Clear inputs after successful addition
            self.username_input.clear()
            self.password_input.clear()
            self.confirm_password_input.clear()
        else:
            QMessageBox.warning(self, 'Error', 'Username already exists')

    def go_back(self):
        """Return to home window."""
        from .home_window import HomeWindow
        self.home_window = HomeWindow(user_role=self.user_role)
        self.home_window.show()
        self.close()

    def closeEvent(self, event):
        """Handle window close event."""
        event.accept()
