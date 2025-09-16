import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QLineEdit, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from database.db import Database
from .base_window import BaseWindow

class AddPatientWindow(BaseWindow):
    def __init__(self,user_role=None):
        # Initialize with back button but no power off button
        super().__init__(show_back_button=True, show_power_off=False)
        self.user_role = user_role
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        # Create main content widget
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Left side - Image
        image_label = QLabel()
        pixmap = QPixmap(os.path.join('assets', 'add_patient.png'))
        image_label.setPixmap(pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        main_layout.addWidget(image_label)
        
        # Right side - Form
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setSpacing(20)
        form_layout.setContentsMargins(40, 40, 40, 40)
        
        # Title
        title_label = QLabel('Add Patient')
        title_label.setFont(QFont('Arial', 36))
        form_layout.addWidget(title_label)
        
        # Input fields style
        input_style = """
            QLineEdit {
                border: 2px solid #E0E0E0;
                border-radius: 25px;
                padding: 15px;
                font-size: 16px;
                margin: 10px 0;
            }
            QLineEdit:focus {
                border-color: #6C63FF;
            }
        """
        
        # First Name
        self.first_name_input = QLineEdit()
        self.first_name_input.setPlaceholderText('Enter first name')
        self.first_name_input.setStyleSheet(input_style)
        form_layout.addWidget(self.first_name_input)
        
        # Family Name
        self.family_name_input = QLineEdit()
        self.family_name_input.setPlaceholderText('Enter family name')
        self.family_name_input.setStyleSheet(input_style)
        form_layout.addWidget(self.family_name_input)
        
        # ID
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText('Enter ID')
        self.id_input.setStyleSheet(input_style)
        form_layout.addWidget(self.id_input)
        
        # Sign Up button
        button_style = """
            QPushButton {
                background-color: #6C63FF;
                color: white;
                border: none;
                border-radius: 25px;
                padding: 15px;
                font-size: 18px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #5B52FF;
            }
        """
        
        signup_btn = QPushButton('Sign Up')
        signup_btn.setStyleSheet(button_style)
        signup_btn.clicked.connect(self.add_patient)
        form_layout.addWidget(signup_btn)
        
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

    def add_patient(self):
        """Add a new patient to the database."""
        first_name = self.first_name_input.text().strip()
        family_name = self.family_name_input.text().strip()
        patient_id = self.id_input.text().strip()
        
        if not first_name or not family_name or not patient_id:
            QMessageBox.warning(self, 'Error', 'Please fill in all fields')
            return
        
        # Get current user as assigned doctor
        db = Database()
        
        if db.add_patient(patient_id, first_name, family_name):
            QMessageBox.information(self, 'Success', 'Patient added successfully')
            # Clear inputs after successful addition
            self.first_name_input.clear()
            self.family_name_input.clear()
            self.id_input.clear()
        else:
            QMessageBox.warning(self, 'Error', 'Patient ID already exists')

    def go_back(self):
        """Return to home window."""
        from .home_window import HomeWindow
        self.home_window = HomeWindow(user_role=self.user_role)  # No role needed for patient window
        self.home_window.show()
        self.close()




    def closeEvent(self, event):
        """Handle window close event."""
        event.accept() 