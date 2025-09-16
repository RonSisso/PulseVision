from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QComboBox, QTableWidget,
                             QTableWidgetItem, QMessageBox)
from database.db import Database

class AdminWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = Database()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('PulseVision - Admin Panel')
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create user management section
        user_section = QWidget()
        user_layout = QHBoxLayout(user_section)

        # Form for adding new users
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)

        # Username input
        username_label = QLabel('Username:')
        self.username_input = QLineEdit()
        form_layout.addWidget(username_label)
        form_layout.addWidget(self.username_input)

        # Password input
        password_label = QLabel('Password:')
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addWidget(password_label)
        form_layout.addWidget(self.password_input)

        # Role selection
        role_label = QLabel('Role:')
        self.role_combo = QComboBox()
        self.role_combo.addItems(['Doctor', 'Administrator'])
        form_layout.addWidget(role_label)
        form_layout.addWidget(self.role_combo)

        # Add user button
        add_button = QPushButton('Add User')
        add_button.clicked.connect(self.add_user)
        form_layout.addWidget(add_button)

        form_layout.addStretch()
        user_layout.addWidget(form_widget)

        # Table showing existing users
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(3)
        self.user_table.setHorizontalHeaderLabels(['Username', 'Role', 'Last Login'])
        user_layout.addWidget(self.user_table)

        layout.addWidget(user_section)

        # Navigation buttons
        button_layout = QHBoxLayout()
        
        monitor_button = QPushButton('Go to Heart Rate Monitor')
        monitor_button.clicked.connect(self.show_monitor)
        button_layout.addWidget(monitor_button)

        logout_button = QPushButton('Logout')
        logout_button.clicked.connect(self.logout)
        button_layout.addWidget(logout_button)

        layout.addLayout(button_layout)

        # Load existing users
        self.load_users()

    def add_user(self):
        """Add a new user to the database."""
        username = self.username_input.text()
        password = self.password_input.text()
        role = self.role_combo.currentText()

        if not username or not password:
            QMessageBox.warning(self, 'Error', 'Please fill in all fields')
            return

        if self.db.add_user(username, password, role):
            QMessageBox.information(self, 'Success', 'User added successfully')
            self.username_input.clear()
            self.password_input.clear()
            self.load_users()
        else:
            QMessageBox.warning(self, 'Error', 'Username already exists')

    def load_users(self):
        """Load existing users into the table."""
        cursor = self.db.conn.cursor()
        cursor.execute('SELECT username, role, last_login FROM users')
        users = cursor.fetchall()

        self.user_table.setRowCount(len(users))
        for i, user in enumerate(users):
            self.user_table.setItem(i, 0, QTableWidgetItem(user[0]))
            self.user_table.setItem(i, 1, QTableWidgetItem(user[1]))
            self.user_table.setItem(i, 2, QTableWidgetItem(str(user[2] or 'Never')))

        self.user_table.resizeColumnsToContents()

    def show_monitor(self):
        """Switch to heart rate monitor window."""
        from .main_window import MainWindow
        self.monitor_window = MainWindow()
        self.monitor_window.show()
        self.close()

    def logout(self):
        """Return to login window."""
        from .login_window import LoginWindow
        self.login_window = LoginWindow()
        self.login_window.show()
        self.close()

    def closeEvent(self, event):
        """Handle window close event."""
        event.accept() 