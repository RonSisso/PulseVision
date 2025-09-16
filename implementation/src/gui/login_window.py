import os
import logging
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, pyqtSlot, QObject, QTimer
from PyQt5.QtWebChannel import QWebChannel
from .home_window import HomeWindow
from database.db import Database
from .base_window import BaseWindow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

class WebBridge(QObject):
    """Bridge class for communication between JavaScript and Python."""
    def __init__(self, login_window):
        super().__init__()
        self.login_window = login_window
        self.logger = logging.getLogger(__name__)
        try:
            self.db = Database()
            self.logger.info("Database connection initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            self.db = None
            QMessageBox.critical(None, "Database Error", 
                               f"Failed to connect to database: {str(e)}\n\n"
                               "Please check if all required packages are installed.")

    @pyqtSlot(str, str)
    def login(self, username, password):
        """Handle login attempt."""
        try:
            if not self.db:
                error_msg = "Database connection not available"
                self.logger.error(error_msg)
                self.show_error(error_msg)
                return
                
            self.logger.info(f"Login attempt for user: {username}")
            success, role = self.db.verify_user(username, password)
            
            if success:
                self.logger.info(f"Login successful for user: {username}")
                QTimer.singleShot(0, lambda: self.login_window.show_home_window(role))
            else:
                error_msg = "Invalid username or password"
                self.logger.warning(f"Login failed for user: {username}")
                self.show_error(error_msg)
                
        except Exception as e:
            error_msg = f"Login error: {str(e)}"
            self.logger.error(error_msg)
            self.show_error(error_msg)

    def show_error(self, message):
        """Show error message in the web view."""
        try:
            js_code = f'document.getElementById("errorMessage").textContent = "{message}";'
            js_code += 'document.getElementById("errorMessage").style.display = "block";'
            js_code += 'document.querySelector(".login-button").disabled = false;'
            js_code += 'document.querySelector(".login-button").textContent = "Login";'
            js_code += 'window.loginInProgress = false;'
            self.login_window.web_view.page().runJavaScript(js_code)
            self.logger.debug(f"Error message displayed: {message}")
        except Exception as e:
            error_msg = f"Error displaying message: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self.login_window, "Error", error_msg)

class LoginWindow(BaseWindow):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Initialize with no back button but with power off button
        super().__init__(show_back_button=False, show_power_off=True)
        self.home_window = None
        self.web_view = None
        self.bridge = None
        self.channel = None
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        try:
            self.init_ui()
            self.logger.info("Login window initialized successfully")
        except Exception as e:
            error_msg = f"Error initializing login window: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def init_ui(self):
        """Initialize the user interface."""
        try:
            # Create web view
            self.web_view = QWebEngineView()
            self.content_layout.addWidget(self.web_view)

            # Set up web channel for JS-Python communication
            self.channel = QWebChannel()
            self.bridge = WebBridge(self)
            self.channel.registerObject('backend', self.bridge)
            self.web_view.page().setWebChannel(self.channel)

            # Load the HTML file
            html_path = os.path.join(os.path.dirname(__file__), 'templates', 'login.html')
            if not os.path.exists(html_path):
                raise FileNotFoundError(f"Login template not found at {html_path}")
                
            # Update image path in HTML
            with open(html_path, 'r') as f:
                html_content = f.read()
            
            # Replace the image path with the correct absolute path
            assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets'))
            login_image_path = os.path.join(assets_dir, 'login.png')
            if os.path.exists(login_image_path):
                normalized_path = login_image_path.replace("\\", "/")
                html_content = html_content.replace(
                    'file:///C:/Users/nadav/Desktop/final_project/assets/login.png',
                    f'file:///{normalized_path}'
                )
            
            # Scale the UI elements based on screen size
            html_content = html_content.replace('font-size: 1.5rem', f'font-size: {int(self.screen_height * 0.03)}px')
            html_content = html_content.replace('font-size: 1rem', f'font-size: {int(self.screen_height * 0.02)}px')
            html_content = html_content.replace('font-size: 0.9rem', f'font-size: {int(self.screen_height * 0.018)}px')
            html_content = html_content.replace('padding: 1rem 2rem', f'padding: {int(self.screen_height * 0.02)}px {int(self.screen_width * 0.02)}px')
            html_content = html_content.replace('padding: 0.75rem', f'padding: {int(self.screen_height * 0.015)}px')
            html_content = html_content.replace('width: 400px', f'width: {int(self.screen_width * 0.3)}px')
            html_content = html_content.replace('width: 50%', f'width: {int(self.screen_width * 0.4)}px')
            html_content = html_content.replace('max-width: 600px', f'max-width: {int(self.screen_width * 0.4)}px')
            
            # Load the modified HTML content
            self.web_view.setHtml(html_content, QUrl.fromLocalFile(os.path.dirname(html_path)))
            
        except Exception as e:
            print(f"Error in init_ui: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error initializing UI: {str(e)}")
            self.close()

    def show_home_window(self, role):
        """Show the home window and close login window."""
        try:
            if self.home_window is None:
                self.home_window = HomeWindow(user_role=role)
            self.home_window.show()
            self.close()
        except Exception as e:
            print(f"Error opening home window: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error opening home window: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event."""
        try:
            event.accept()
        except Exception as e:
            print(f"Error during window close: {str(e)}")
            event.accept() 