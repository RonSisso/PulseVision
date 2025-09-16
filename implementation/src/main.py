import sys
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication

from gui.login_window import LoginWindow

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'app.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point of the application."""
    try:
        # Set up logging
        setup_logging()
        logging.info("Starting PulseVision application")
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create and show login window
        window = LoginWindow()
        window.show()
        
        # Start Qt event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 