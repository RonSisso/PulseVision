# Asset Management System

This document explains how the asset management system works in the PulseVision application, ensuring that images work correctly for everyone who copies the repository.

## Overview

The application uses a centralized asset management system that automatically resolves image paths regardless of where the project is located on the file system. This ensures that images work correctly for all users, regardless of their operating system or where they've cloned the repository.

## File Structure

```
implementation/
├── assets/                    # All image assets are stored here
│   ├── login.png
│   ├── menu.png
│   ├── back_button.png
│   └── add_patient.png
├── src/
│   ├── utils/
│   │   └── path_utils.py      # Asset path resolution utilities
│   └── gui/                   # GUI modules that use assets
└── ...
```

## How It Works

### 1. Path Utilities (`src/utils/path_utils.py`)

The `path_utils.py` module provides several functions for handling asset paths:

- **`get_project_root()`**: Returns the absolute path to the `implementation` directory
- **`get_assets_path(filename)`**: Returns the full path to an asset file
- **`get_assets_path_for_html(filename)`**: Returns a `file://` URL for use in HTML
- **`asset_exists(filename)`**: Checks if an asset file exists

### 2. Cross-Platform Compatibility

The system automatically handles:
- **Windows**: Uses backslashes (`\`) in file paths
- **Linux/macOS**: Uses forward slashes (`/`) in file paths
- **HTML URLs**: Converts paths to proper `file://` URLs with forward slashes

### 3. Usage in GUI Modules

All GUI modules now import and use the path utilities:

```python
from utils.path_utils import get_assets_path, get_assets_path_for_html

# For PyQt5 QPixmap
pixmap = QPixmap(get_assets_path('menu.png'))

# For HTML templates
image_url = get_assets_path_for_html('login.png')
```

## Benefits

1. **Portability**: Images work regardless of where the project is cloned
2. **Cross-Platform**: Works on Windows, Linux, and macOS
3. **Maintainability**: Centralized path management
4. **Reliability**: Automatic path resolution prevents broken image links

## Adding New Assets

To add new image assets:

1. Place the image file in the `assets/` directory
2. Use the path utilities in your code:
   ```python
   from utils.path_utils import get_assets_path
   pixmap = QPixmap(get_assets_path('your_new_image.png'))
   ```

## Troubleshooting

If images don't appear:

1. Check that the image file exists in the `assets/` directory
2. Verify the filename is spelled correctly (case-sensitive)
3. Ensure the image file is not corrupted
4. Check the console for any path-related error messages

## Technical Details

The system works by:
1. Finding the current file's directory (`src/utils/path_utils.py`)
2. Going up two levels to reach the `implementation` directory
3. Appending `assets/` and the filename
4. Converting to the appropriate format for the use case (file path or URL)

This approach ensures that the asset paths are always resolved correctly, regardless of the current working directory when the application is run.
