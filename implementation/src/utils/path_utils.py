"""
Utility functions for handling file paths in a cross-platform way.
"""
import os
import sys


def get_project_root():
    """
    Get the project root directory (implementation folder).
    This works regardless of where the script is run from.
    """
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to get to src, then up one more to get to implementation
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    return project_root


def get_assets_path(filename):
    """
    Get the full path to an asset file.
    
    Args:
        filename (str): The name of the asset file (e.g., 'login.png')
    
    Returns:
        str: The full path to the asset file
    """
    project_root = get_project_root()
    assets_dir = os.path.join(project_root, 'assets')
    return os.path.join(assets_dir, filename)


def get_assets_path_for_html(filename):
    """
    Get the file:// URL path for an asset file to use in HTML.
    
    Args:
        filename (str): The name of the asset file (e.g., 'login.png')
    
    Returns:
        str: The file:// URL path for the asset
    """
    asset_path = get_assets_path(filename)
    # Convert to file:// URL format
    normalized_path = asset_path.replace("\\", "/")
    return f'file:///{normalized_path}'


def asset_exists(filename):
    """
    Check if an asset file exists.
    
    Args:
        filename (str): The name of the asset file
    
    Returns:
        bool: True if the file exists, False otherwise
    """
    return os.path.exists(get_assets_path(filename))
