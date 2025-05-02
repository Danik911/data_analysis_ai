"""
Utility functions for data analysis workflow

This module provides common utility functions used across the data analysis workflow,
including enhanced JSON serialization and file handling capabilities.
"""

import os
import json
import numpy as np
import pandas as pd
import traceback
import tempfile
import shutil
from typing import Any, Dict, Optional


def json_serial(obj):
    """
    Custom JSON serializer for handling types that aren't serializable by default
    
    Args:
        obj: Object to serialize
        
    Returns:
        Serialized representation of the object
    """
    if isinstance(obj, (np.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        return str(obj)  # Convert everything else to strings


def save_json_atomic(data: Dict[str, Any], file_path: str, indent: int = 2) -> bool:
    """
    Save JSON data to file atomically to prevent corruption.
    
    Args:
        data: Data to save as JSON
        file_path: Path to save file
        indent: JSON indentation level
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            json.dump(data, temp_file, default=json_serial, indent=indent)
        
        # Verify JSON was written correctly by reading it back
        with open(temp_file.name, 'r') as f:
            # This will raise an exception if the JSON is invalid
            json.load(f)
            
        # If we get here, the JSON is valid, so move it to target location
        shutil.move(temp_file.name, file_path)
        return True
        
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {str(e)}")
        traceback.print_exc()
        
        # Clean up temporary file if it exists
        if 'temp_file' in locals() and os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception:
                pass
        return False


def read_json_safe(file_path: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Safely read a JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value to return if reading fails
        
    Returns:
        Dict: Parsed JSON data or default value
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {str(e)}")
        return default if default is not None else {}


def verify_json_file(file_path: str) -> Dict[str, Any]:
    """
    Verify if a file exists and contains valid JSON.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict containing status information about the file
    """
    status = {"exists": False, "valid": False, "error": None, "data": None}
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            status["error"] = "File does not exist"
            return status
        
        status["exists"] = True
        
        # Check if file is valid JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
            status["valid"] = True
            status["data"] = data
        
        return status
        
    except json.JSONDecodeError as e:
        status["error"] = f"Invalid JSON format: {str(e)}"
    except Exception as e:
        status["error"] = f"Error: {str(e)}"
    
    return status