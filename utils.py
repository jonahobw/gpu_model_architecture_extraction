"""
Utility functions for file operations, timing, and dictionary handling.

This module provides various utility functions for:
- Time formatting and conversion
- System detection
- File operations (finding latest/oldest files)
- Dictionary operations and validation
- JSON serialization

Example Usage:
    ```python
    # Format time in seconds to HH:MM:SS
    formatted_time = timer(3661)  # Returns "01:01:01"

    # Find latest file in a directory
    latest = latest_file(Path("./data"), "*.json")

    # Convert dictionary to formatted string
    config_str = dict_to_str({"model": "resnet", "epochs": 100})
    ```
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def timer(time_in_s: float) -> str:
    """
    Convert time in seconds to HH:MM:SS format.

    Args:
        time_in_s (float): Time in seconds to convert

    Returns:
        str: Time formatted as "HH:MM:SS" with leading zeros

    Example:
        ```python
        formatted = timer(3661)  # Returns "01:01:01"
        ```
    """
    hours, rem = divmod(time_in_s, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def getSystem() -> str:
    """
    Detect the operating system.

    Returns:
        str: "linux" for Unix-like systems, "windows" for Windows systems

    Example:
        ```python
        system = getSystem()  # Returns "linux" or "windows"
        ```
    """
    if os.name != "nt":
        system = "linux"
    else:
        system = "windows"
    return system


def latest_file(
    path: Path,
    pattern: str = "*",
    oldest: bool = False
) -> Optional[Path]:
    """
    Find the latest or oldest file in a directory matching a pattern.

    Args:
        path (Path): Directory to search in
        pattern (str): Glob pattern to match files. Defaults to "*".
        oldest (bool): If True, return oldest file instead of latest.
            Defaults to False.

    Returns:
        Optional[Path]: Path to the latest/oldest file, or None if no files found

    Example:
        ```python
        # Find latest JSON file
        latest = latest_file(Path("./data"), "*.json")
        
        # Find oldest log file
        oldest = latest_file(Path("./logs"), "*.log", oldest=True)
        ```
    """
    files = [x for x in path.glob(pattern)]
    if len(files) == 0:
        print(f"Warning: no files with pattern {pattern} found in folder {path}")
        return None
    return latestFileFromList(files, oldest=oldest)


def latestFileFromList(paths: List[Path], oldest: bool = False) -> Path:
    """
    Find the latest or oldest file from a list of paths.

    Args:
        paths (List[Path]): List of file paths to check
        oldest (bool): If True, return oldest file instead of latest.
            Defaults to False.

    Returns:
        Path: Path to the latest/oldest file

    Note:
        Uses file creation time (st_ctime) to determine latest/oldest.
        Source: https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder

    Example:
        ```python
        files = [Path("file1.txt"), Path("file2.txt")]
        latest = latestFileFromList(files)
        ```
    """
    if oldest:
        return min(paths, key=lambda x: x.stat().st_ctime)
    return max(paths, key=lambda x: x.stat().st_ctime)


def dict_to_str(dictionary: Dict[str, Any], indent: int = 4) -> str:
    """
    Convert a dictionary to a formatted string with proper JSON serialization.

    Args:
        dictionary (Dict[str, Any]): Dictionary to convert
        indent (int): Number of spaces for indentation. Defaults to 4.

    Returns:
        str: Formatted string representation of the dictionary

    Note:
        Handles non-serializable objects by converting them to strings.
        If string conversion fails, includes type information in the output.

    Example:
        ```python
        config = {"model": "resnet", "epochs": 100}
        config_str = dict_to_str(config)
        # Returns:
        # {
        #     "model": "resnet",
        #     "epochs": 100
        # }
        ```
    """
    def default(x: Any) -> str:
        try:
            res = str(x)
            return res
        except ValueError:
            pass
        return f"JSON Parse Error for Object of type: {type(x)}"

    return json.dumps(dictionary, indent=indent, default=default)


def checkDict(config: Dict[str, Any], args_dict: Dict[str, Any]) -> bool:
    """
    Recursively check if a configuration dictionary contains all required arguments.

    This function verifies that all keys and values in args_dict exist in config,
    supporting nested dictionaries at any depth.

    Args:
        config (Dict[str, Any]): Configuration dictionary to check against
        args_dict (Dict[str, Any]): Dictionary of required arguments to check for

    Returns:
        bool: True if all arguments exist in config with matching values,
              False otherwise

    Example:
        ```python
        config = {
            "model": {
                "name": "resnet",
                "layers": 50
            }
        }
        required = {
            "model": {
                "name": "resnet"
            }
        }
        is_valid = checkDict(config, required)  # Returns True
        ```
    """
    for arg in args_dict:
        if arg not in config:
            return False
        elif isinstance(args_dict[arg], dict):
            if not isinstance(config[arg], dict) or not checkDict(
                config[arg], args_dict[arg]
            ):
                return False
        elif args_dict[arg] != config[arg]:
            return False
    return True
