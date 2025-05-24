"""Creates an executable based on tensorflow_inference.py for your OS.

This script provides functionality to create a standalone executable from tensorflow_inference.py
using PyInstaller. It handles platform-specific paths, hidden imports, and cleanup.
The executable can be used for profiling single model inferences.

Dependencies:
    - pyinstaller: For creating the executable
    - site: For finding site-packages
    - shutil: For file operations
    - subprocess: For running PyInstaller

Usage: Run this script to create an executable for your OS.
"""

import os
import shlex
import shutil
import site
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def createHiddenImportStr() -> str:
    """Create PyInstaller hidden imports string.
    
    Returns:
        String containing all hidden imports for PyInstaller
    """
    HIDDEN_IMPORTS: List[str] = [
        "sklearn.utils._typedefs",
        "sklearn.utils._heap",
        "sklearn.utils._sorting",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors.quad_tree",
        "sklearn.tree._utils",
        "sklearn.neighbors._typedefs",
        "sklearn.utils._typedefs",
        "sklearn.neighbors._partition_nodes",
        "sklearn.utils._vector_sentinel",
        "sklearn.metrics.pairwise",
        "sklearn.metrics._pairwise_distances_reduction._datasets_pair",
        "sklearn.metrics._pairwise_distances_reduction",
        # "torch",
        # "torchvision",
        # "torch.jit",
    ]
    s = ""
    for imprt in HIDDEN_IMPORTS:
        s += f'--hidden-import="{imprt}" '
    return s


def createAddDataStr() -> str:
    """Create PyInstaller add-data string for required packages.
    
    Returns:
        String containing add-data arguments for PyInstaller
    """
    site_packs_folder = Path(site.getsitepackages()[0])
    pkgs: List[str] = ["torch"]
    s = ""
    for pkg in pkgs:
        folder = str(site_packs_folder / pkg)
        s += f'--add-data="{folder}:." '
    return s


def createExcludeModsStr() -> str:
    """Create PyInstaller exclude-modules string.
    
    Returns:
        String containing exclude-module arguments for PyInstaller
    """
    exclude: List[str] = ["torch.distributions"]
    s = ""
    for x in exclude:
        s += f'--exclude-module="{x}" '
    return s


def create_exe() -> None:
    """Create executable using PyInstaller.
    
    This function:
    1. Runs PyInstaller with necessary arguments
    2. Copies the executable to the appropriate platform-specific folder
    3. Handles both Windows and Linux paths
    
    Raises:
        subprocess.CalledProcessError: If PyInstaller fails
        FileNotFoundError: If PyInstaller executable is not found
    """
    command = (
        f"pyinstaller "  # {createHiddenImportStr()}"# {createAddDataStr()} {createExcludeModsStr()}"
        f" --onefile --clean tensorflow_inference.py"
    )
    output = subprocess.run(shlex.split(command), stdout=sys.stdout)
    exe_file = Path.cwd() / "dist" / "tensorflow_inference.exe"
    if os.name != "nt":
        # linux
        destination_folder = Path.cwd() / "exe" / "linux"
        if not destination_folder.exists():
            destination_folder.mkdir(exist_ok=True, parents=True)
        destination = destination_folder / "linux_inference.exe"
        exe_file = Path.cwd() / "dist" / "tensorflow_inference"
    else:
        # windows
        destination_folder = Path.cwd() / "exe" / "windows"
        if not destination_folder.exists():
            destination_folder.mkdir(exist_ok=True)
        destination = destination_folder / "windows_inference.exe"

    shutil.copy(exe_file, destination)


def cleanup() -> None:
    """Clean up temporary files created by PyInstaller.
    
    Removes:
        - dist/ directory
        - build/ directory
        - tensorflow_inference.spec file
    """
    dist_folder = Path.cwd() / "dist"
    if dist_folder.exists():
        shutil.rmtree(dist_folder)

    build_folder = Path.cwd() / "build"
    if build_folder.exists():
        shutil.rmtree(build_folder)

    spec_file = Path.cwd() / "tensorflow_inference.spec"
    spec_file.unlink(missing_ok=True)


# Main execution
cleanup()
try:
    create_exe()
finally:
    cleanup()
