"""
Creates an exe based on model_inference.py for your OS.
This is the exe file that can be used for profiling one inference.
"""
import os
import shlex
import shutil
import site
import subprocess
import sys
from pathlib import Path


def createHiddenImportStr():
    HIDDEN_IMPORTS = [
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


def createAddDataStr():
    site_packs_folder = Path(site.getsitepackages()[0])
    pkgs = ["torch"]
    s = ""
    for pkg in pkgs:
        folder = str(site_packs_folder / pkg)
        s += f'--add-data="{folder}:." '
    return s


def createExcludeModsStr():
    exclude = ["torch.distributions"]
    s = ""
    for x in exclude:
        s += f'--exclude-module="{x}" '
    return s


def create_exe():
    command = (
        f"pyinstaller {createHiddenImportStr()}"  # {createAddDataStr()} {createExcludeModsStr()}"
        f" --onefile --clean model_inference.py"
    )
    output = subprocess.run(shlex.split(command), stdout=sys.stdout)
    exe_file = Path.cwd() / "dist" / "model_inference.exe"
    if os.name != "nt":
        # linux
        destination_folder = Path.cwd() / "exe" / "linux"
        if not destination_folder.exists():
            destination_folder.mkdir(exist_ok=True, parents=True)
        destination = destination_folder / "linux_inference.exe"
        exe_file = Path.cwd() / "dist" / "model_inference"
    else:
        # windows
        destination_folder = Path.cwd() / "exe" / "windows"
        if not destination_folder.exists():
            destination_folder.mkdir(exist_ok=True)
        destination = destination_folder / "windows_inference.exe"

    shutil.copy(exe_file, destination)


def cleanup():
    dist_folder = Path.cwd() / "dist"
    if dist_folder.exists():
        shutil.rmtree(dist_folder)

    build_folder = Path.cwd() / "build"
    if build_folder.exists():
        shutil.rmtree(build_folder)

    spec_file = Path.cwd() / "model_inference.spec"
    spec_file.unlink(missing_ok=True)


cleanup()
try:
    create_exe()
finally:
    cleanup()
