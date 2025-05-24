import sys
from pathlib import Path

# setting path
sys.path.append("../edge_profile")

from utils import latest_file


def removeLatestProfiles(path: Path, number: int):
    """
    path: a folder containing profiles ending with .csv
    number: how many profiles to remove
    """
    for i in range(number):
        latest_profile = latest_file(path, pattern="*.csv", oldest=True)
        if latest_profile is None:
            print(f"Only could remove {i} profiles")
            return
        assert latest_profile.exists()
        latest_file.unlink()


profile_path = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"

models = ["wide_resnet101_2", "wide_resnet50_2"]

for model in models:
    path = profile_path / model
    print(latest_file(path, oldest=True))
    # THIS DOESNT WORK YET!! TEST latest_file stuff.
