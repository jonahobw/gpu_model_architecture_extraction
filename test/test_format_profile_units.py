from pathlib import Path
import sys

# setting path
sys.path.append("../edge_profile")

from format_profiles import parse_one_aggregate_profile
from utils import latest_file

arch = "googlenet"
profile = "googlenet2732560.csv"
profile = (
    Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained" / arch / profile
)

features = parse_one_aggregate_profile(csv_file=profile)
print(features)
