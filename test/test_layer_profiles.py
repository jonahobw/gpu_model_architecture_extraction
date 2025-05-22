"""
This module will run single PyTorch operations and
generate profiles on them.
"""
import argparse
import json
import sys
from pathlib import Path
import time

from utils import latest_file

# setting path
sys.path.append("../edge_profile")

import torch

from collect_profiles import run_command
from construct_input import construct_input


def runNVProf(
    path: Path,
    operation: str,
    seed: int = 47,
    n: int = 10,
    input: str = "0",
    gpu=0,
):
    """
    Adds a profile file profile_{pid}.csv and associated params_{pid}.json file to
    the folder <path> if the profile succeeded.
    Note - this function does not check for collisions in pid.
    """

    path.mkdir(exist_ok=True)
    prefix = path / "profile_"
    executable = Path(__file__)
    print(f"Using executable {executable.name} for nvprof")
    command = (
        f"nvprof --csv --log-file {prefix}%p.csv --system-profiling on "
        f"--profile-child-processes {executable} -profile -gpu {gpu} "
        f" -seed {seed} -n {n} -input {input} -operation {operation}"
    )

    print(f"\nCommand being run:\n{command}\n\n")

    success, file = run_command(path, command)
    retries = 0
    print(f"{'Success' if success else 'Failure'} on file {file}")
    while not success:
        print("\nNvprof retrying ... \n")
        time.sleep(10)
        profile_file = latest_file(path, "profile_")
        if profile_file is not None and profile_file.exists():
            profile_file.unlink()
        success, file = run_command(path, command)
        retries += 1
        if retries > 5:
            print("Reached 5 retries, exiting...")
            break
    if not success:
        latest_file(path, "profile_").unlink()
        raise RuntimeError("Nvprof failed 5 times in a row.")
    profile_num = str(file.name).split("_")[1].split(".")[0]
    params = {
        "file": str(file),
        "profile_number": profile_num,
        "seed": seed,
        "n": n,
        "input": input,
        "success": success,
        "gpu": gpu,
        "gpu_type": torch.cuda.get_device_name(0).lower().replace(" ", "_"),
        "operation": operation,
    }
    with open(path / f"params_{profile_num}.json", "w") as f:
        json.dump(params, f, indent=4)


def executeOperation(
    operation: str, seed: int = 47, n: int = 10, input: str = "0", gpu=0
):
    device = torch.device(f"cuda:{gpu}")
    dev_name = f"gpu{gpu}"

    print(f"Running {n} {operation} operations on {dev_name}...")

    for i in range(n):
        inputs = construct_input(type=input, number=1, seed=seed)
        inputs = inputs.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-profile",
        type=bool,
        action="store_true",
        required=False,
        help="If provided, will invoke the profiler, this option is only provided by the program, "
        "not the user.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        required=False,
        help="number of inferences per operation. default 10",
    )
    parser.add_argument(
        "-gpu",
        type=int,
        default=0,
        required=False,
        help="number of gpu, default 0",
    )
    parser.add_argument(
        "-sleep",
        type=int,
        default=1,
        required=False,
        help="how long to sleep in between models in seconds, default 1",
    )
    parser.add_argument(
        "-input", type=str, help="Input type to pass to model. See construct_inputs.py"
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=47,
        help="random seed",
    )
    parser.add_argument(
        "-folder",
        type=str,
        default=None,
        help="Name of subfolder under cwd/operation_profiles/<gpu_name>/ to save these profiles to.  "
        "Default is the date and time.",
    )

    args = parser.parse_args()

    if args.profile:
        executeOperation(
            operation=args.operation,
            seed=args.seed,
            n=args.n,
            input=args.input,
            gpu=args.gpu,
        )
        exit(0)

    gpu_name = torch.cuda.get_device_name(0).lower().replace(" ", "_")

    # create folder for these profiles
    subfolder = args.folder
    if not subfolder:
        subfolder = time.strftime("%I%M%p_%m-%d-%y", time.gmtime())

    profile_folder = Path.cwd() / "operation_profiles" / gpu_name / subfolder
    profile_folder.mkdir(parents=True, exist_ok=True)

    OPERATIONS = []

    for op in OPERATIONS:
        runNVProf(
            path=profile_folder,
            operation=op,
            seed=args.seed,
            n=args.n,
            input=args.input,
            gpu=args.gpu,
        )
        time.sleep(args.sleep)
