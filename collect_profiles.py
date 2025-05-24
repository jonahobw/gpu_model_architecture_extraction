"""Runs an executable to generate and save profiles.

This module provides functionality to collect GPU profiles by running model inference
with various configurations. It supports both PyTorch and TensorFlow models, and can
generate profiles with different input types and random seeds.

Dependencies:
    - torch: For GPU device information
    - subprocess: For running external commands
    - pathlib: For path manipulation
    - config: For configuration settings
    - format_profiles: For profile validation
    - utils: For utility functions

Example Usage:
    ```python
    # Run profiling with default settings
    python collect_profiles.py -n 10 -i 50 -gpu 0 -input random
    
    # Run profiling for specific models
    python collect_profiles.py -models resnet50 vgg16 -n 5 -i 20
    ```
"""

import argparse
import json
import random
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Tuple

import torch

import config
from format_profiles import validProfile
from utils import dict_to_str, getSystem, latest_file


def run_command(folder: Path, command: str) -> Tuple[bool, Path]:
    """Runs a command which is assumed to add a new profile to <folder>. Then validate the profile.

    Args:
        folder: Directory where the profile will be saved
        command: Command to execute for profiling

    Returns:
        Tuple of (is_valid_profile, profile_file_path)
    """
    # should be a blocking call, so the latest file is valid.
    subprocess.run(shlex.split(command), stdout=sys.stdout)
    profile_file = latest_file(folder)
    return validProfile(profile_file), profile_file


def generateExeName(use_exe: bool, use_tf: bool) -> str:
    """Generate the executable name based on configuration.

    Args:
        use_exe: Whether to use compiled executable
        use_tf: Whether to use TensorFlow

    Returns:
        Path to executable or Python command

    Raises:
        AssertionError: If both use_exe and use_tf are True
    """
    if use_tf:
        assert not use_exe
        return "python " + str(
            Path(__file__).parent / "tensorflow" / "model_inference.py"
        )
    system = getSystem()
    executable = f"exe/{system}/{system}_inference.exe"
    if not use_exe:
        # use python file instead
        executable = "python model_inference.py"
    return executable


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        required=False,
        help="number of inferences per profile. default 10",
    )
    parser.add_argument(
        "-i",
        type=int,
        default=50,
        required=False,
        help="number of profiles to run per model, default 50",
    )
    parser.add_argument(
        "-gpu",
        type=int,
        default=0,
        required=False,
        help="-1 for cpu, else number of gpu, default 0",
    )
    parser.add_argument(
        "-sleep",
        type=int,
        default=10,
        required=False,
        help="how long to sleep in between models in seconds, default 10",
    )
    parser.add_argument(
        "-input", type=str, help="Input type to pass to model. See construct_inputs.py"
    )
    parser.add_argument(
        "-pretrained", action="store_true", help="Use a pretrained model"
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=-1,
        help="If random inputs are specified and this seed is given: "
        "will generate the same inputs for every profile. "
        "Example: profile1: modelA: inputsX, modelB: inputsX, "
        "profile2: modelA: inputsX, modelB: inputsX. "
        "If random inputs are specified and this seed is not given: "
        "will generate different inputs for each profile of the same model, but different "
        "models have the same inputs.  Example: profile1: modelA: inputsX, modelB: inputsX, "
        "profile2: modelA: inputsY, modelB: inputsY.",
    )
    parser.add_argument(
        "-folder",
        type=str,
        default=None,
        help="Name of subfolder under cwd/profiles/<gpu_name>/ to save these profiles to.  "
        "Default is the date and time.",
    )
    parser.add_argument(
        "-noexe",
        action="store_true",
        help="If provided, will run the inference using the python file"
        " rather than the executable file. This is faster but "
        " is not the type of attack vector considered, so it "
        " should only be used for debugging.",
    )
    parser.add_argument(
        "-nosave", action="store_true", help="do not save any traces, just debug."
    )
    parser.add_argument(
        "-models",
        default=[],
        required=False,
        nargs="*",
        help="List of models to profile separated by spaces.  Default is all models.",
    )
    parser.add_argument(
        "-use_tf",
        action="store_true",
        help="Use tensorflow for profiling.  Requires -noexe flag as well.",
    )
    parser.add_argument(
        "-email",
        action="store_true",
        help="send emails when each model is done profiling.",
    )

    args = parser.parse_args()

    models_to_profile = config.MODELS
    if len(args.models) > 0:
        models_to_profile = args.models
        print(f"Profiling models {models_to_profile}")

    gpu_name = torch.cuda.get_device_name(0).lower().replace(" ", "_")

    # create folder for these profiles
    subfolder = args.folder
    if not subfolder:
        subfolder = time.strftime("%I%M%p_%m-%d-%y", time.gmtime())

    profile_folder = Path.cwd() / "profiles" / gpu_name / subfolder
    profile_folder.mkdir(parents=True, exist_ok=True)

    # random seeds
    i_seeds = [random.randint(0, 999999) for i in range(args.i)]

    # file to execute
    executable = generateExeName(use_exe=not args.noexe, use_tf=args.use_tf)

    # save arguments to json file
    file = profile_folder / "arguments.json"
    save_args = vars(args)
    save_args["executable"] = executable
    save_args["random_seed"] = i_seeds
    save_args["system"] = getSystem()
    save_args["folder"] = str(profile_folder)
    save_args["gpu_name"] = gpu_name

    if file.exists():
        with open(file, "r") as f:
            old_conf = json.load(f)
        for arg in old_conf:
            if arg not in ["random_seed", "models", "i"]:
                assert old_conf[arg] == save_args[arg]
    else:
        with open(file, "w") as f:
            json.dump(save_args, f, indent=4)
    start = time.time()
    for model_num, model in enumerate(models_to_profile):
        try:
            model_folder = profile_folder / model
            model_folder.mkdir(parents=True, exist_ok=True)
            log_file_prefix = model_folder / model
            iter_start = time.time()  # used for email below

            for i in range(args.i):
                print(f"Profiling {model} iteration {i+1}/{args.i}")

                if args.seed < 0:
                    # Use the seed corresponding to profile i
                    # different inputs for each profile i, although each model gets the same inputs at profile i
                    seed = i_seeds[i]
                else:
                    # each profile uses the same inputs
                    seed = args.seed

                command = (
                    f"nvprof --csv --log-file {log_file_prefix}%p.csv --system-profiling on "
                    f"--profile-child-processes {executable} -gpu {args.gpu} -model {model} -seed {seed} "
                    f"-n {args.n} -input {args.input}"
                )

                if args.pretrained:
                    command += " -pretrained"

                if args.use_tf:
                    command = (
                        f"nvprof --csv --log-file {log_file_prefix}%p.csv --system-profiling on "
                        f"--profile-child-processes {executable} -gpu {args.gpu} -model {model}"
                    )

                # sometimes nvprof fails, keep trying until it succeeds.
                success, file = run_command(model_folder, command)
                retries = 0
                while not success:
                    print(
                        f"\nNvprof failed while running command\n\n{command}\n\nretrying ... \n"
                    )
                    time.sleep(10)
                    latest_file(model_folder).unlink()
                    success, file = run_command(model_folder, command)
                    retries += 1
                    if retries > 5:
                        print("Reached 5 retries, exiting...")
                        if args.nosave:
                            shutil.rmtree(profile_folder)
                        raise RuntimeError("Nvprof failed 5 times in a row.")

                elapsed_model_time = (time.time() - iter_start) / 60  # in minutes
                avg_prof_time = elapsed_model_time / (i + 1)
                est_time = (args.i - i + 1) * avg_prof_time
                print(
                    f"Average {str(avg_prof_time)[:4]}mins per profile on {model}, "
                    f"estimated time left {str(est_time)[:4]} mins"
                )
            if args.email:
                config.EMAIL.email_update(
                    start=start,
                    iter_start=iter_start,
                    iter=model_num,
                    total_iters=len(models_to_profile),
                    subject=f"Profiles Collected for {model}",
                    params=save_args,
                )
            print("Allowing GPUs to cool between models ...")
            time.sleep(args.sleep)

        except Exception as e:
            tb = traceback.format_exc()
            config.EMAIL.email(
                f"PROGRAM CRASHED During Profile Collection for {model}",
                f"{tb}\n\n{dict_to_str(save_args)}",
            )
            raise e

        if args.nosave:
            shutil.rmtree(profile_folder)
