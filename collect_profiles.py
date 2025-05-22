"""
Runs an executable to generate and save profiles.

Some parameters come from command line and some from config.py.  TODO - make everything come from config.
"""

import json
import os
import shutil
import sys
import subprocess
from pathlib import Path
import argparse
import shlex
import time
import random
import traceback
from typing import List

import torch

import config
from utils import getSystem, latest_file, dict_to_str
from format_profiles import validProfile


def run_command(folder, command):
    """Runs a command which is assumed to add a new profile to <folder>.  Then validate the profile."""
    # should be a blocking call, so the latest file is valid.
    output = subprocess.run(shlex.split(command), stdout=sys.stdout)
    profile_file = latest_file(folder)
    return validProfile(profile_file), profile_file


def run_command_popen(folder, command, model_type):
    """
    DOESN'T WORK, USE run_command() INSTEAD. Uses subprocess.Popen() instead of subprocess.run() to get the process id.

    The reason this does not work is because the command run is nvprof, and nvprof starts another process
    which is the actual executable with another process id.
    """
    process = subprocess.Popen(shlex.split(command))
    process.wait()

    process_id = process.pid
    profile_file = folder / f"{model_type}{process_id}.csv"
    if not profile_file.exists():
        raise FileNotFoundError(
            f"File {profile_file} does not exist and cannot be validated."
        )
    return validProfile(profile_file), profile_file


def generateExeName(use_exe: bool, use_tf: bool) -> str:
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
