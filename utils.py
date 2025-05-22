import json
import os
from pathlib import Path
from typing import List


def timer(time_in_s: float) -> str:
    hours, rem = divmod(time_in_s, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def getSystem() -> str:
    if os.name != "nt":
        system = "linux"
    else:
        system = "windows"
    return system


def latest_file(path: Path, pattern: str = "*", oldest: bool = False) -> Path:
    """Return the latest file in the folder <path>.  If oldest is true, return oldest file."""
    files = [x for x in path.glob(pattern)]
    if len(files) == 0:
        print(f"Warning: no files with pattern {pattern} found in folder {path}")
        return None
    return latestFileFromList(files, oldest=oldest)


def latestFileFromList(paths: List[Path], oldest: bool = False) -> Path:
    """
    Given a list of paths, return the path of the latest file.
    If oldest is true, return oldest file.
    """
    # source https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder
    if oldest:
        return min(paths, key=lambda x: x.stat().st_ctime)
    return max(paths, key=lambda x: x.stat().st_ctime)


def dict_to_str(dictionary, indent: int = 4) -> str:
    def default(x):
        try:
            res = str(x)
            return res
        except ValueError:
            pass
        return f"JSON Parse Error for Object of type: {type(x)}"

    return json.dumps(dictionary, indent=indent, default=default)


def checkDict(config: dict, args_dict: dict) -> bool:
    """
    Recursive function for checking that arguments in
    a dictionary match a config. supports multilevel dicts.
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
