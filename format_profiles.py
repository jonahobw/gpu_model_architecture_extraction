"""
Aggregates and processes NVIDIA GPU profiling data from nvprof into structured CSV files.

This module provides functionality to:
1. Parse individual nvprof profile CSV files
2. Aggregate multiple profiles into a single dataset
3. Validate profile integrity and class balance
4. Process both GPU activities and API calls
5. Handle system metrics (clock, memory, temperature, power, fan)

Dependencies:
    - pandas: For data manipulation and CSV handling
    - numpy: For numerical operations
    - pathlib: For path handling
    - typing: For type hints

Example Usage:
    ```python
    # Parse all profiles in a directory
    parse_all_profiles("path/to/profiles")
    
    # Read aggregated data
    df = read_csv("path/to/profiles")
    
    # Validate profiles
    validate_all("path/to/profiles")
    ```

Note:
    - Profiles should be organized in subdirectories by model architecture
    - Each profile should be a valid nvprof output CSV
    - GPU activities and API calls are processed separately
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Union, Optional

import numpy as np
import pandas as pd

from get_model import name_to_family


def check_profile(profile_csv: Union[str, Path]) -> bool:
    """
    Validates if an nvprof profile CSV file was generated successfully.

    Performs two checks:
    1. Verifies the file has more than 2 lines (nvprof failure case)
    2. Checks for excessive warning lines (more than 3 lines starting with '==')

    Args:
        profile_csv: Path to the nvprof profile CSV file

    Returns:
        bool: True if profile is valid, False otherwise
    """
    with open(profile_csv, "r") as f:
        equal_line_count = 0
        for i, line in enumerate(f):
            if line.startswith("=="):
                equal_line_count += 1
                if equal_line_count > 3:
                    print(
                        f"nvprof failed for profile {profile_csv}: 3 beginning lines start with =="
                    )
                    return False  # check 2
            if i >= 5:
                return True
        print(f"nvprof failed for profile {profile_csv}, not enough lines in the file.")
    return False  # check 1


def check_for_nans(profile_csv: Union[str, Path], gpu: int = 0) -> List[str]:
    """
    Identifies columns containing NaN values in an nvprof profile.

    Checks both GPU activity data and system metrics for NaN values.

    Args:
        profile_csv: Path to the nvprof profile CSV file
        gpu: GPU device number to check (default: 0)

    Returns:
        List[str]: Names of columns containing NaN values
    """
    # aggregate gpu data first:
    skiprows = 3
    with open(profile_csv) as f:
        for i, line in enumerate(f):
            if line == "\n":
                break
    nrows = i - skiprows - 1

    df = pd.read_csv(profile_csv, header=0, skiprows=skiprows, nrows=nrows)
    df = df.drop(0)
    null_cols = df.columns[df.isna().any()].tolist()

    # system data
    skiprows = i + 2
    df = pd.read_csv(profile_csv, header=0, skiprows=skiprows, nrows=5 * (gpu + 1))
    # filter out rows with '=='
    df = df[df["Unnamed: 0"].str.contains("==") == False]
    null_system_cols = df.columns[df.isna().any()].tolist()

    null_cols.extend(null_system_cols)

    if len(null_cols) > 0:
        print(
            f"nvprof failed for profile {profile_csv}, null values in columns {null_cols}"
        )

    return null_cols


def validProfile(profile_csv: Union[str, Path], gpu: int = 0) -> bool:
    """
    Comprehensive validation of an nvprof profile.

    Combines checks for profile validity and NaN values.

    Args:
        profile_csv: Path to the nvprof profile CSV file
        gpu: GPU device number to check (default: 0)

    Returns:
        bool: True if profile is valid and contains no NaNs, False otherwise
    """
    return check_profile(profile_csv) and len(check_for_nans(profile_csv, gpu)) == 0


def add_model_family(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'model_family' column to the dataframe based on the 'model' column.

    Maps model names to their architectural families (e.g., 'wide_resnet50' -> 'resnet').

    Args:
        df: DataFrame containing a 'model' column

    Returns:
        pd.DataFrame: Original DataFrame with added 'model_family' column
    """

    def label_family(row):
        return name_to_family[row["model"]]

    df["model_family"] = df.apply(label_family, axis=1)
    return df


def parse_one_aggregate_profile(
    csv_file: Optional[Union[str, Path]] = None,
    example: bool = False,
    nrows: Optional[int] = None,
    skiprows: int = 3,
    gpu_activities_only: bool = False,
    api_calls_only: bool = False,
) -> pd.Series:
    """
    Parses a single nvprof aggregate profile CSV file into a pandas Series.

    Processes GPU activities and API calls, handling unit conversions and data cleaning.

    Args:
        csv_file: Path to the nvprof profile CSV file (generated with aggregate mode)
        example: If True, uses a debug example profile instead of csv_file
        nrows: Number of rows to read from the CSV (auto-detected if None)
        skiprows: Number of header rows to skip (default: 3)
        gpu_activities_only: If True, only process GPU activities
        api_calls_only: If True, only process API calls

    Returns:
        pd.Series: Processed profile data with metrics for each activity/call

    Raises:
        ValueError: If csv_file is not provided when example is False
        ValueError: If csv_file does not exist
    """
    if nrows is None:
        try:
            with open(csv_file) as f:
                for i, line in enumerate(f):
                    if line == "\n":
                        break
        except TimeoutError:
            raise TimeoutError(f"TimeoutError on file {csv_file}")
        nrows = i - skiprows

    if example:
        csv_file = Path.cwd() / "debug_profiles" / "resnet" / "resnet750691.csv"
    elif not csv_file:
        raise ValueError("csv_file must be provided if example is false.")

    if not csv_file.exists():
        raise ValueError(f"File {csv_file} does not exist")

    gpu_columns = {
        "Type": "type",
        "Time": "time_ms",
        "Time(%)": "time_percent",
        "Calls": "num_calls",
        "Avg": "avg_us",
        "Min": "min_us",
        "Max": "max_ms",
        "Name": "name",
    }

    gpu_prof = pd.read_csv(csv_file, header=0, skiprows=skiprows, nrows=nrows)
    gpu_prof = gpu_prof.rename(columns=gpu_columns)
    units_row = gpu_prof.iloc[0]
    gpu_prof = gpu_prof.drop(0, axis=0)  # drop the units row

    # fix the units
    for col in ["time_ms", "avg_us", "min_us", "max_ms"]:
        unit = col.split("_")[1]
        if units_row[col] != unit:
            assert units_row[col] in [
                "ms",
                "us",
            ], f"Profile {csv_file} column {col} has unit {units_row[col]}"
            # unit is wrong, since we only have us or ms, convert to the other
            if units_row[col] == "ms":
                # convert to us, multiply by 1000
                gpu_prof[col] = pd.to_numeric(gpu_prof[col]) * 1000
            else:
                # unit is in us, convert to ms, divide by 1000
                gpu_prof[col] = pd.to_numeric(gpu_prof[col]) / 1000

    assert not (gpu_activities_only and api_calls_only)

    if gpu_activities_only:
        gpu_prof = gpu_prof[gpu_prof["type"] == "GPU activities"]
    if api_calls_only:
        gpu_prof = gpu_prof[gpu_prof["type"] == "API calls"]

    attribute_cols = [
        "time_percent",
        "time_ms",
        "num_calls",
        "avg_us",
        "min_us",
        "max_ms",
    ]

    result = gpu_prof.apply(
        lambda row: retrieve_row_attrs(
            row, name_col="name", attribute_cols=attribute_cols
        ),
        axis=1,
    )  # results in sparse dataframe
    result = result.backfill()  # put all of the information in the first row

    return result.iloc[0]


def retrieve_row_attrs(
    row: pd.Series, name_col: str, attribute_cols: List[str]
) -> pd.Series:
    """
    Transforms a row of GPU profile data into a flattened Series.

    Creates a new Series with column names that combine the attribute name
    with the activity/call name.

    Args:
        row: A row from the GPU profile DataFrame
        name_col: Name of the column containing activity/call names
        attribute_cols: List of attribute columns to process

    Returns:
        pd.Series: Flattened data with combined column names

    Example:
        Input row:
            type            time_percent    time_ms     num_calls   avg_us  min_us  max_ms      name
            GPU activities  88.005407       38.058423   125         304.467 0.864   13.759156   [CUDA memcpy HtoD]

        Output Series:
            time_percent_[CUDA memcpy HtoD]    88.005407
            time_ms_[CUDA memcpy HtoD]         38.058423
            num_calls_[CUDA memcpy HtoD]       125
            avg_us_[CUDA memcpy HtoD]          304.467
            min_us_[CUDA memcpy HtoD]          0.864
            max_ms_[CUDA memcpy HtoD]          13.759156
    """
    return pd.Series(
        {
            f"{attribute}_{row[name_col]}": float(row[attribute])
            for attribute in attribute_cols
        }
    )


def parse_one_system_profile(
    csv_file: Optional[Union[str, Path]] = None,
    example: bool = False,
    nrows: int = 5,
    skiprows: Optional[int] = None,
    gpu: int = 0,
) -> pd.Series:
    """
    Parses system metrics from an nvprof profile CSV file.

    Processes metrics like clock speed, memory clock, temperature, power, and fan speed.

    Args:
        csv_file: Path to the nvprof profile CSV file
        example: If True, uses a debug example profile instead of csv_file
        nrows: Number of rows to read for each GPU (default: 5)
        skiprows: Number of header rows to skip (auto-detected if None)
        gpu: GPU device number to process (default: 0)

    Returns:
        pd.Series: Processed system metrics with min/max/avg values

    Raises:
        ValueError: If csv_file is not provided when example is False
        ValueError: If csv_file does not exist
    """
    if skiprows is None:
        with open(csv_file) as f:
            for i, line in enumerate(f):
                if line == "\n":
                    break
        skiprows = i + 2  # one blank line and one line with ==System profile result

    if example:
        csv_file = Path.cwd() / "debug_profiles" / "resnet" / "resnet750691.csv"
    elif not csv_file:
        raise ValueError("csv_file must be provided if example is false.")

    if not csv_file.exists():
        raise ValueError(f"File {csv_file} does not exist")

    system_columns = {
        "Device": "device",
        "Count": "count",
        "Avg": "avg",
        "Min": "min",
        "Max": "max",
        "Unnamed: 0": "signal",
    }

    system_prof = pd.read_csv(
        csv_file, header=0, skiprows=skiprows, nrows=nrows * (gpu + 1)
    )
    system_prof = system_prof.rename(columns=system_columns)
    system_prof["signal"] = system_prof["signal"].apply(
        lambda x: x.lower().replace(" ", "_")
    )  # format signal names

    if gpu > 0:
        # drop rows for other gpus
        system_prof = system_prof.drop(list(range(gpu * 5)))

    attribute_cols = ["avg", "min", "max"]

    result = system_prof.apply(
        lambda row: retrieve_row_attrs(
            row, name_col="signal", attribute_cols=attribute_cols
        ),
        axis=1,
    )  # results in sparse dataframe
    result = result.backfill()  # put all of the information in the first row
    return result.iloc[0]


def parse_one_profile(
    csv_file: Optional[Union[str, Path]] = None,
    example: bool = False,
    gpu: int = 0,
    remove_nans: bool = True,
    gpu_activities_only: bool = False,
    api_calls_only: bool = False,
) -> pd.Series:
    """
    Parses both GPU activities and system metrics from an nvprof profile.

    Combines data from parse_one_aggregate_profile and parse_one_system_profile.

    Args:
        csv_file: Path to the nvprof profile CSV file
        example: If True, uses a debug example profile instead of csv_file
        gpu: GPU device number to process (default: 0)
        remove_nans: If True, removes columns containing NaN values
        gpu_activities_only: If True, only process GPU activities
        api_calls_only: If True, only process API calls

    Returns:
        pd.Series: Combined GPU and system metrics data

    Raises:
        ValueError: If csv_file is not provided when example is False
        ValueError: If csv_file does not exist
    """
    csv_file = Path(csv_file)
    gpu_prof = parse_one_aggregate_profile(
        csv_file,
        example=example,
        gpu_activities_only=gpu_activities_only,
        api_calls_only=api_calls_only,
    )
    system_prof = parse_one_system_profile(csv_file, example=example, gpu=gpu)
    df = pd.concat((gpu_prof, system_prof))
    if remove_nans:
        df.dropna(inplace=True)
    return df


def avgProfiles(profile_paths: List[Path], gpu: int = 0) -> pd.Series:
    """
    Calculates average metrics across multiple nvprof profiles.

    Args:
        profile_paths: List of paths to nvprof profile CSV files
        gpu: GPU device number to process (default: 0)

    Returns:
        pd.Series: Average metrics across all profiles
    """
    combined = pd.DataFrame()
    for path in profile_paths:
        features = parse_one_profile(csv_file=path, gpu=gpu)
        features = features.to_frame().T
        combined = pd.concat((combined, features), ignore_index=True, axis=0)

    return np.mean(combined, axis=0)


def minProfiles(profile_paths: List[Path], gpu: int = 0) -> pd.Series:
    """
    Calculates minimum metrics across multiple nvprof profiles.

    Args:
        profile_paths: List of paths to nvprof profile CSV files
        gpu: GPU device number to process (default: 0)

    Returns:
        pd.Series: Minimum metrics across all profiles
    """
    combined = pd.DataFrame()
    for path in profile_paths:
        features = parse_one_profile(csv_file=path, gpu=gpu)
        features = features.to_frame().T
        combined = pd.concat((combined, features), ignore_index=True, axis=0)

    return np.min(combined, axis=0)


def parse_all_profiles(
    folder: Union[Path, str],
    save_filename: Optional[str] = None,
    gpu: int = 0,
    verbose: bool = True,
    gpu_activities_only: bool = False,
    api_calls_only: bool = False,
) -> None:
    """
    Processes all nvprof profiles in a directory structure and saves aggregated data.

    The folder structure should be:
        ./profiles/<folder>/<model_architecture>/<profile_files>.csv

    Args:
        folder: Root folder containing model architecture subfolders
        save_filename: Name for the output CSV file (default: aggregated.csv)
        gpu: GPU device number to process (default: 0)
        verbose: If True, prints progress information
        gpu_activities_only: If True, only process GPU activities
        api_calls_only: If True, only process API calls

    Raises:
        FileNotFoundError: If folder does not exist
        ValueError: If profiles are invalid, contain NaNs, or have class imbalance
    """
    # validate that no profiles are corrupt and that there is a class balance
    validate_all(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    combined = pd.DataFrame()

    for subdir in [x for x in folder.iterdir() if x.is_dir()]:
        model = subdir.name
        if verbose:
            print(f"Parsing profiles for {model}")
        for csv_profile in [x for x in subdir.iterdir()]:
            file = csv_profile.name
            if verbose:
                print(f"\t{file}")
            prof_first = pd.Series({"file": file, "model": model})
            prof_second = parse_one_profile(
                csv_file=csv_profile,
                gpu=gpu,
                gpu_activities_only=gpu_activities_only,
                api_calls_only=api_calls_only,
            )
            prof = pd.concat((prof_first, prof_second)).to_frame().T
            combined = pd.concat((combined, prof), ignore_index=True, axis=0)

    if save_filename is None:
        save_filename = "aggregated.csv"
    if gpu_activities_only:
        assert not api_calls_only
        save_filename = "aggregated_gpu_only.csv"
    if api_calls_only:
        save_filename = "aggregated_api_only.csv"

    save_path = folder / save_filename

    combined = add_model_family(combined)
    combined.to_csv(save_path, index=False)
    return


def validate_all(folder: Path) -> None:
    """
    Performs comprehensive validation of all nvprof profiles in a directory.

    Validates:
    1. Profile integrity (nvprof execution success)
    2. Data quality (absence of NaN values)
    3. Class balance (equal number of profiles per model architecture)

    Args:
        folder: Root folder containing model architecture subfolders

    Raises:
        ValueError: If any validation check fails. Also, the user will have the option to remove profiles
        based on a response to a question in the console.
    """
    # check that all profiles are valid
    valid, _ = validate_nvprof(folder, remove=False)
    if not valid:
        response = input(
            "\n\n\nThere are invalid profiles.  Enter 'yes' to delete them, anything "
            "else to keep them.  An error will be raised either way.  This error will "
            "continue occuring until they are moved or deleted."
        )
        if response.lower() == "yes":
            _ = validate_nvprof(folder, remove=True)
        raise ValueError("Invalid profiles, fix before aggregating.")

    no_nans, _ = validate_nans(folder, remove=False)
    if not no_nans:
        response = input(
            "\n\n\nThere are profiles with NaNs.  Enter 'yes' to delete them, anything "
            "else to keep them.  An error will be raised either way.  This error will "
            "continue occuring until they are fixed or deleted."
        )
        if response.lower() == "yes":
            _ = validate_nans(folder, remove=True)
        raise ValueError("Profiles have NaNs, fix before aggregating.")

    # check that classes are balanced
    balanced = validate_class_balance(folder, remove=False)
    if not balanced:
        response = input(
            "\n\n\nThere is a class imbalance. Enter 'yes' to delete extra profiles, "
            "enter anything else to keep them.  An error will be raised either way. "
            "This error will continue occuring until the classes are balanced."
        )
        if response.lower() == "yes":
            _ = validate_class_balance(folder, remove=True)
        raise ValueError("Class imbalance, fix before aggregating.")


def validate_nvprof(
    folder: Path, remove: bool = False
) -> Tuple[bool, Mapping[str, Mapping[str, Union[int, List[str]]]]]:
    """
    Validates nvprof execution success for all profiles in a directory.

    Args:
        folder: Root folder containing model architecture subfolders
        remove: If True, deletes invalid profiles

    Returns:
        Tuple containing:
            - bool: True if all profiles are valid
            - Dict: Mapping of model names to invalid profile information
                {
                    "model_name": {
                        "num_invalid": int,
                        "invalid_profiles": List[str]
                    }
                }

    Raises:
        FileNotFoundError: If folder does not exist
    """
    print("Checking profile validity ... ")

    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    all_valid = True
    invalid_profiles = {}

    for subdir in [x for x in folder.iterdir() if x.is_dir()]:
        model = subdir.name
        invalid_profiles[model] = {"num_invalid": 0, "invalid_profiles": []}
        print(f"Parsing profiles for {model}")
        for csv_profile in [x for x in subdir.iterdir()]:
            file = csv_profile.name
            valid = check_profile(csv_profile)
            if not valid:
                all_valid = False
                print(f"\t{file} is invalid!")
                invalid_profiles[model]["num_invalid"] += 1
                invalid_profiles[model]["invalid_profiles"].append(str(csv_profile))
                if remove:
                    csv_profile.unlink()

    if all_valid:
        print("All profiles valid!\n\n")
    else:
        print("Invalid profiles!")
        print(json.dumps(invalid_profiles, indent=4))
    return all_valid, invalid_profiles


def validate_class_balance(folder: Path, remove: bool = False) -> bool:
    """
    Validates equal number of profiles across all model architectures.

    Args:
        folder: Root folder containing model architecture subfolders
        remove: If True, deletes extra profiles to achieve balance

    Returns:
        bool: True if all model architectures have equal number of profiles

    Raises:
        FileNotFoundError: If folder does not exist
    """
    print("Checking class balance ... ")

    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    profiles = {}

    for subdir in [x for x in folder.iterdir() if x.is_dir()]:
        model = subdir.name
        profiles[model] = {"num": 0, "profiles": []}
        print(f"Parsing profiles for {model}")
        for csv_profile in [x for x in subdir.iterdir()]:
            profiles[model]["num"] += 1
            profiles[model]["profiles"].append(csv_profile)

    model_counts = [profiles[model]["num"] for model in profiles]
    balance = len(model_counts) == model_counts.count(model_counts[0])

    if balance:
        print("Classes are balanced!\n\n")
    else:
        print("Classes are imbalanced!")
        print(
            json.dumps(
                {model: f"{profiles[model]['num']} profiles" for model in profiles},
                indent=4,
            )
        )

    if remove:
        keep = min(model_counts)
        for model in profiles:
            count = profiles[model]["num"]
            need_to_remove = count - keep
            if need_to_remove > 0:
                for i in range(need_to_remove):
                    file = profiles[model]["profiles"][i]
                    print(f"Removing {file}")
                    file.unlink()
    return balance


def validate_nans(
    folder: Path, remove: bool = False
) -> Tuple[bool, Mapping[str, Mapping[str, Union[int, List[str]]]]]:
    """
    Validates absence of NaN values in all profiles.

    Args:
        folder: Root folder containing model architecture subfolders
        remove: If True, deletes profiles containing NaN values

    Returns:
        Tuple containing:
            - bool: True if no profiles contain NaN values
            - Dict: Mapping of model names to NaN profile information
                {
                    "model_name": {
                        "num_with_nan": int,
                        "profiles": List[str]
                    }
                }

    Raises:
        FileNotFoundError: If folder does not exist
    """
    print("Checking profiles for NaNs ... ")

    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    no_nans = True
    profiles_with_nan = {}

    for subdir in [x for x in folder.iterdir() if x.is_dir()]:
        model = subdir.name
        profiles_with_nan[model] = {"num_with_nan": 0, "profiles": []}
        print(f"Parsing profiles for {model}")
        for csv_profile in [x for x in subdir.iterdir()]:
            file = csv_profile.name
            cols = check_for_nans(csv_profile)
            if len(cols) > 0:
                no_nans = False
                print(f"\t{file} has NaNs in columns {cols}")
                profiles_with_nan[model]["num_with_nan"] += 1
                profiles_with_nan[model]["profiles"].append(str(csv_profile))
                if remove:
                    csv_profile.unlink()

    if no_nans:
        print("No NaNs in any profiles!\n\n")
    else:
        print("NaNs found in profiles!")
        print(json.dumps(profiles_with_nan, indent=4))
    return no_nans, profiles_with_nan


def read_csv(
    folder: Optional[Path] = None,
    gpu: int = 0,
    gpu_activities_only: bool = False,
    api_calls_only: bool = False,
) -> pd.DataFrame:
    """
    Reads aggregated profile data from a CSV file, creating it if necessary.

    Args:
        folder: Root folder containing model architecture subfolders
        gpu: GPU device number to process (default: 0)
        gpu_activities_only: If True, only process GPU activities
        api_calls_only: If True, only process API calls

    Returns:
        pd.DataFrame: Aggregated profile data

    Note:
        If the aggregated CSV does not exist, it will be created by parsing
        all profiles in the folder.
    """
    if not folder:
        folder = Path.cwd() / "profiles" / "debug_profiles"

    filename = "aggregated.csv"
    if gpu_activities_only:
        assert not api_calls_only
        filename = "aggregated_gpu_only.csv"
    if api_calls_only:
        filename = "aggregated_api_only.csv"

    aggregated_csv_file = folder / filename
    if not aggregated_csv_file.exists():
        parse_all_profiles(
            folder,
            gpu=gpu,
            gpu_activities_only=gpu_activities_only,
            api_calls_only=api_calls_only,
        )
    return pd.read_csv(aggregated_csv_file, index_col=False)


def combineCsv(
    profile_folders: List[Path],
    gpus: List[int],
    destination: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Combines multiple aggregated profile CSVs into a single dataset.

    Args:
        profile_folders: List of folders containing aggregated.csv files
        gpus: List of GPU device numbers corresponding to each folder
        destination: Optional path to save the combined CSV

    Returns:
        pd.DataFrame: Combined profile data

    Raises:
        AssertionError: If number of folders and GPUs don't match
    """
    assert len(profile_folders) == len(gpus)
    profiles = []
    for folder, gpu in zip(profile_folders, gpus):
        profiles.append(read_csv(folder=folder, gpu=gpu))
    result = pd.concat(profiles, ignore_index=True)
    if destination is not None:
        result.to_csv(destination, index=False)
    return result


def findProfiles(folder: Path) -> Dict[str, List[Path]]:
    """
    Finds all nvprof profile CSV files in a directory structure.

    Args:
        folder: Root folder containing model architecture subfolders

    Returns:
        Dict[str, List[Path]]: Mapping of model architecture names to lists of profile paths
    """
    result = {}
    for subdir in [x for x in folder.iterdir() if x.is_dir()]:
        architecture = subdir.name
        model_profiles = list(subdir.glob("*.csv"))
        result[architecture] = model_profiles
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process nvprof profiles")
    parser.add_argument("-folder", type=str, required=True, help="folder with profiles")
    args = parser.parse_args()
    read_csv(args.folder)
    exit(0)
