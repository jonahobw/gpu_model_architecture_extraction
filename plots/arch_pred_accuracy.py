"""
Generate plots analyzing architecture prediction accuracy based on feature selection.

This module analyzes how the number of features affects the accuracy of architecture
prediction models. It generates plots showing the relationship between the number of
features used for training and the model's accuracy on training, validation, and test sets.

The analysis includes:
- Feature ranking and selection
- Multiple experiments with different feature counts
- Statistical analysis (mean and standard deviation) across experiments
- Support for multiple model types and evaluation metrics
- GPU kernel and memory operation filtering

Example Usage:
    ```python
    # Generate feature ranking and accuracy analysis
    df = getDF(path=folder, gpu_activities_only=True)
    feature_rank = generateFeatureRank("rf", df)
    report = generateReport(df, x_axis=range(1, 51), feature_rank=feature_rank)
    plotFromReport(report, model_names=["rf", "lr"], datasets=["test"])
    ```
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append("../edge_profile")

from architecture_prediction import (
    ArchPredBase,
    RFArchPred,
    arch_model_names,
    get_arch_pred_model,
)
from config import SYSTEM_SIGNALS
from data_engineering import (
    add_indicator_cols_to_input,
    all_data,
    filter_cols,
    get_data_and_labels,
    remove_cols,
    removeColumnsFromOther,
    shared_data,
)
from experiments import predictVictimArchs
from format_profiles import parse_one_profile
from utils import latest_file

# Configure matplotlib settings
rc("font", **{"family": "serif", "serif": ["Times"], "size": 14})
rc("figure", **{"figsize": (5, 4)})


def getDF(
    path: Optional[Path] = None,
    to_keep_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    gpu_activities_only: bool = False,
) -> pd.DataFrame:
    """
    Load and preprocess profiling data for architecture prediction.

    Args:
        path (Optional[Path]): Path to the profiling data directory.
            Defaults to to_keep_path.
        to_keep_path (Optional[Path]): Path to reference data for column filtering.
            Defaults to quadro_rtx_8000/zero_exe_pretrained.
        save_path (Optional[Path]): Path to save the processed DataFrame.
            Defaults to None (no saving).
        gpu_activities_only (bool): Whether to include only GPU activity data.
            Defaults to False.

    Returns:
        pd.DataFrame: Processed DataFrame containing profiling data

    The function:
    1. Loads data from the specified path
    2. Filters columns based on reference data
    3. Optionally filters for GPU activities only
    4. Optionally saves the processed data
    """
    if to_keep_path is None:
        to_keep_path = (
            Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
        )
    if path is None:
        path = to_keep_path
    df = all_data(
        path,
        no_system_data=gpu_activities_only,
        gpu_activities_only=gpu_activities_only,
    )

    keep_df = all_data(to_keep_path)
    df = removeColumnsFromOther(keep_df, df)

    exclude_cols = SYSTEM_SIGNALS
    print(f"Number of remaining dataframe columns: {len(df.columns)}")
    if save_path is not None:
        df.to_csv(save_path)
    return df


def generateReport(
    df: pd.DataFrame,
    x_axis: List[int],
    feature_rank: List[str],
    model_names: Optional[List[str]] = None,
    model_kwargs: Dict[str, Dict] = {},
    num_experiments: int = 10,
    save_report_name: Optional[str] = None,
) -> Dict:
    """
    Generate a report analyzing model accuracy across different feature counts.

    For each model and feature count, performs multiple experiments to analyze
    the relationship between number of features and model accuracy.

    Args:
        df (pd.DataFrame): Input dataset containing model profiles
        x_axis (List[int]): List of feature counts to evaluate
        feature_rank (List[str]): Ordered list of feature names by importance
        model_names (Optional[List[str]]): List of model types to evaluate.
            Defaults to all available models.
        model_kwargs (Dict[str, Dict]): Additional arguments for each model.
            Defaults to empty dict.
        num_experiments (int): Number of experiments to run for each configuration.
            Defaults to 10.
        save_report_name (Optional[str]): Name for the output report file.
            If None, report is not saved.

    Returns:
        Dict: Report containing accuracy metrics and experiment parameters

    The report includes:
    - Mean and standard deviation of accuracy for each model and feature count
    - Training, validation, and test set performance
    - Experiment configuration parameters
    - Feature ranking information
    """
    if model_names is None:
        model_names = arch_model_names()

    report = {}
    for model_name in model_names:
        report[model_name] = {
            "train_acc": np.empty((len(x_axis), num_experiments)),
            "val_acc": np.empty((len(x_axis), num_experiments)),
            "test_acc": np.empty((len(x_axis), num_experiments)),
            "kwargs": model_kwargs.get(model_name, {}),
        }

    try:
        for i, num_features in enumerate(x_axis):
            print(f"Running {num_experiments} experiments with {num_features} features.")
            new_features = feature_rank[:num_features]
            new_df = filter_cols(df, substrs=new_features)
            for model_name in model_names:
                for exp in range(num_experiments):
                    model = get_arch_pred_model(
                        model_name, df=new_df, kwargs=report[model_name]["kwargs"]
                    )
                    report[model_name]["train_acc"][i][exp] = model.evaluateTrain()
                    report[model_name]["val_acc"][i][exp] = model.evaluateTest()
                    report[model_name]["test_acc"][i][exp] = predictVictimArchs(
                        model,
                        folder=Path.cwd() / "victim_profiles",
                        save=False,
                        topk=1,
                        verbose=False,
                    )["accuracy_k"][1]
                    if model.deterministic:
                        # For deterministic models, only need one experiment
                        report[model_name]["train_acc"][i] = np.full(
                            (num_experiments), report[model_name]["train_acc"][i][exp]
                        )
                        report[model_name]["test_acc"][i] = np.full(
                            (num_experiments), report[model_name]["test_acc"][i][exp]
                        )
                        report[model_name]["val_acc"][i] = np.full(
                            (num_experiments), report[model_name]["val_acc"][i][exp]
                        )
                        break

        # Calculate statistics for each model
        for model_name in report:
            report[model_name]["train_std"] = report[model_name]["train_acc"].std(axis=1)
            report[model_name]["train_mean"] = report[model_name]["train_acc"].mean(axis=1)
            report[model_name]["val_std"] = report[model_name]["val_acc"].std(axis=1)
            report[model_name]["val_mean"] = report[model_name]["val_acc"].mean(axis=1)
            report[model_name]["test_std"] = report[model_name]["test_acc"].std(axis=1)
            report[model_name]["test_mean"] = report[model_name]["test_acc"].mean(axis=1)
    except KeyboardInterrupt:
        pass

    # Add metadata to report
    report["feature_rank"] = feature_rank
    report["df_cols"] = list(df.columns)
    report["num_experiments"] = num_experiments
    report["x_axis"] = x_axis

    if save_report_name is not None:
        # Make numpy arrays JSON serializable
        def json_handler(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            raise TypeError(f"Unserializable object {x} of type {type(x)}")

        if not save_report_name.endswith(".json"):
            save_report_name += ".json"

        report_folder = Path(__file__).parent.absolute() / "reports"
        report_folder.mkdir(exist_ok=True)
        save_path = report_folder / save_report_name
        with open(save_path, "w") as f:
            json.dump(report, f, indent=4, default=json_handler)
    return report


def loadReport(filename: str, feature_rank: bool = False) -> Dict:
    """
    Load a report from a JSON file.

    Args:
        filename (str): Name of the report file (with or without .json extension)
        feature_rank (bool): If True, loads from feature_ranks directory.
            Defaults to False.

    Returns:
        Dict: Loaded report data

    Example:
        ```python
        report = loadReport("rf_gpu_kernels_nomem.json")
        feature_rank = loadReport("feature_rank.json", feature_rank=True)
        ```
    """
    if not filename.endswith(".json"):
        filename += ".json"

    report_path = Path(__file__).parent.absolute() / "reports" / filename
    if feature_rank:
        report_path = Path(__file__).parent.absolute() / "feature_ranks" / filename
    with open(report_path, "r") as f:
        report = json.load(f)
    return report


def generateFeatureRank(
    arch_model_name: str, df: pd.DataFrame, kwargs: Dict = {}
) -> List[str]:
    """
    Generate a ranking of features by importance using the specified model.

    Args:
        arch_model_name (str): Name of the model to use for feature ranking
        df (pd.DataFrame): Input dataset containing model profiles
        kwargs (Dict): Additional arguments for the model. Defaults to empty dict.

    Returns:
        List[str]: Ordered list of feature names by importance

    Example:
        ```python
        feature_rank = generateFeatureRank("rf", df, kwargs={"rfe_num": 1})
        ```
    """
    kwargs.update({"rfe_num": 1})
    return get_arch_pred_model(
        arch_model_name, df=df, kwargs={"rfe_num": 1, "verbose": True}
    ).featureRank(suppress_output=True)


def saveFeatureRank(
    feature_rank: List[str], metadata: Dict = {}, save_name: Optional[str] = None
) -> None:
    """
    Save feature ranking and metadata to a JSON file.

    Args:
        feature_rank (List[str]): Ordered list of feature names by importance
        metadata (Dict): Additional metadata to save with the ranking.
            Defaults to empty dict.
        save_name (Optional[str]): Name for the output file.
            Defaults to "feature_rank.json".

    Example:
        ```python
        saveFeatureRank(
            feature_rank=feature_rank,
            metadata={"model": "rf", "gpu_only": True},
            save_name="rf_gpu_features.json"
        )
        ```
    """
    if save_name is None:
        save_name = "feature_rank.json"
    elif not save_name.endswith(".json"):
        save_name += ".json"

    report = {"feature_rank": feature_rank, **metadata}
    feature_rank_folder = Path(__file__).parent.absolute() / "feature_ranks"
    feature_rank_folder.mkdir(exist_ok=True)
    save_path = feature_rank_folder / save_name
    with open(save_path, "w") as f:
        json.dump(report, f, indent=4)


def plotFromReport(
    report: Dict,
    model_names: List[str],
    datasets: Optional[List[str]] = None,
    xlim_upper: Optional[int] = None,
    save_name: Optional[str] = None,
    title: bool = True,
) -> None:
    """
    Generate plots from the accuracy report.

    Args:
        report (Dict): Report generated by generateReport
        model_names (List[str]): List of model types to plot
        datasets (Optional[List[str]]): List of datasets to plot ('train', 'val', 'test').
            Defaults to ['val'].
        xlim_upper (Optional[int]): Upper limit for x-axis. Defaults to None.
        save_name (Optional[str]): Name for the output plot file.
            Defaults to "arch_pred_acc.png".
        title (bool): Whether to include plot title. Defaults to True.

    The plot shows:
    - Accuracy vs. number of features for each model and dataset
    - Standard deviation bands around mean accuracy
    - Clear labels and legend
    - Configurable axis limits and title
    """
    if save_name is None:
        save_name = "arch_pred_acc.png"
    if datasets is None:
        datasets = ["val"]

    x_axis = report["x_axis"]
    for model_name in model_names:
        for dataset in datasets:
            label = model_name if len(datasets) == 1 else f"{model_name}_{dataset}"
            plt.plot(x_axis, report[model_name][f"{dataset}_mean"], label=label)
            minus_std = []
            plus_std = []
            for mean, std in zip(
                report[model_name][f"{dataset}_mean"],
                report[model_name][f"{dataset}_std"],
            ):
                minus_std.append(mean - std)
                plus_std.append(mean + std)
            plt.fill_between(
                x_axis,
                minus_std,
                plus_std,
                alpha=0.2,
            )

    plt.tight_layout()
    plt.xlabel("Number of Features to Train Architecture Prediction Model")

    x_axis_lim = max(x_axis) if xlim_upper is None else xlim_upper
    interval = x_axis_lim // 10
    ticks = [x for x in range(0, x_axis_lim, interval)]
    ticks[0] = 1
    ticks.append(x_axis_lim)
    plt.xticks(ticks)

    dataset_name_map = {
        "val": "Validation",
        "train": "Train",
        "test": "Test",
    }
    datasets_str = ""
    for ds in datasets:
        datasets_str += f"{dataset_name_map[ds]}/"
    plt.ylabel("Architecture Prediction Accuracy")
    if title:
        plt.title(
            f"Architecture Prediction Accuracy on {datasets_str[:-1]} Data\n"
            f"by Number of Features"
        )
    if xlim_upper is not None:
        plt.xlim(left=0, right=xlim_upper)

    plt.legend(loc=(0.68, 0.23))

    if not save_name.endswith(".png"):
        save_name += ".png"

    plt.savefig(
        Path(__file__).parent / "arch_pred_acc" / save_name,
        dpi=500,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    # Feature ranking configuration
    load_feature_rank = True  # Load features from file or generate new ones
    features_model = "rf"
    features_model_kwargs = {}
    gpu_activities_only = True
    no_memory = True
    model_kwargs = {}

    # Generate feature filename based on configuration
    features_filename = f"{features_model}"
    if gpu_activities_only:
        features_filename += "_gpu_kernels"
    if no_memory:
        features_filename += "_nomem"

    # ------------------------------------------------------------------------------

    # Report generation configuration
    load_report = True  # Load existing report or generate new one
    report_name = features_filename

    # Experiment configuration
    folder = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
    model_names = arch_model_names()
    num_experiments = 10
    x_axis = [i for i in range(1, 51)]
    x_axis.extend([i for i in range(60, 200, 10)])

    # ------------------------------------------------------------------------------

    # Plotting configuration
    plot = True
    plot_model_names = model_names
    plot_datasets = ["val"]  # ['val', 'train', 'test']
    xlim_upper = 30
    plot_save_name = f"{report_name}_{plot_datasets[-1]}"
    title = False

    # ------------------------------------------------------------------------------

    df = getDF(path=folder, gpu_activities_only=gpu_activities_only)
    if no_memory:
        df = remove_cols(df, substrs=["mem"])

    # Generate or load feature ranking
    if not load_feature_rank:
        feature_rank = generateFeatureRank(
            arch_model_name=features_model, df=df, kwargs=features_model_kwargs
        )
        saveFeatureRank(
            feature_rank=feature_rank,
            metadata={
                "features_model": features_model,
                "features_model_kwargs": features_model_kwargs,
                "df_cols": list(df.columns),
                "gpu_activities_only": gpu_activities_only,
                "no_mem": no_memory,
            },
            save_name=features_filename,
        )
    else:
        feature_rank = loadReport(features_filename, feature_rank=True)["feature_rank"]

    # Generate or load report
    if not load_report:
        report = generateReport(
            df=df,
            x_axis=x_axis,
            feature_rank=feature_rank,
            model_names=model_names,
            model_kwargs=model_kwargs,
            num_experiments=num_experiments,
            save_report_name=report_name,
        )
    else:
        report = loadReport(report_name)

    # Generate plot if requested
    if plot:
        plotFromReport(
            report=report,
            model_names=plot_model_names,
            datasets=plot_datasets,
            xlim_upper=xlim_upper,
            save_name=plot_save_name,
            title=title,
        )
