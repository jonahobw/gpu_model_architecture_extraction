"""
Generate plots analyzing architecture prediction accuracy based on dataset size.

This module analyzes how the size of the training dataset affects the accuracy of
architecture prediction models. It generates plots showing the relationship between
the number of profiles per architecture in the training dataset and the model's
accuracy on training, validation, and test sets.

The analysis includes:
- Multiple experiments with different dataset sizes
- Statistical analysis (mean and standard deviation) across experiments
- Support for multiple model types and evaluation metrics
- Configurable dataset size ranges and step sizes

Example Usage:
    ```python
    # Generate and plot accuracy analysis for all models
    folder = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
    df = getDF(path=folder)
    report = generateReport(df, num_experiments=10)
    plotFromReport(report, model_names=["rf", "lr"], datasets=["test"])
    ```
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

# Add parent directory to path for imports
sys.path.append("../edge_profile")

from arch_pred_accuracy import getDF
from architecture_prediction import arch_model_names, get_arch_pred_model
from data_engineering import filter_cols
from experiments import predictVictimArchs

# Configure matplotlib settings
rc("font", **{"family": "serif", "serif": ["Times"], "size": 14})
rc("figure", **{"figsize": (5, 4)})

# Define output directories
REPORT_FOLDER = Path(__file__).parent.absolute() / "arch_pred_acc_by_dataset_size"
REPORT_FOLDER.mkdir(exist_ok=True)
plot_folder = REPORT_FOLDER / "plots"
plot_folder.mkdir(exist_ok=True)


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

    report_path = REPORT_FOLDER / filename
    if feature_rank:
        report_path = Path(__file__).parent.absolute() / "feature_ranks" / filename
    with open(report_path, "r") as f:
        report = json.load(f)
    return report


def generateReport(
    df: pd.DataFrame,
    model_names: Optional[List[str]] = None,
    model_kwargs: Dict[str, Dict] = {},
    num_experiments: int = 10,
    step_size: int = 1,
    dataset_start: int = 10,
    dataset_cap: int = 38,
    save_report_name: Optional[str] = None,
) -> Dict:
    """
    Generate a report analyzing model accuracy across different dataset sizes.

    For each model and dataset size, performs multiple experiments to analyze
    the relationship between training data size and model accuracy.

    Args:
        df (pd.DataFrame): Input dataset containing model profiles
        model_names (Optional[List[str]]): List of model types to evaluate.
            Defaults to all available models.
        model_kwargs (Dict[str, Dict]): Additional arguments for each model.
            Defaults to empty dict.
        num_experiments (int): Number of experiments to run for each configuration.
            Defaults to 10.
        step_size (int): Increment between dataset sizes. Defaults to 1.
        dataset_start (int): Starting number of profiles per architecture.
            Defaults to 10.
        dataset_cap (int): Maximum number of profiles per architecture.
            Defaults to 38.
        save_report_name (Optional[str]): Name for the output report file.
            If None, report is not saved.

    Returns:
        Dict: Report containing accuracy metrics and experiment parameters

    The report includes:
    - Mean and standard deviation of accuracy for each model and dataset size
    - Training, validation, and test set performance
    - Experiment configuration parameters
    - Feature ranking information
    """
    if save_report_name is not None and not save_report_name.endswith(".json"):
        save_report_name += ".json"

    if model_names is None:
        model_names = arch_model_names()

    x_axis = [x for x in range(dataset_start, dataset_cap + 1, step_size)]
    assert max(x_axis) <= len(df) / len(df["model"].unique())

    report = {}
    for model_name in model_names:
        report[model_name] = {
            "train_acc": np.empty((len(x_axis), num_experiments)),
            "val_acc": np.empty((len(x_axis), num_experiments)),
            "test_acc": np.empty((len(x_axis), num_experiments)),
            "kwargs": model_kwargs.get(model_name, {}),
        }

    # Run experiments for each dataset size
    for i, dataset_size in enumerate(x_axis):
        print(
            f"Running {num_experiments} experiments with {dataset_size} dataset size."
        )
        for model_name in model_names:
            kwargs = report[model_name]["kwargs"].copy()
            kwargs["train_size"] = dataset_size * len(df["model"].unique())
            kwargs["test_size"] = len(df) - kwargs["train_size"]
            for exp in range(num_experiments):
                model = get_arch_pred_model(model_name, df=df, kwargs=kwargs)
                assert dataset_size == len(model.x_tr) / model.num_classes
                report[model_name]["train_acc"][i][exp] = model.evaluateTrain()
                report[model_name]["val_acc"][i][exp] = model.evaluateTest()
                report[model_name]["test_acc"][i][exp] = predictVictimArchs(
                    model, folder=Path.cwd() / "victim_profiles", save=False, topk=1
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

    # Add metadata to report
    report["feature_rank"] = feature_rank
    report["df_cols"] = list(df.columns)
    report["num_experiments"] = num_experiments
    report["step_size"] = step_size
    report["dataset_cap"] = dataset_cap
    report["x_axis"] = x_axis

    if save_report_name is not None:
        # Make numpy arrays JSON serializable
        def json_handler(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            raise TypeError(f"Unserializable object {x} of type {type(x)}")

        save_path = REPORT_FOLDER / save_report_name
        with open(save_path, "w") as f:
            json.dump(report, f, indent=4, default=json_handler)
    return report


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
            Defaults to "arch_pred_acc_by_dataset_size.png".
        title (bool): Whether to include plot title. Defaults to True.

    The plot shows:
    - Accuracy vs. dataset size for each model and dataset
    - Standard deviation bands around mean accuracy
    - Clear labels and legend
    - Configurable axis limits and title
    """
    if save_name is None:
        save_name = "arch_pred_acc_by_dataset_size.png"
    if not save_name.endswith(".png"):
        save_name += ".png"
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
    plt.legend(loc="lower right")
    plt.xlabel("Number of Profiles per Architecture in Training Dataset")

    x_axis_lim = max(x_axis) if xlim_upper is None else xlim_upper
    interval = x_axis_lim // 7
    ticks = [x for x in range(0, x_axis_lim, interval)]
    ticks[0] = 1
    ticks.append(x_axis_lim)
    plt.xticks(ticks)

    plt.ylabel("Architecture Prediction Accuracy")
    dataset_name_map = {
        "val": "Validation",
        "train": "Train",
        "test": "Test",
    }
    datasets_str = ""
    for ds in datasets:
        datasets_str += f"{dataset_name_map[ds]}/"
    if title:
        plt.title(
            f"Architecture Prediction Accuracy on {datasets_str[:-1]} Data\n"
            f"by Number of Profiles per Architecture in Train Dataset"
        )
    if xlim_upper is not None:
        plt.xlim(right=xlim_upper)
    plt.savefig(REPORT_FOLDER / "plots" / save_name, dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    # Configuration
    folder = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
    model_names = arch_model_names()
    num_experiments = 10
    step_size = 1
    dataset_start = 1
    dataset_cap = 38  # Maximum profiles per architecture (75/25 split of 50 profiles)

    # Feature selection
    features_filename = "rf_gpu_kernels_nomem"
    num_features = 25
    model_kwargs = {}
    report_name = features_filename

    # Analysis settings
    load_report = True  # Load existing report or generate new one
    plot = True
    plot_model_names = model_names
    plot_datasets = ["test"]  # ['val', 'train', 'test']
    xlim_upper = None
    title = False

    # ---------------------------------------------------------------------

    # Load and prepare data
    df = getDF(path=folder)

    feature_rank = loadReport(features_filename, feature_rank=True)["feature_rank"]
    selected_features = feature_rank[:num_features]
    df = filter_cols(df, substrs=selected_features)

    # Generate or load report
    if not load_report:
        report = generateReport(
            df=df,
            model_names=model_names,
            model_kwargs=model_kwargs,
            num_experiments=num_experiments,
            step_size=step_size,
            dataset_start=dataset_start,
            dataset_cap=dataset_cap,
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
            save_name=f"{features_filename}_{plot_datasets[-1]}.png",
            title=title,
        )
