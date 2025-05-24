"""
Generate accuracy tables for architecture prediction models across different data subsets.

This module analyzes the performance of architecture prediction models when trained on
different subsets of profiling data. It generates tables showing top-k accuracy metrics
for training, validation, and test sets, organized by data subset and model type.

Generate a table (csv format/ pandas dataframe) where the columns are the 
train/val acc1 and acc5 by the subset of data used to train the architecture
prediction model (with the count of # of features), and the rows are the type 
of architecture prediction model.

The analysis includes:
- Semantic data subsets (e.g., system data, GPU kernel data, API calls)
- Top-ranked feature subsets based on feature importance
- Cross-framework performance analysis

Example Usage:
    ```python
    # Generate full analysis table
    createTable()

    # Analyze specific GPU kernel features
    small()

    # Analyze cross-framework performance
    crossMlFramework()
    ```
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append("../edge_profile")

from architecture_prediction import (
    ArchPredBase,
    RFArchPred,
    arch_model_full_name,
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

# Define paths for profile data
PROFILE_FOLDER = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
CROSS_ML_FRAMEWORK_PROFILE_FOLDER = (
    Path.cwd() / "profiles" / "quadro_rtx_8000" / "tensorflow_and_zero_exe_pretrained"
)
SAVE_FOLDER = Path(__file__).parent.absolute() / "arch_pred_acc_by_data_subset"
QUADRO_VICT_PROFILE_FOLDER = Path.cwd() / "victim_profiles"


def loadReport(filename: str) -> Dict:
    """
    Load a feature ranking report from JSON file.

    Args:
        filename (str): Name of the feature ranking report file

    Returns:
        Dict: Loaded feature ranking report

    Example:
        ```python
        report = loadReport("feature_rank_lr.json")
        feature_rank = report["feature_rank"]
        ```
    """
    report_path = Path(__file__).parent.absolute() / "feature_ranks" / filename
    with open(report_path, "r") as f:
        report = json.load(f)
    return report


def generateTable(
    data_subsets: Dict[str, pd.DataFrame],
    victim_profile_folder: Path,
    model_names: Optional[List[str]] = None,
    topk: List[int] = [1],
    save_name: Optional[str] = None,
) -> None:
    """
    Generate a table of accuracy metrics for different data subsets and models.

    Args:
        data_subsets (Dict[str, pd.DataFrame]): Dictionary mapping subset names to their data
        victim_profile_folder (Path): Path to victim profile data
        model_names (Optional[List[str]]): List of model types to evaluate.
            Defaults to all available models.
        topk (List[int]): List of k values for top-k accuracy. Defaults to [1].
        save_name (Optional[str]): Name for the output CSV file.
            Defaults to "arch_pred_acc_by_data_subset.csv".

    The generated table includes:
    - Architecture prediction model type
    - For each data subset:
        - Number of features
        - Top-k accuracy for training, validation, and test sets
    """
    if save_name is None:
        save_name = "arch_pred_acc_by_data_subset.csv"
    save_path = SAVE_FOLDER / save_name

    if model_names is None:
        model_names = arch_model_names()

    columns = ["Architecture Prediction Model Type"]
    num_columns = {}

    # Generate column names for the table
    for data_subset in data_subsets:
        num_columns[data_subset] = len(list(data_subsets[data_subset].columns)) - 3
        for k in topk:
            columns.append(f"{data_subset} ({num_columns[data_subset]}) Top {k}")
            columns.append(f"{data_subset} ({num_columns[data_subset]}) Top {k} Train")
            columns.append(f"{data_subset} ({num_columns[data_subset]}) Top {k} Val")
            columns.append(f"{data_subset} ({num_columns[data_subset]}) Top {k} Test")

    table = pd.DataFrame(columns=columns)

    # Evaluate each model on each data subset
    for model_type in model_names:
        row_data = {
            "Architecture Prediction Model Type": arch_model_full_name()[model_type]
        }
        for data_subset in data_subsets:
            print(f"Training {model_type} on {data_subset}")
            model = get_arch_pred_model(
                model_type=model_type, df=data_subsets[data_subset]
            )
            for k in topk:
                model_pred_fn = None
                if hasattr(model.model, "decision_function"):
                    model_pred_fn = model.model.decision_function
                elif hasattr(model.model, "predict_proba"):
                    model_pred_fn = model.model.predict_proba

                test = predictVictimArchs(
                    model, victim_profile_folder, save=False, topk=1, verbose=False
                )["accuracy_k"][1]

                if model_pred_fn is not None:
                    train = top_k_accuracy_score(
                        model.y_train, model_pred_fn(model.x_tr), k=k
                    )
                    val = top_k_accuracy_score(
                        model.y_test, model_pred_fn(model.x_test), k=k
                    )
                else:
                    train = np.nan
                    val = np.nan
                    if k == 1:
                        train = model.evaluateTrain()
                        val = model.evaluateTest()

                row_data[
                    f"{data_subset} ({num_columns[data_subset]}) Top {k}"
                ] = "{:.3g}/{:.3g}".format(train * 100, val * 100)
                row_data[
                    f"{data_subset} ({num_columns[data_subset]}) Top {k} Train"
                ] = (train * 100)
                row_data[f"{data_subset} ({num_columns[data_subset]}) Top {k} Val"] = (
                    val * 100
                )
                row_data[f"{data_subset} ({num_columns[data_subset]}) Top {k} Test"] = (
                    test * 100
                )

        table = table.append(row_data, ignore_index=True)

    # Save results
    table.to_csv(save_path)
    transpose_path = SAVE_FOLDER / f"{save_path.name[:-4]}_transpose.csv"
    table.T.to_csv(transpose_path)

    printNumFeatures(data_subsets)


def semanticSubsets(profile_folder: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate semantic subsets of the profiling data.

    Args:
        profile_folder (Optional[Path]): Path to profile data.
            Defaults to PROFILE_FOLDER.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping subset names to their data

    The subsets include:
    - All: Complete dataset
    - System: System-level data only
    - No System: Data excluding system-level information
    - GPU Kernel: GPU kernel activity data
    - API Calls: API call data
    - Indicator: Indicator columns only
    - No Indicator: Data excluding indicator columns
    - GPU Kernel, No Memory: GPU kernel data excluding memory operations
    - GPU Kernel, Memory Only: GPU kernel memory operations only
    """
    if profile_folder is None:
        profile_folder = PROFILE_FOLDER
    data_subsets = {
        "All": all_data(profile_folder),
        "System": all_data(profile_folder, system_data_only=True),
        "No System": all_data(profile_folder, no_system_data=True),
        "GPU Kernel": all_data(
            profile_folder, gpu_activities_only=True, no_system_data=True
        ),
        "API Calls": all_data(profile_folder, api_calls_only=True, no_system_data=True),
        "Indicator": all_data(profile_folder, indicators_only=True),
        "No Indicator": remove_cols(all_data(profile_folder), substrs=["indicator"]),
        "GPU Kernel, No Memory": remove_cols(
            all_data(profile_folder, gpu_activities_only=True, no_system_data=True),
            substrs=["mem"],
        ),
        "GPU Kernel, Memory Only": filter_cols(
            all_data(profile_folder, gpu_activities_only=True, no_system_data=True),
            substrs=["mem"],
        ),
    }
    return data_subsets


def topFeatureSubsets(
    feature_rank_file: str, num_features: List[int] = [5]
) -> Dict[str, pd.DataFrame]:
    """
    Generate data subsets based on top-ranked features.

    Args:
        feature_rank_file (str): Name of the feature ranking report file
        num_features (List[int]): List of feature counts to include.
            Defaults to [5].

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping subset names to their data

    Example:
        ```python
        subsets = topFeatureSubsets("feature_rank_lr.json", num_features=[5, 10])
        ```
    """
    feature_rank = loadReport(feature_rank_file)["feature_rank"]

    data_subsets = {}
    for feature_count in num_features:
        new_features = feature_rank[:feature_count]
        data_subsets[f"Top {feature_count} Features"] = filter_cols(
            all_data(PROFILE_FOLDER), substrs=new_features
        )
    return data_subsets


def createTable() -> None:
    """
    Create a comprehensive analysis table including all semantic and top feature subsets.
    """
    subsets = semanticSubsets()
    subsets.update(topFeatureSubsets(feature_rank_file="feature_rank_lr.json"))
    generateTable(subsets, victim_profile_folder=QUADRO_VICT_PROFILE_FOLDER)


def printNumFeatures(subsets: Dict[str, pd.DataFrame]) -> None:
    """
    Print the number of features in each data subset.

    Args:
        subsets (Dict[str, pd.DataFrame]): Dictionary of data subsets to analyze
    """
    for subset in subsets:
        print(f"{subset.ljust(30)}: {len(list(subsets[subset].columns)) - 3} features")


def small() -> None:
    """
    Run a small-scale analysis focusing on GPU kernel features without memory operations.
    """
    subsets = semanticSubsets()
    subsets = {"GPU Kernel, No Memory": subsets["GPU Kernel, No Memory"]}

    generateTable(
        subsets,
        victim_profile_folder=QUADRO_VICT_PROFILE_FOLDER,
        save_name="gpu_nomem.csv",
    )


def best_rf_gpu_nomem() -> None:
    """
    Analyze the best Random Forest model performance on GPU kernel features.
    """
    subsets = topFeatureSubsets(
        feature_rank_file="rf_gpu_kernels_nomem.json", num_features=[3, 25, 1000]
    )
    generateTable(
        subsets,
        victim_profile_folder=QUADRO_VICT_PROFILE_FOLDER,
        save_name="rf_gpu_nomem_rank.csv",
    )


def crossMlFramework() -> None:
    """
    Analyze model performance across different machine learning frameworks.
    """
    subsets = {"All": all_data(CROSS_ML_FRAMEWORK_PROFILE_FOLDER)}
    generateTable(
        subsets,
        victim_profile_folder=QUADRO_VICT_PROFILE_FOLDER,
        save_name="cross_ml_frameworks.csv",
    )


if __name__ == "__main__":
    # createTable()
    # small()
    # best_rf_gpu_nomem()
    crossMlFramework()
    exit(0)
