"""
Generate cross-ML framework architecture prediction accuracy analysis.

This module analyzes the accuracy of architecture prediction models across different ML frameworks
(PyTorch and TensorFlow) and data subsets. It generates tables comparing model performance
with various feature configurations and training scenarios.

The analysis includes:
1. Accuracy across different ML frameworks (PyTorch, TensorFlow, combined)
2. Performance with different feature subsets (system, GPU kernel, API calls, etc.)
3. Top-k accuracy metrics for each configuration
4. Training and validation accuracy comparisons

Example Usage:
    ```python
    # Generate cross-ML framework analysis
    crossMlFramework(feature_rank_file="rf_gpu_kernels_nomem.json")
    
    # Generate analysis with semantic subsets
    subsets = semanticSubsets()
    generateTable(subsets, topk=[1, 5])
    ```
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import top_k_accuracy_score

# Add parent directory to path for imports
sys.path.append("../edge_profile")

from architecture_prediction import (
    arch_model_full_name,
    arch_model_names,
    get_arch_pred_model,
)
from data_engineering import (
    all_data,
    filter_cols,
    remove_cols,
)

# Constants for file paths
PYTORCH_FOLDER = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
TF_FOLDER = Path.cwd() / "profiles" / "quadro_rtx_8000" / "tensorflow_profiles"
ALL_FOLDER = (
    Path.cwd() / "profiles" / "quadro_rtx_8000" / "tensorflow_and_zero_exe_pretrained"
)
SAVE_FOLDER = Path(__file__).parent.absolute() / "cross_ml_framwork"


def loadReport(filename: str) -> Dict:
    """
    Load a report from a JSON file.

    Args:
        filename (str): Name of the report file

    Returns:
        Dict: The loaded report data
    """
    report_path = Path(__file__).parent.absolute() / "feature_ranks" / filename
    with open(report_path, "r") as f:
        report = json.load(f)
    return report


def generateTable(
    data_subsets: Dict[str, pd.DataFrame],
    model_names: Optional[List[str]] = None,
    topk: List[int] = [1],
    save_name: Optional[str] = None,
) -> None:
    """
    Generate a table comparing model performance across different data subsets.

    Args:
        data_subsets (Dict[str, pd.DataFrame]): Dictionary mapping subset names to their data
        model_names (Optional[List[str]]): List of model types to evaluate
        topk (List[int]): List of k values for top-k accuracy
        save_name (Optional[str]): Name for the output CSV file

    The table includes:
    - Top-k accuracy for each model and data subset
    - Training and validation accuracy
    - Feature counts for each subset
    """
    if save_name is None:
        save_name = "cross_ml_framework.csv"
    save_path = SAVE_FOLDER / save_name

    if model_names is None:
        model_names = arch_model_names()

    columns = ["Architecture Prediction Model Type"]
    num_columns: Dict[str, int] = {}

    for data_subset in data_subsets:
        num_columns[data_subset] = len(list(data_subsets[data_subset].columns)) - 3
        for k in topk:
            columns.append(f"{data_subset} ({num_columns[data_subset]}) Top {k}")
            columns.append(f"{data_subset} ({num_columns[data_subset]}) Top {k} Train")
            columns.append(f"{data_subset} ({num_columns[data_subset]}) Top {k} Val")

    table = pd.DataFrame(columns=columns)

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
        # row_data[f"PyTorch Top 1"] = model.evaluateAcc(all_data(PYTORCH_FOLDER))
        # row_data[f"TensorFlow Top 1"] = model.evaluateAcc(all_data(TF_FOLDER))

        table = table.append(row_data, ignore_index=True)

    table.to_csv(save_path)
    transpose_path = SAVE_FOLDER / f"{save_path.name[:-4]}_transpose.csv"
    table.T.to_csv(transpose_path)

    printNumFeatures(data_subsets)


def semanticSubsets(profile_folder: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate semantic subsets of the data based on different feature categories.

    Args:
        profile_folder (Optional[Path]): Path to the profile data folder

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping subset names to their data
    """
    if profile_folder is None:
        profile_folder = PYTORCH_FOLDER
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
    Generate subsets based on top-ranked features.

    Args:
        feature_rank_file (str): Path to the feature ranking file
        num_features (List[int]): List of feature counts to use

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping subset names to their data
    """
    feature_rank = loadReport(feature_rank_file)["feature_rank"]

    data_subsets = {}
    for feature_count in num_features:
        new_features = feature_rank[:feature_count]
        data_subsets[f"Top {feature_count} Features"] = filter_cols(
            all_data(PYTORCH_FOLDER), substrs=new_features
        )
    return data_subsets


def createTable() -> None:
    """
    Create a comprehensive table with all semantic and top feature subsets.
    """
    subsets = semanticSubsets()
    subsets.update(topFeatureSubsets(feature_rank_file="feature_rank_lr.json"))
    generateTable(subsets)


def printNumFeatures(subsets: Dict[str, pd.DataFrame]) -> None:
    """
    Print the number of features in each subset.

    Args:
        subsets (Dict[str, pd.DataFrame]): Dictionary of data subsets
    """
    for subset in subsets:
        print(f"{subset.ljust(30)}: {len(list(subsets[subset].columns)) - 3} features")


def small() -> None:
    """
    Generate a small experiment table focusing on GPU kernel features without memory.
    """
    subsets = semanticSubsets()
    subsets = {"GPU Kernel, No Memory": subsets["GPU Kernel, No Memory"]}
    generateTable(subsets, save_name="gpu_nomem.csv")


def best_rf_gpu_nomem() -> None:
    """
    Generate a table for the best RF model with GPU kernel features (no memory).
    """
    subsets = topFeatureSubsets(
        feature_rank_file="rf_gpu_kernels_nomem.json", num_features=[3, 25, 1000]
    )
    generateTable(subsets, save_name="rf_gpu_nomem_rank.csv")


def crossMlFramework(feature_rank_file: str = "rf_gpu_kernels_nomem.json") -> None:
    """
    Generate cross-ML framework analysis table.

    Args:
        feature_rank_file (str): Path to the feature ranking file
    """
    feature_rank = loadReport(feature_rank_file)["feature_rank"]
    subsets = {
        "PyTorch All Features": all_data(PYTORCH_FOLDER),
        "PyTorch GPU Kernel, No Memory": remove_cols(
            all_data(PYTORCH_FOLDER, gpu_activities_only=True, no_system_data=True),
            substrs=["mem"],
        ),
        "PyTorch Top 25": filter_cols(
            all_data(PYTORCH_FOLDER), substrs=feature_rank[:25]
        ),
        "PyTorch and TF": all_data(ALL_FOLDER),
    }
    models = ["rf", "knn", "centroid", "nb"]
    generateTable(subsets, save_name="cross_ml_frameworks.csv", model_names=models)


if __name__ == "__main__":
    if not SAVE_FOLDER.exists():
        SAVE_FOLDER.mkdir(parents=True)
    # createTable()
    # small()
    # best_rf_gpu_nomem()
    crossMlFramework()
    exit(0)
