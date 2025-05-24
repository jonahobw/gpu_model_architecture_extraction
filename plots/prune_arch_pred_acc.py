"""
Generate accuracy analysis for architecture prediction on pruned victim models.

This module analyzes the accuracy of architecture prediction models when tested on
pruned victim models. It generates tables comparing model performance with various
feature configurations and training scenarios.  The features used and dataset size
are configurable. 

The analysis includes:
1. Top-k accuracy metrics for each model
2. Training and testing accuracy comparisons
3. Family-level accuracy analysis
4. Performance with different feature subsets

Example Usage:
    ```python
    # Generate pruned model accuracy report
    model_names = arch_model_names()
    generatePruneReport(
        quadro_train=quadro_train,
        config=config,
        model_names=model_names,
        topk=[1, 5]
    )
    ```
"""

import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from sklearn.metrics import top_k_accuracy_score

# Add parent directory to path for imports
sys.path.append("../edge_profile")

from architecture_prediction import (
    ArchPredBase,
    arch_model_full_name,
    arch_model_names,
    get_arch_pred_model,
)
from data_engineering import (
    all_data,
    filter_cols,
    remove_cols,
    removeColumnsFromOther,
)
from experiments import predictVictimArchs
from model_manager import VictimModelManager

# Constants for file paths
SAVE_FOLDER = Path(__file__).parent.absolute() / "prune_pred_acc"
if not SAVE_FOLDER.exists():
    SAVE_FOLDER.mkdir(exist_ok=True)

QUADRO_TRAIN = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
QUADRO_PRUNE_TEST = Path.cwd() / "victim_profiles_pruned"


def loadFeatureRank(filename: str) -> List[str]:
    """
    Load feature ranking from a JSON file.

    Args:
        filename (str): Name of the feature ranking file (with or without .json extension)

    Returns:
        List[str]: List of features in ranked order
    """
    if not filename.endswith(".json"):
        filename += ".json"
    report_path = Path(__file__).parent.absolute() / "feature_ranks" / filename
    with open(report_path, "r") as f:
        report = json.load(f)
    return report["feature_rank"]


def getDF(
    path: Path,
    to_keep_path: Optional[Path] = None,
    df_args: Dict = {},
) -> pd.DataFrame:
    """
    Load and preprocess data from a directory.

    Args:
        path (Path): Path to the data directory
        to_keep_path (Optional[Path]): Path to reference data for column filtering
        df_args (Dict): Additional arguments for data loading

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = all_data(path, **df_args)
    if to_keep_path is not None:
        keep_df = all_data(to_keep_path, **df_args)
        df = removeColumnsFromOther(keep_df, df)
    return df


def generatePruneReport(
    quadro_train: pd.DataFrame,
    config: Dict,
    model_names: List[str],
    topk: List[int] = [1, 5],
    train_size: Optional[int] = None,
) -> None:
    """
    Generate a report comparing model performance on pruned victim models.

    Args:
        quadro_train (pd.DataFrame): Training data from Quadro RTX 8000
        config (Dict): Configuration parameters for the analysis
        model_names (List[str]): List of model types to evaluate
        topk (List[int]): List of k values for top-k accuracy
        train_size (Optional[int]): Number of profiles to keep per architecture

    The report includes:
    - Top-k accuracy for each model
    - Training and testing accuracy comparisons
    - Family-level accuracy
    - Detailed configuration parameters
    """
    columns = [
        "Architecture Prediction Model Type",
        "Train/Test",
        "Train",
        "Test",
        "Test Family Accuracy",
    ]

    for k in topk:
        columns.append(f"Test Top {k}")

    table = pd.DataFrame(columns=columns)

    for model_type in model_names:
        row_data = {
            "Architecture Prediction Model Type": arch_model_full_name()[model_type]
        }
        
        # Train model
        model = get_arch_pred_model(
            model_type=model_type,
            df=quadro_train,
            kwargs={"train_size": train_size}
        )
        train_top1 = model.evaluateTrain() * 100
        row_data["Train"] = train_top1

        # Test on pruned models
        test_k_report = predictVictimArchs(
            model, QUADRO_PRUNE_TEST, save=False, topk=max(topk), verbose=False
        )
        k_acc = test_k_report["accuracy_k"]

        row_data["Test Family Accuracy"] = test_k_report["family_accuracy"] * 100

        # Format top-k accuracy results
        topk_str = ""
        for k in topk:
            row_data[f"Test Top {k}"] = k_acc[k] * 100
            topk_str += "{:.3g}/".format(k_acc[k] * 100)

        row_data["Test"] = topk_str[:-1]
        row_data["Train/Test"] = "{:.3g}/{:.3g}".format(train_top1, k_acc[1] * 100)

        table = table.append(row_data, ignore_index=True)

    # Save results
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    table.to_csv(SAVE_FOLDER / f"{time}.csv")
    transpose_path = SAVE_FOLDER / f"{time}_transpose.csv"
    table.T.to_csv(transpose_path)

    # Save configuration
    config["topk"] = topk
    with open(SAVE_FOLDER / f"{time}.json", "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    # Configuration
    model_names = arch_model_names()
    topk = [1, 5]
    train_size = None  # Number of profiles to keep per architecture
    feature_ranking_file = "rf_gpu_kernels_nomem.json" # None for all features
    feature_num = 25  # Number of features to use
    df_args = {}  # Arguments for data loading, should be empty if feature_ranking_file is not None
    df_remove_substrs = []  # Substrings to remove from column names

    # ----------------------------------------------------------------------

    # Validate configuration
    if feature_ranking_file is not None:
        assert len(df_args) == 0
        assert len(df_remove_substrs) == 0

    # Load and preprocess training data
    quadro_train = getDF(QUADRO_TRAIN, df_args=df_args)

    # Apply feature selection if specified
    if feature_ranking_file is not None:
        feature_ranking = loadFeatureRank(feature_ranking_file)
        relevant_features = feature_ranking[:feature_num]
        quadro_train = filter_cols(quadro_train, substrs=relevant_features)

    # Remove specified columns
    quadro_train = remove_cols(quadro_train, substrs=df_remove_substrs)

    # Prepare configuration
    config = {
        "train_size": train_size,
        "feature_ranking_file": feature_ranking_file,
        "feature_num": feature_num,
        "df_args": df_args,
        "df_remove_substrs": df_remove_substrs,
        "model_names": model_names,
    }

    # Generate report
    generatePruneReport(
        quadro_train=quadro_train,
        config=config,
        model_names=model_names,
        topk=topk,
        train_size=train_size,
    )
