"""
Generate cross-GPU architecture prediction accuracy analysis.

This module analyzes the accuracy of architecture prediction models when trained and tested
across different GPU architectures (Quadro RTX 8000 and Tesla T4). It generates a table
comparing model performance in different training/testing scenarios.

The analysis includes:
1. Accuracy of models trained on Quadro RTX 8000 and tested on Tesla T4
2. Accuracy of models trained on Tesla T4 and tested on Quadro RTX 8000
3. Top-k accuracy metrics for each scenario
4. Family-level accuracy comparisons

Example Usage:
    ```python
    # Generate cross-GPU accuracy report with default settings
    model_names = arch_model_names()
    generateCrossGPUReport(
        quadro_train=quadro_train,
        tesla_train=tesla_train,
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
from typing import Dict, List, Optional, Set, Union

import pandas as pd

# setting path
sys.path.append("../edge_profile")

from architecture_prediction import (
    ArchPredBase,
    arch_model_full_name,
    arch_model_names,
    get_arch_pred_model,
)
from data_engineering import all_data, filter_cols, remove_cols, removeColumnsFromOther
from experiments import predictVictimArchs
from model_manager import VictimModelManager

SAVE_FOLDER = Path(__file__).parent.absolute() / "cross_gpu_acc"
if not SAVE_FOLDER.exists():
    SAVE_FOLDER.mkdir(exist_ok=True)

QUADRO_TRAIN = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
QUADRO_TEST = Path.cwd() / "victim_profiles"
TESLA_TRAIN = Path.cwd() / "profiles" / "tesla_t4" / "colab_zero_exe_pretrained"
TESLA_TEST = Path.cwd() / "victim_profiles_tesla"


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
        # remove cols of df if they aren't in keep_df
        df = removeColumnsFromOther(keep_df, df)
    return df


def getTestAcc(model: ArchPredBase, gpu_type: str, verbose: bool = False):
    """NOT USING THIS, INSTEAD LOAD THE PROFILES INTO A FOLDER
    USING loadProfilesToFolder() from model_manager.py
    """
    vict_model_paths = VictimModelManager.getModelPaths()
    for vict_path in vict_model_paths:
        print(f"Getting profiles for {vict_path.parent.name}...")
        manager = VictimModelManager.load(vict_path)


def generateCrossGPUReport(
    quadro_train: pd.DataFrame,
    tesla_train: pd.DataFrame,
    config: Dict,
    model_names: List[str],
    topk: List[int] = [1, 5],
    train_size: Optional[int] = None,
) -> None:
    """
    Generate a report comparing model performance across different GPU architectures.

    Args:
        quadro_train (pd.DataFrame): Training data from Quadro RTX 8000
        tesla_train (pd.DataFrame): Training data from Tesla T4
        config (Dict): Configuration parameters for the analysis
        model_names (List[str]): List of model types to evaluate
        topk (List[int]): List of k values for top-k accuracy
        train_size (Optional[int]): Number of profiles to keep per architecture

    The report includes:
    - Top-k accuracy for each model in both training scenarios
    - Family-level accuracy comparisons
    - Detailed configuration and feature analysis
    """
    columns = [
        "Architecture Prediction Model Type",
        "Train Quadro RTX 8000, Test Tesla",
        "Train Quadro RTX 8000, Test Tesla Family",
        "Train Tesla, Test Quadro RTX 8000",
        "Train Tesla, Test Quadro RTX 8000 Family",
    ]

    for k in topk:
        columns.append(f"Train Quadro RTX 8000, Test Tesla Top {k}")
        columns.append(f"Train Tesla, Test Quadro RTX 8000 Top {k}")

    table = pd.DataFrame(columns=columns)

    for model_type in model_names:
        row_data = {
            "Architecture Prediction Model Type": arch_model_full_name()[model_type]
        }
        # train on quadro
        model = get_arch_pred_model(
            model_type=model_type, df=quadro_train, kwargs={"train_size": train_size}
        )

        # test on tesla
        k_acc_report = predictVictimArchs(
            model, TESLA_TEST, save=False, topk=max(topk), verbose=False
        )
        k_acc = k_acc_report["accuracy_k"]

        row_data["Train Quadro RTX 8000, Test Tesla Family"] = k_acc_report[
            "family_accuracy"
        ]

        topk_str = ""
        for k in topk:
            row_data[f"Train Quadro RTX 8000, Test Tesla Top {k}"] = k_acc[k] * 100
            topk_str += "{:.3g}/".format(k_acc[k] * 100)

        row_data["Train Quadro RTX 8000, Test Tesla"] = topk_str[:-1]

        # train on tesla
        model = get_arch_pred_model(
            model_type=model_type, df=tesla_train, kwargs={"train_size": train_size}
        )

        # test on quadro
        k_acc_report = predictVictimArchs(
            model, QUADRO_TEST, save=False, topk=max(topk), verbose=False
        )
        k_acc = k_acc_report["accuracy_k"]

        row_data["Train Tesla, Test Quadro RTX 8000 Family"] = k_acc_report[
            "family_accuracy"
        ]

        topk_str = ""
        for k in topk:
            row_data[f"Train Tesla, Test Quadro RTX 8000 Top {k}"] = k_acc[k] * 100
            topk_str += "{:.3g}/".format(k_acc[k] * 100)

        row_data["Train Tesla, Test Quadro RTX 8000"] = topk_str[:-1]

        table = table.append(row_data, ignore_index=True)

    # save table and config
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    table.to_csv(SAVE_FOLDER / f"{time}.csv")

    config["topk"] = topk
    config["feature_analysis"] = featureAnalysis(quadro_train, tesla_train)
    with open(SAVE_FOLDER / f"{time}.json", "w") as f:
        json.dump(config, f, indent=4)


def featureAnalysis(
    quadro_train: pd.DataFrame,
    tesla_train: pd.DataFrame,
) -> Dict[str, Union[int, List[str], Dict[str, Union[int, List[str]]]]]:
    """
    Analyze feature overlap between Quadro and Tesla datasets.

    Args:
        quadro_train (pd.DataFrame): Training data from Quadro RTX 8000
        tesla_train (pd.DataFrame): Training data from Tesla T4

    Returns:
        Dict containing:
            - Feature counts for each GPU
            - Unique features for each GPU
            - Shared features between GPUs
            - Complete feature lists
    """
    quadro_features: Set[str] = set(list(quadro_train.columns))
    tesla_features: Set[str] = set(list(tesla_train.columns))

    shared_cols = quadro_features.intersection(tesla_features)

    quadro_unique = quadro_features - tesla_features
    tesla_unique = tesla_features - quadro_features

    print("\n\nNote: these feature counts include the 3 label features")
    print("model, model_family, and file")
    print(
        f"Quadro:          {len(quadro_features)} features, {len(quadro_unique)} unique."
    )
    for feature in quadro_unique:
        print(f"\t{feature}")
    print(
        f"Tesla:           {len(tesla_features)} features, {len(tesla_unique)} unique."
    )
    for feature in tesla_unique:
        print(f"\t{feature}")
    print(f"Shared Features: {len(shared_cols)}")

    report = {
        "quadro": {
            "num_features": len(quadro_features),
            "num_unique_features": len(quadro_unique),
            "unique_features": list(quadro_unique),
            "features": list(quadro_features),
        },
        "tesla": {
            "num_features": len(tesla_features),
            "num_unique_features": len(tesla_unique),
            "unique_features": list(tesla_unique),
            "features": list(tesla_features),
        },
        "num_shared_features": len(shared_cols),
        "shared_features": list(shared_cols),
    }
    return report


if __name__ == "__main__":

    model_names = arch_model_names()  # all the models we want to use

    topk = [1, 5]

    # if not None, can be an int representing how many profiles to
    # keep per architecture in the training data
    train_size = None

    # if None, use all features. Else, this is a name of a feature ranking under
    # feature_ranks/
    feature_ranking_file = "rf_gpu_kernels_nomem.json"
    feature_num = 25  # the number of features to use

    # args to pass to load training data, if feature rank file is provided,
    # then this should be an empty dict
    df_args = {}  # {"no_system_data": True, "gpu_activities_only": True}

    # substrings to remove from the dataframe, if feature rank file is provided,
    # then this should be empty
    df_remove_substrs = []

    # ----------------------------------------------------------------------
    if feature_ranking_file is not None:
        assert len(df_args) == 0
        assert len(df_remove_substrs) == 0

    quadro_train = getDF(QUADRO_TRAIN, df_args=df_args)
    tesla_train = getDF(TESLA_TRAIN, df_args=df_args)

    if feature_ranking_file is not None:
        feature_ranking = loadFeatureRank(feature_ranking_file)
        relevant_features = feature_ranking[:feature_num]

        quadro_train = filter_cols(quadro_train, substrs=relevant_features)
        tesla_train = filter_cols(tesla_train, substrs=relevant_features)

    quadro_train = remove_cols(quadro_train, substrs=df_remove_substrs)
    tesla_train = remove_cols(tesla_train, df_remove_substrs)

    config = {
        "train_size": train_size,
        "feature_ranking_file": feature_ranking_file,
        "feature_num": feature_num,
        "df_args": df_args,
        "df_remove_substrs": df_remove_substrs,
        "model_names": model_names,
    }

    generateCrossGPUReport(
        quadro_train=quadro_train,
        tesla_train=tesla_train,
        config=config,
        model_names=model_names,
        topk=topk,
        train_size=train_size,
    )
