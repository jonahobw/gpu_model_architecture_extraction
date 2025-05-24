"""
Analyze and visualize the distribution of top-ranked features across model architectures.

This module generates plots showing how different model architectures utilize the top N
features from a feature ranking. It normalizes feature values and compares their
distribution across different model families and architectures.

Key Features:
1. Feature value normalization using MinMaxScaler
2. Support for multiple model families and architectures
3. Configurable feature ranking and number of features
4. Customizable feature name formatting
5. Statistical analysis (mean, std) per feature and architecture

Example Usage:
    ```python
    # Generate plot for top 5 features
    feature_rank = loadFeatureRank("rf_gpu_kernels_nomem.json")
    features = feature_rank[:5]
    df = filter_cols(all_data(QUADRO_TRAIN), substrs=features)
    generatePlot(df, model_families, features)
    ```
"""

import datetime
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler

# Configure matplotlib settings
rc("font", **{"family": "serif", "serif": ["Times"]})

# Add parent directory to path for imports
sys.path.append("../edge_profile")

from data_engineering import all_data, filter_cols
from get_model import model_families

# Constants
SAVE_FOLDER = Path(__file__).parent.absolute() / "top_n_features"
SAVE_FOLDER.mkdir(exist_ok=True)

# Data paths
QUADRO_TRAIN = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
TESLA_TRAIN = Path.cwd() / "profiles" / "tesla_t4" / "colab_zero_exe_pretrained"

# Feature name formatting
replace_feature_str = {
    "time_ms_": "Time (ms) ",
    "avg_us_": "Avg (us) ",
    "num_calls_": "#Calls ",
}


def loadFeatureRank(filename: str) -> List[str]:
    """
    Load feature ranking from a JSON file.

    Args:
        filename (str): Name of the feature ranking file (with or without .json extension)

    Returns:
        List[str]: List of ranked feature names

    Example:
        ```python
        features = loadFeatureRank("rf_gpu_kernels_nomem.json")
        ```
    """
    if not filename.endswith(".json"):
        filename += ".json"
    report_path = Path(__file__).parent.absolute() / "feature_ranks" / filename
    with open(report_path, "r") as f:
        report = json.load(f)
    return report["feature_rank"]


def generatePlot(
    df: pd.DataFrame,
    model_architectures: Tuple[str, List[str]],
    features: List[str],
    feature_chars: int = 50,
) -> None:
    """
    Generate a plot showing feature value distribution across model architectures.

    The plot shows:
    - Normalized feature values for each model architecture
    - Different point styles for different model families
    - Feature names and descriptions in the plot
    - Statistical analysis (mean, std) for each feature

    Args:
        df (pd.DataFrame): DataFrame containing feature values and model information
        model_architectures (Tuple[str, List[str]]): Tuple of (family_name, [architecture_names])
        features (List[str]): List of features to plot
        feature_chars (int): Number of characters to show in feature names

    The data structure for feature analysis is:
    ```python
    {
        "model_family": {
            "model_architecture": {
                "feature": {
                    "mean": mean,
                    "std": std,
                }
            }
        }
    }
    ```
    """
    # Normalize feature values
    for feature in features:
        df[feature] = MinMaxScaler().fit_transform(np.array(df[feature]).reshape(-1, 1))

    # Calculate statistics per family and architecture
    family_mean_std = {fam: {} for fam, _ in model_families}
    for family, architectures in model_architectures:
        for arch in architectures:
            arch_data = {}
            for feature in features:
                data = df[df["model"] == arch][feature]
                feature_data = {"mean": data.mean(), "std": data.std()}
                arch_data[feature] = feature_data
            family_mean_std[family][arch] = arch_data
    # print(json.dumps(family_mean_std, indent=4))

    # Configure plot settings
    plt.rcParams["figure.figsize"] = (3, 4)
    plt.tight_layout()

    # Plot data points
    point_styles = ["x", "+", ".", "*", "x", "s", "2", "1", "3"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, family in enumerate(family_mean_std):
        for arch in family_mean_std[family]:
            for feature_count, feature in enumerate(features):
                plt.plot(
                    feature_count,
                    family_mean_std[family][arch][feature]["mean"],
                    point_styles[i],
                    label=family,
                    color=colors[i],
                )

    # Customize plot appearance
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=(1.02, 0.3), title="DNN Family")
    # plt.xlabel(f"Top {len(features)} Non-Memory GPU Kernel Features", loc="right")
    plt.xlabel(f"Top {len(features)} Features")
    # plt.xticks([x for x in range(len(features))], labels=[x[:feature_chars] for x in features], rotation=45)
    plt.xticks(
        [x for x in range(len(features))],
        labels=[f"({i})" for i in range(1, len(features) + 1)],
    )
    plt.ylabel("Feature Value (Normalized)")
    plt.yticks([])

    # Add feature descriptions
    feature_chars_list_hardcoded = [50, 51, 52, 58, 51]
    feature_str = ""
    for i, feature in enumerate(features):
        feature_cutoff = feature[: feature_chars_list_hardcoded[i]]
        for prefix in replace_feature_str:
            if feature_cutoff.startswith(prefix):
                feature_cutoff = feature_cutoff.replace(
                    prefix, replace_feature_str[prefix]
                )
                break
        feature_str += f"({i+1}) {feature_cutoff}...\n"

    plt.text(-0.3, -0.45, feature_str[:-1], fontsize=10)

    # Save plot and configuration
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(SAVE_FOLDER / f"{time}.png", dpi=500, bbox_inches="tight")

    # Save configuration
    family_mean_std["features"] = features
    config_filename = SAVE_FOLDER / f"config_{time}.json"
    with open(config_filename, "w") as f:
        json.dump(family_mean_std, f, indent=4)


if __name__ == "__main__":
    # Configuration
    feature_rank_filename = "rf_gpu_kernels_nomem.json"
    num_features = 5

    # list of tuples (family, [list of arch names for that family])
    model_architectures = model_families
    data_folder = QUADRO_TRAIN

    # ------------------------------------------------------

    feature_rank = loadFeatureRank(feature_rank_filename)
    features = feature_rank[:num_features]
    df = filter_cols(all_data(data_folder), substrs=features)
    generatePlot(df, model_architectures, features)
