"""
Analyze the impact of noise on architecture prediction accuracy.

This module evaluates how adding controlled noise to GPU profiles affects the accuracy
of architecture prediction models. It supports different adversary models and noise
levels to understand the robustness of architecture prediction.

Key Features:
1. Multiple adversary models:
   - Unaware adversary (baseline)
   - Aware adversary with averaging (5/10/25 samples)
   - Aware adversary with minimum subtraction (5/10/25 samples)
2. Configurable noise levels and features
3. Multiple architecture prediction models
4. Comprehensive reporting and visualization

Example Usage:
    ```python
    # Generate noise analysis report
    generateReport(
        profile_df=df,
        noise_features=noise_features,
        num_experiments=10,
        noise_levels=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
        arch_model_names=["rf", "knn"],
        offset=1
    )
    ```
"""

import datetime
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

# Add parent directory to path for imports
sys.path.append("../edge_profile")

from arch_pred_accuracy import getDF
from architecture_prediction import get_arch_pred_model
from data_engineering import filter_cols

# Configure matplotlib settings
rc("font", **{"family": "serif", "serif": ["Times"], "size": 12})
rc("figure", **{"figsize": (6, 4)})

# Constants for file paths
REPORT_FOLDER = Path(__file__).parent.absolute() / "theoretical_dummy"
REPORT_FOLDER.mkdir(exist_ok=True)
plot_folder = REPORT_FOLDER / "plots"
plot_folder.mkdir(exist_ok=True)


def generateReport(
    profile_df: pd.DataFrame,
    noise_features: List[str],
    num_experiments: int,
    noise_levels: List[float],
    arch_model_names: List[str],
    offset: float = 0,
) -> None:
    """
    Generate a comprehensive report on the impact of noise on architecture prediction.

    This function evaluates multiple adversary models:
    1. Unaware adversary: Trains on clean data, tests on noisy data
    2. Aware adversary (averaging): Trains on clean data, tests on averaged noisy data
    3. Aware adversary (subtraction): Trains on clean data, tests on minimum noisy data

    Args:
        profile_df (pd.DataFrame): Clean GPU profile data
        noise_features (List[str]): Features to add noise to
        num_experiments (int): Number of experiments per noise level
        noise_levels (List[float]): Noise levels to test (multiples of feature STD)
        arch_model_names (List[str]): Architecture prediction models to evaluate
        offset (float): Constant offset for noise generation

    The report includes:
    - Accuracy results for each adversary model
    - Multiple noise levels
    - Multiple architecture prediction models
    - Statistical analysis (mean, std) across experiments
    - results stores the accuracy of arch pred models by an adversary who is
       unaware that any noise is being added.
    - results_subtract_adversary5 stores the accuracy of arch pred models by an
      adversary who is aware that noise might be added as a defense, but is
      unaware of the noise distribution.  This is a weaker adversary, and they
      will attempt to get rid of the noise by profiling models 5 times
      and taking the minimum of each profile feature.
    - results_avg_adversary stores the accuracy of arch pred models by an
      adversary who is aware that noise might be added as a defense, but is
      unaware of the noise distribution
      This adversary trains on noiseless data.
      They then profile the model 5 or 10 times and use the average.
    """
    report = {
        "features": list(profile_df.columns),
        "noise_features": noise_features,
        "num_experiments": num_experiments,
        "noise_levels": noise_levels,
        "arch_model_names": arch_model_names,
        "offset": offset,
        "results": {},
        "results_aware_adversary5": {},
        "results_aware_adversary10": {},
        "results_aware_adversary25": {},
        "results_subtract_adversary5": {},
        "results_subtract_adversary10": {},
        "results_subtract_adversary25": {},
    }
    dataset_size = len(profile_df)

    for noise_level in noise_levels:
        print(f"Generating results for noise level {noise_level}...")
        # Initialize result containers for each model type
        noise_config = {model_name: [] for model_name in arch_model_names}
        noise_config_aware5 = {model_name: [] for model_name in arch_model_names}
        noise_config_aware10 = {model_name: [] for model_name in arch_model_names}
        noise_config_aware25 = {model_name: [] for model_name in arch_model_names}
        noise_config_subtract5 = {model_name: [] for model_name in arch_model_names}
        noise_config_subtract10 = {model_name: [] for model_name in arch_model_names}
        noise_config_subtract25 = {model_name: [] for model_name in arch_model_names}

        for experiment in range(num_experiments):
            # Create copies for different adversary models
            df_experiment = df.copy(deep=True)  # Unaware adversary
            df_aware5 = df.copy(deep=True)  # Aware adversary (5 samples)
            df_aware10 = df.copy(deep=True)  # Aware adversary (10 samples)
            df_aware25 = df.copy(deep=True)  # Aware adversary (25 samples)
            df_sub5 = df.copy(deep=True)  # Subtraction adversary (5 samples)
            df_sub10 = df.copy(deep=True)  # Subtraction adversary (10 samples)
            df_sub25 = df.copy(deep=True)  # Subtraction adversary (25 samples)

            # Add noise if noise level > 0
            if noise_level > 0.0:
                for noise_feature in noise_features:
                    std = df[noise_feature].std()
                    const = df_experiment[noise_feature].std() * offset

                    # Generate noise for different adversary models
                    noise_to_add = np.random.uniform(
                        const, const + noise_level * std, size=dataset_size
                    )
                    min_noise_5 = np.random.uniform(
                        const, const + noise_level * std, size=(dataset_size, 5)
                    )
                    min_noise_10 = np.random.uniform(
                        const, const + noise_level * std, size=(dataset_size, 10)
                    )
                    min_noise_25 = np.random.uniform(
                        const, const + noise_level * std, size=(dataset_size, 25)
                    )

                    # Add noise to each dataset
                    df_experiment[noise_feature] += noise_to_add

                    # Aware adversary: Average noise and subtract expected value
                    expected_val = (noise_level * std) / 2 + const
                    df_aware5[noise_feature] += min_noise_5.mean(axis=1) - expected_val
                    df_aware10[noise_feature] += (
                        min_noise_10.mean(axis=1) - expected_val
                    )
                    df_aware25[noise_feature] += (
                        min_noise_25.mean(axis=1) - expected_val
                    )

                    # Subtraction adversary: Use minimum noise
                    df_sub5[noise_feature] += min_noise_5.min(axis=1)
                    df_sub10[noise_feature] += min_noise_10.min(axis=1)
                    df_sub25[noise_feature] += min_noise_25.min(axis=1)

            # Evaluate each architecture prediction model
            for model_name in arch_model_names:
                # Unaware adversary
                model = get_arch_pred_model(model_name, df=df)
                noise_config[model_name].append(model.evaluateAcc(df_experiment))

                # Aware adversary (averaging)
                aware_model = get_arch_pred_model(model_name, df=df)
                noise_config_aware5[model_name].append(
                    aware_model.evaluateAcc(df_aware5)
                )
                noise_config_aware10[model_name].append(
                    aware_model.evaluateAcc(df_aware10)
                )
                noise_config_aware25[model_name].append(
                    aware_model.evaluateAcc(df_aware25)
                )

                # Aware adversary (subtraction)
                noise_config_subtract5[model_name].append(model.evaluateAcc(df_sub5))
                noise_config_subtract10[model_name].append(model.evaluateAcc(df_sub10))
                noise_config_subtract25[model_name].append(model.evaluateAcc(df_sub25))

        # Store results for this noise level
        report["results"][noise_level] = noise_config
        report["results_aware_adversary5"][noise_level] = noise_config_aware5
        report["results_aware_adversary10"][noise_level] = noise_config_aware10
        report["results_aware_adversary25"][noise_level] = noise_config_aware25
        report["results_subtract_adversary5"][noise_level] = noise_config_subtract5
        report["results_subtract_adversary10"][noise_level] = noise_config_subtract10
        report["results_subtract_adversary25"][noise_level] = noise_config_subtract25

    # Save report
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = REPORT_FOLDER / f"{time}.json"
    with open(save_path, "w+") as f:
        json.dump(report, f, indent=4)


def plotFromReport(report_path: Path, arch_model_names: List[str]) -> None:
    """
    Generate plots comparing architecture prediction accuracy across noise levels.

    Args:
        report_path (Path): Path to the report JSON file
        arch_model_names (List[str]): List of architecture prediction models to plot

    The plot shows:
    - Accuracy vs noise level for each model
    - Standard deviation bands
    - Clear model comparison
    """
    with open(report_path, "r+") as f:
        report = json.load(f)

    for model_name in arch_model_names:
        model_avgs = []
        model_stds = []
        for noise_lvl in report["noise_levels"]:
            model_results = report["results"][str(noise_lvl)][model_name]
            model_avgs.append(sum(model_results) / len(model_results))
            model_stds.append(np.std(model_results))

        # Plot mean accuracy
        plt.plot(report["noise_levels"], model_avgs, label=model_name)

        # Add standard deviation bands
        minus_std = [mean - std for mean, std in zip(model_avgs, model_stds)]
        plus_std = [mean + std for mean, std in zip(model_avgs, model_stds)]
        plt.fill_between(
            report["noise_levels"],
            minus_std,
            plus_std,
            alpha=0.2,
        )

    # Customize plot
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.xlabel(
        "Range of Uniform Distribution Used to Add Noise\n(as a multiple of feature STD)"
    )
    plt.xticks(report["noise_levels"])
    plt.ylabel("Architecture Prediction Accuracy")
    plt.title(
        "Architecture Prediction Accuracy\nby Range of Uniform Distribution Used to Add Noise to Profiles"
    )

    # Save plot
    plt.savefig(
        REPORT_FOLDER / "plots" / report_path.name.replace(".json", ".png"),
        dpi=500,
        bbox_inches="tight",
    )


def plotOneArchFromReport(report_path: Path, arch_model_name: str) -> None:
    """
    Generate plots comparing different adversary models for a single architecture predictor.

    Args:
        report_path (Path): Path to the report JSON file
        arch_model_name (str): Name of the architecture prediction model to plot

    The plot shows:
    - Accuracy vs noise level for each adversary model
    - Standard deviation bands
    - Clear adversary model comparison
    """
    with open(report_path, "r+") as f:
        report = json.load(f)

    result_names = [
        "results",
        "results_aware_adversary5",
        "results_aware_adversary10",
        "results_aware_adversary25",
        "results_subtract_adversary5",
        "results_subtract_adversary10",
        "results_subtract_adversary25",
    ]

    for name in result_names:
        model_avgs = []
        model_stds = []
        for noise_lvl in report["noise_levels"]:
            model_results = report[name][str(noise_lvl)][arch_model_name]
            model_avgs.append(sum(model_results) / len(model_results))
            model_stds.append(np.std(model_results))

        # Plot mean accuracy with appropriate style
        if name == "results":
            plt.plot(report["noise_levels"], model_avgs, "--", label="weak_adversary")
        else:
            plt.plot(
                report["noise_levels"], model_avgs, label=name[name.find("_") + 1 :]
            )

        # Add standard deviation bands
        minus_std = [mean - std for mean, std in zip(model_avgs, model_stds)]
        plus_std = [mean + std for mean, std in zip(model_avgs, model_stds)]
        plt.fill_between(
            report["noise_levels"],
            minus_std,
            plus_std,
            alpha=0.2,
        )

    # Customize plot
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.xlabel(
        "Range of Uniform Distribution Used to Add Noise\n(as a multiple of feature STD)"
    )
    plt.xticks(report["noise_levels"])
    plt.ylabel("Architecture Prediction Accuracy")
    plt.title(
        f"{arch_model_name} Architecture Prediction Accuracy by Range\n"
        "of Uniform Distribution Used to Add Noise to Profiles"
    )

    # Save plot
    plt.savefig(
        REPORT_FOLDER / "plots" / report_path.name.replace(".json", ".png"),
        dpi=500,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    # Configuration
    GPU_PROFILES_PATH = (
        Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
    )
    FEATURE_RANK_PATH = (
        Path.cwd() / "plots" / "feature_ranks" / "rf_gpu_kernels_nomem.json"
    )
    NUM_FEATURES = 3  # Number of features to train architecture prediction models
    NOISE_FEATURES = 3  # Number of features to add noise to

    # ---------------------------------------------------

    # Experiment parameters
    NOISE_LEVELS = [0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2]
    NUM_EXPERIMENTS = 10
    OFFSET = 1
    ARCH_MODEL_NAMES = ["rf", "knn"]

    # Control flags
    GENERATE_REPORT = False
    PLOT = True
    PLOT_ONE_ARCH_MODEL_BY_ADV = "rf"  # None for all models
    PLOT_FILENAME = REPORT_FOLDER / "20230513-184716.json"

    # Generate report if requested
    if GENERATE_REPORT:
        assert NOISE_FEATURES <= NUM_FEATURES

        # Load feature ranking
        with open(FEATURE_RANK_PATH, "r") as f:
            report = json.load(f)["feature_rank"]
        features = report[:NUM_FEATURES]
        noise_features = report[:NOISE_FEATURES]

        # Load and filter data
        df = getDF(GPU_PROFILES_PATH)
        df = filter_cols(df, substrs=features)

        # Generate report
        generateReport(
            profile_df=df,
            noise_features=noise_features,
            num_experiments=NUM_EXPERIMENTS,
            noise_levels=NOISE_LEVELS,
            arch_model_names=ARCH_MODEL_NAMES,
            offset=OFFSET,
        )

    # Generate plots if requested
    if PLOT:
        if PLOT_ONE_ARCH_MODEL_BY_ADV is not None:
            plotOneArchFromReport(
                report_path=PLOT_FILENAME, arch_model_name=PLOT_ONE_ARCH_MODEL_BY_ADV
            )
        else:
            plotFromReport(report_path=PLOT_FILENAME, arch_model_names=ARCH_MODEL_NAMES)
