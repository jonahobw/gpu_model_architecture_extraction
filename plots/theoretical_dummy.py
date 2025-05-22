"""
This file will take a set of GPU profiles
as input, a set of GPU kernels/features to
perturb, and a set of GPU kernels/features
on which to train an architecture prediction
model, then will plot the architecture
prediction accuracy as various levels of noise
are added to the perturbed features.

**Note - we train the architecture prediction 
models on clean profiles (no noise added) and 
then evaluate the accuracy on profiles with 
noise added. 
"""
from pathlib import Path
import sys
import json
from typing import List
import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

# setting path
sys.path.append("../edge_profile")

from arch_pred_accuracy import getDF
from data_engineering import (
    filter_cols,
)
from architecture_prediction import (
    get_arch_pred_model,
    arch_model_names,
)

rc("font", **{"family": "serif", "serif": ["Times"], "size": 12})
rc("figure", **{"figsize": (6, 4)})

REPORT_FOLDER = Path(__file__).parent.absolute() / "theoretical_dummy"
REPORT_FOLDER.mkdir(exist_ok=True)
plot_folder = REPORT_FOLDER / "plots"
plot_folder.mkdir(exist_ok=True)


def generateReport(
    profile_df: pd.DataFrame,
    noise_features: List[str],
    num_experiments: int,
    noise_levels: List[int],
    arch_model_names: List[str],
    offset: float = 0,
):
    """
    results stores the accuracy of arch pred models by an adversary who is
    unaware that any noise is being added.

    results_avg_adversary stores the accuracy of arch pred models by an
    adversary who is aware that noise might be added as a defense, but is
    unaware of the noise distribution
    This adversary trains on noiseless data.
    They then profile the model 5 or 10 times and use the average.

    results_subtract_adversary5 stores the accuracy of arch pred models by an
    adversary who is aware that noise might be added as a defense, but is
    unaware of the noise distribution.  This is a weaker adversary, and they
    will attempt to get rid of the noise by profiling models 5 times
    and taking the minimum of each profile feature.
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
        noise_config = {model_name: [] for model_name in arch_model_names}
        noise_config_aware5 = {model_name: [] for model_name in arch_model_names}
        noise_config_aware10 = {model_name: [] for model_name in arch_model_names}
        noise_config_aware25 = {model_name: [] for model_name in arch_model_names}
        noise_config_subtract5 = {model_name: [] for model_name in arch_model_names}
        noise_config_subtract10 = {model_name: [] for model_name in arch_model_names}
        noise_config_subtract25 = {model_name: [] for model_name in arch_model_names}
        for experiment in range(num_experiments):
            # this is for testing the unaware adversary and results_subtract_adversary
            # the results_aware_adversary will use this for training
            df_experiment = df.copy(deep=True)
            # this is for testing the results_aware_adversary
            df_aware5 = df.copy(deep=True)
            df_aware10 = df.copy(deep=True)
            df_aware25 = df.copy(deep=True)
            # for testing the subtraction adversaries
            df_sub5 = df.copy(deep=True)
            df_sub10 = df.copy(deep=True)
            df_sub25 = df.copy(deep=True)
            # only need to add noise for noise level > 0
            if noise_level > 0.0:
                for noise_feature in noise_features:
                    std = df[noise_feature].std()
                    const = df_experiment[noise_feature].std() * offset
                    # replace feature column with noise added
                    noise_to_add = np.random.uniform(
                        const, const + noise_level * std, size=dataset_size
                    )
                    df_experiment[noise_feature] += noise_to_add

                    min_noise_5 = np.random.uniform(
                        const, const + noise_level * std, size=(dataset_size, 5)
                    )
                    min_noise_10 = np.random.uniform(
                        const, const + noise_level * std, size=(dataset_size, 10)
                    )
                    min_noise_25 = np.random.uniform(
                        const, const + noise_level * std, size=(dataset_size, 25)
                    )

                    # for the distribution aware adversary, we need to generate 5 or 10 random
                    # numbers per profile and take the avg, then subtract the mean of the distribution,
                    # then add to the dataframe
                    expected_val = (noise_level * std) / 2 + const
                    df_aware5[noise_feature] += min_noise_5.mean(axis=1) - expected_val
                    df_aware10[noise_feature] += (
                        min_noise_10.mean(axis=1) - expected_val
                    )
                    df_aware25[noise_feature] += (
                        min_noise_25.mean(axis=1) - expected_val
                    )
                    # for the subtraction adversary, we need to generate 5 or 10 random
                    # numbers per profile and take the min, then add that to the dataframe
                    df_sub5[noise_feature] += min_noise_5.min(axis=1)
                    df_sub10[noise_feature] += min_noise_10.min(axis=1)
                    df_sub25[noise_feature] += min_noise_25.min(axis=1)
            for model_name in arch_model_names:
                # unaware adversary - train on noiseless data and test on noisy data
                model = get_arch_pred_model(model_name, df=df)
                noise_config[model_name].append(model.evaluateAcc(df_experiment))

                # results_aware_adversary - train on noiseless and test on averaged out noise
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

                # results_subtract_adversary - train on noiseless, but for testing, take
                # the min of multiple profiles from noisy data
                # can reuse the <model> variable from above since it is trained on noiseless data
                noise_config_subtract5[model_name].append(model.evaluateAcc(df_sub5))
                noise_config_subtract10[model_name].append(model.evaluateAcc(df_sub10))
                noise_config_subtract25[model_name].append(model.evaluateAcc(df_sub25))

        report["results"][noise_level] = noise_config
        report["results_aware_adversary5"][noise_level] = noise_config_aware5
        report["results_aware_adversary10"][noise_level] = noise_config_aware10
        report["results_aware_adversary25"][noise_level] = noise_config_aware25
        report["results_subtract_adversary5"][noise_level] = noise_config_subtract5
        report["results_subtract_adversary10"][noise_level] = noise_config_subtract10
        report["results_subtract_adversary25"][noise_level] = noise_config_subtract25

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = REPORT_FOLDER / f"{time}.json"
    with open(save_path, "w+") as f:
        json.dump(report, f, indent=4)


def plotFromReport(report_path: Path, arch_model_names: List[str]):
    with open(report_path, "r+") as f:
        report = json.load(f)

    for model_name in arch_model_names:
        model_avgs = []
        model_stds = []
        for noise_lvl in report["noise_levels"]:
            model_results = report["results"][str(noise_lvl)][model_name]
            model_avgs.append(sum(model_results) / len(model_results))
            model_stds.append(np.std(model_results))

        plt.plot(report["noise_levels"], model_avgs, label=model_name)
        minus_std = []
        plus_std = []
        for mean, std in zip(model_avgs, model_stds):
            minus_std.append(mean - std)
            plus_std.append(mean + std)
        plt.fill_between(
            report["noise_levels"],
            minus_std,
            plus_std,
            alpha=0.2,
        )
    # plt.rcParams["figure.figsize"] = (10, 10)
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.xlabel(
        "Range of Uniform Distribution Used to Add Noise\n(as a multiple of feature STD)"
    )

    plt.xticks(report["noise_levels"])

    plt.ylabel(f"Architecture Prediction Accuracy")

    plt.title(
        f"Architecture Prediction Accuracy\nby Range of Uniform Distribution Used to Add Noise to Profiles"
    )

    plt.savefig(
        REPORT_FOLDER / "plots" / report_path.name.replace(".json", ".png"),
        dpi=500,
        bbox_inches="tight",
    )


def plotOneArchFromReport(report_path: Path, arch_model_name: str):
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

        if name == "results":
            plt.plot(report["noise_levels"], model_avgs, "--", label="weak_adversary")
        else:
            plt.plot(
                report["noise_levels"], model_avgs, label=name[name.find("_") + 1 :]
            )
        minus_std = []
        plus_std = []
        for mean, std in zip(model_avgs, model_stds):
            minus_std.append(mean - std)
            plus_std.append(mean + std)
        plt.fill_between(
            report["noise_levels"],
            minus_std,
            plus_std,
            alpha=0.2,
        )
    # plt.rcParams["figure.figsize"] = (10, 10)
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.xlabel(
        "Range of Uniform Distribution Used to Add Noise\n(as a multiple of feature STD)"
    )

    plt.xticks(report["noise_levels"])

    plt.ylabel(f"Architecture Prediction Accuracy")

    plt.title(
        f"""{arch_model_name} Architecture Prediction Accuracy by Range 
        of Uniform Distribution Used to Add Noise to Profiles"""
    )

    plt.savefig(
        REPORT_FOLDER / "plots" / report_path.name.replace(".json", ".png"),
        dpi=500,
        bbox_inches="tight",
    )


if __name__ == "__main__":

    GPU_PROFILES_PATH = (
        Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
    )
    FEATURE_RANK_PATH = (
        Path.cwd() / "plots" / "feature_ranks" / "rf_gpu_kernels_nomem.json"
    )
    NUM_FEATURES = 3  # num features to train architecture prediction models
    NOISE_FEATURES = 3  # number of features to add noise to

    NOISE_LEVELS = [0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2]
    NUM_EXPERIMENTS = 10
    OFFSET = 1
    ARCH_MODEL_NAMES = ["rf", "knn"]  # arch_model_names()

    GENERATE_REPORT = False

    # PLOT_FILENAME = REPORT_FOLDER / "20230511-140312.json"
    # PLOT_FILENAME = REPORT_FOLDER / "20230513-153017.json"
    PLOT_FILENAME = REPORT_FOLDER / "20230513-184716.json"
    PLOT = True
    PLOT_ONE_ARCH_MODEL_BY_ADV = "rf"  # None

    # ---------------------------------------------------
    if GENERATE_REPORT:
        assert NOISE_FEATURES <= NUM_FEATURES

        with open(FEATURE_RANK_PATH, "r") as f:
            report = json.load(f)["feature_rank"]
        features = report[:NUM_FEATURES]
        noise_features = report[:NOISE_FEATURES]

        df = getDF(GPU_PROFILES_PATH)
        df = filter_cols(df, substrs=features)

        generateReport(
            profile_df=df,
            noise_features=noise_features,
            num_experiments=NUM_EXPERIMENTS,
            noise_levels=NOISE_LEVELS,
            arch_model_names=ARCH_MODEL_NAMES,
            offset=OFFSET,
        )

    if PLOT:
        if PLOT_ONE_ARCH_MODEL_BY_ADV is not None:
            plotOneArchFromReport(
                report_path=PLOT_FILENAME, arch_model_name=PLOT_ONE_ARCH_MODEL_BY_ADV
            )
        else:
            plotFromReport(report_path=PLOT_FILENAME, arch_model_names=ARCH_MODEL_NAMES)
