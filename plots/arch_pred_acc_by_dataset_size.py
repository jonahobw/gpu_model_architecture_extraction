"""
Generate a plot of architecture prediction accuracy with number of features on the x axis
and on the y axis, accuracy of the model on the test and train set, with one line per model
"""

import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

# plt.style.use('ggplot')

# setting path
sys.path.append("../edge_profile")

from arch_pred_accuracy import getDF

from architecture_prediction import arch_model_names, get_arch_pred_model
from config import SYSTEM_SIGNALS
from data_engineering import filter_cols
from experiments import predictVictimArchs

rc("font", **{"family": "serif", "serif": ["Times"], "size": 14})
rc("figure", **{"figsize": (5, 4)})

REPORT_FOLDER = Path(__file__).parent.absolute() / "arch_pred_acc_by_dataset_size"
REPORT_FOLDER.mkdir(exist_ok=True)
plot_folder = REPORT_FOLDER / "plots"
plot_folder.mkdir(exist_ok=True)


def loadReport(filename: str, feature_rank: bool = False):
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
    model_names: List[str] = None,
    model_kwargs: dict = {},
    num_experiments: int = 10,
    step_size: int = 1,
    dataset_start: int = 10,
    dataset_cap: int = 38,
    save_report_name: str = None,
):
    """
    For each model in model_names, and for each # of profiles in the dataset per architecture
    in range(1, feature_cap, step_size), trains the model with that number of profiles
    and gets the train, val, and test mean and std across num_experiments experiments.
    """

    if not save_report_name.endswith(".json"):
        save_report_name += ".json"

    if model_names is None:
        model_names = arch_model_names()  # all the models we want to use

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
                    # only need to run one experiment, copy the result to the array
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

    for model_name in report:
        report[model_name]["train_std"] = report[model_name]["train_acc"].std(axis=1)
        report[model_name]["train_mean"] = report[model_name]["train_acc"].mean(axis=1)
        report[model_name]["val_std"] = report[model_name]["val_acc"].std(axis=1)
        report[model_name]["val_mean"] = report[model_name]["val_acc"].mean(axis=1)
        report[model_name]["test_std"] = report[model_name]["test_acc"].std(axis=1)
        report[model_name]["test_mean"] = report[model_name]["test_acc"].mean(axis=1)

    report["feature_rank"] = feature_rank
    report["df_cols"] = list(df.columns)
    report["num_experiments"] = num_experiments
    report["step_size"] = step_size
    report["dataset_cap"] = dataset_cap
    report["x_axis"] = x_axis

    if save_report_name is not None:

        # make numpy arrays json serializable
        def json_handler(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            raise TypeError("Unserializable object {} of type {}".format(x, type(x)))

        save_path = REPORT_FOLDER / save_report_name
        with open(save_path, "w") as f:
            json.dump(report, f, indent=4, default=json_handler)
    return report


def plotFromReport(
    report: dict,
    model_names: List[str],
    datasets: List[str] = None,
    xlim_upper: int = None,
    save_name: str = None,
    title: bool = True,
):
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
    # plt.rcParams["figure.figsize"] = (10, 10)
    plt.tight_layout()
    plt.legend(loc="lower right")
    plt.xlabel("Number of Profiles per Architecture in Training Dataset")

    x_axis_lim = max(x_axis) if xlim_upper is None else xlim_upper

    interval = x_axis_lim // 7
    ticks = [x for x in range(0, x_axis_lim, interval)]
    ticks[0] = 1
    ticks.append(x_axis_lim)
    plt.xticks(ticks)

    plt.ylabel(f"Architecture Prediction Accuracy")
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
            f"Architecture Prediction Accuracy on {datasets_str[:-1]} Data\nby Number of Profiles per Architecture in Train Dataset"
        )
    if xlim_upper is not None:
        plt.xlim(right=xlim_upper)
    plt.savefig(REPORT_FOLDER / "plots" / save_name, dpi=500, bbox_inches="tight")


if __name__ == "__main__":

    folder = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
    model_names = arch_model_names()  # all the models we want to use
    # model_names = ["lr"]
    num_experiments = 10
    step_size = 1
    dataset_start = 1
    # the upper limit on number of profiles per architecture in training set
    # should not be above since we have 50 profiles, a 75/25 split is
    # 38/12, so number of training profiles is limited 1-38
    dataset_cap = 38

    # load feature rank, must be generated by arch_pred_accuracy.py
    features_filename = f"rf_gpu_kernels_nomem"
    num_features = 25  # how many features to use
    model_kwargs = {}
    report_name = features_filename

    load_report = (
        True  # if true, load report from file, if false, generate report and save
    )
    # plotting
    plot = True  # whether or not to plot
    plot_model_names = model_names
    plot_datasets = ["test"]  # ['val', 'train', 'test']
    xlim_upper = None
    title = False

    # ---------------------------------------------------------------------

    df = getDF(path=folder)

    feature_rank = loadReport(features_filename, feature_rank=True)["feature_rank"]
    selected_features = feature_rank[:num_features]
    df = filter_cols(df, substrs=selected_features)

    if not load_report:
        # generate report and save
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
        # load report
        report = loadReport(report_name)

    if plot:
        plotFromReport(
            report=report,
            model_names=plot_model_names,
            datasets=plot_datasets,
            xlim_upper=xlim_upper,
            save_name=f"{features_filename}_{plot_datasets[-1]}.png",
            title=title,
        )
