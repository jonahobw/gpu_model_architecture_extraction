"""
Generate a plot of architecture prediction accuracy with number of features on the x axis
and on the y axis, accuracy of the model on the test and train set, with one line per model
"""

import json
import shutil
import sys
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from tqdm import tqdm

# plt.style.use('ggplot')

# setting path
sys.path.append("../edge_profile")

from architecture_prediction import (ArchPredBase, RFArchPred,
                                     arch_model_names, get_arch_pred_model)
from config import SYSTEM_SIGNALS
from data_engineering import (add_indicator_cols_to_input, all_data,
                              filter_cols, get_data_and_labels, remove_cols,
                              removeColumnsFromOther, shared_data)
from experiments import predictVictimArchs
from format_profiles import parse_one_profile
from utils import latest_file

rc("font", **{"family": "serif", "serif": ["Times"], "size": 14})
rc("figure", **{"figsize": (5, 4)})


def getDF(
    path: Path = None,
    to_keep_path: Path = None,
    save_path: Path = None,
    gpu_activities_only=False,
):
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
    # remove cols of df if they aren't in keep_df
    df = removeColumnsFromOther(keep_df, df)

    exclude_cols = SYSTEM_SIGNALS
    # exclude_cols.extend(["mem"])
    # exclude_cols.extend(["avg_ms", "time_ms", "max_ms", "min_us"])
    # exclude_cols.extend(["memcpy", "Malloc", "malloc", "memset"])#, "avg_us", "time_ms", "max_ms", "min_us", "indicator"])
    # df = remove_cols(df, substrs=exclude_cols)
    # df = filter_cols(df, substrs=["indicator"])
    # df = filter_cols(df, substrs=["num_calls"])
    # df = filter_cols(df, substrs=["num_calls", "indicator"])
    # df = filter_cols(df, substrs=["num_calls", "time_percent", "indicator"])
    # df = filter_cols(df, substrs=["gemm", "conv", "volta", "void", "indicator", "num_calls", "time_percent"])
    # df = filter_cols(df, substrs=[
    #     "indicator_void im2col4d_kernel<float, int>(",
    #     "max_ms_void cudnn::detail::bn_fw_inf_1C11_ker",
    #     "time_ms_void cudnn::detail::implicit_convolve_s",
    #     "time_percent_void cudnn::detail::explicit_convolve",
    #     # "avg_us_[CUDA memcpy HtoD]",
    #     "indicator__ZN2at6native18elementwise_kernelILi128E",
    #     "indicator_void at::native::_GLOBAL__N__60_tmpxft_00",
    #     "num_calls_void at::native::vectorized_elementwise_k",
    #     "avg_us_void cudnn::detail::explicit_convolve_sgemm<flo",
    #     ]
    # )
    print(f"Number of remaining dataframe columns: {len(df.columns)}")
    if save_path is not None:
        df.to_csv(save_path)
    return df


def generateReport(
    df: pd.DataFrame,
    x_axis: List[int],
    feature_rank: List[str],
    model_names: List[str] = None,
    model_kwargs: dict = {},
    num_experiments: int = 10,
    save_report_name: str = None,
):
    """
    For each model in model_names, and for each # of features in range(1, feature_cap, step_size),
    trains the model with that number of features (using the top ranked features from feature_rank),
    and gets the train, val, and test mean and std across num_experiments experiments.
    """
    if model_names is None:
        model_names = arch_model_names()  # all the models we want to use

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
            print(
                f"Running {num_experiments} experiments with {num_features} features."
            )
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
            report[model_name]["train_std"] = report[model_name]["train_acc"].std(
                axis=1
            )
            report[model_name]["train_mean"] = report[model_name]["train_acc"].mean(
                axis=1
            )
            report[model_name]["val_std"] = report[model_name]["val_acc"].std(axis=1)
            report[model_name]["val_mean"] = report[model_name]["val_acc"].mean(axis=1)
            report[model_name]["test_std"] = report[model_name]["test_acc"].std(axis=1)
            report[model_name]["test_mean"] = report[model_name]["test_acc"].mean(
                axis=1
            )
    except KeyboardInterrupt:
        pass

    report["feature_rank"] = feature_rank
    report["df_cols"] = list(df.columns)
    report["num_experiments"] = num_experiments
    report["x_axis"] = x_axis

    if save_report_name is not None:

        # make numpy arrays json serializable
        def json_handler(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            raise TypeError("Unserializable object {} of type {}".format(x, type(x)))

        if not save_report_name.endswith(".json"):
            save_report_name += ".json"

        report_folder = Path(__file__).parent.absolute() / "reports"
        report_folder.mkdir(exist_ok=True)
        save_path = report_folder / save_report_name
        with open(save_path, "w") as f:
            json.dump(report, f, indent=4, default=json_handler)
    return report


def loadReport(filename: str, feature_rank: bool = False):
    if not filename.endswith(".json"):
        filename += ".json"

    report_path = Path(__file__).parent.absolute() / "reports" / filename
    if feature_rank:
        report_path = Path(__file__).parent.absolute() / "feature_ranks" / filename
    with open(report_path, "r") as f:
        report = json.load(f)
    return report


def generateFeatureRank(arch_model_name: str, df: pd.DataFrame, kwargs: dict = {}):
    kwargs.update({"rfe_num": 1})
    return get_arch_pred_model(
        arch_model_name, df=df, kwargs={"rfe_num": 1, "verbose": True}
    ).featureRank(suppress_output=True)


def saveFeatureRank(feature_rank: List[str], metadata: dict = {}, save_name=None):
    if save_name is None:
        save_name = "feature_rank.json"
    elif not save_name.endswith(".json"):
        save_name += ".json"

    report = {"feature_rank": feature_rank, **metadata}
    feature_rank_folder = Path(__file__).parent.absolute() / "feature_ranks"
    feature_rank_folder.mkdir(exist_ok=True)
    save_path = feature_rank_folder / save_name
    with open(save_path, "w") as f:
        report = json.dump(report, f, indent=4)
    return


def plotFromReport(
    report: dict,
    model_names: List[str],
    datasets: List[str] = None,
    xlim_upper: int = None,
    save_name: str = None,
    title: bool = True,
):
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

    # plt.rcParams["figure.figsize"] = (3, 3)
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
    plt.ylabel(f"Architecture Prediction Accuracy")
    if title:
        plt.title(
            f"Architecture Prediction Accuracy on {datasets_str[:-1]} Data\nby Number of Features"
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

    # ------------------------------------------------------------------------------

    load_feature_rank = (
        True  # if true, load features from file, if false, generate features and save
    )
    features_model = "rf"

    features_model_kwargs = {}
    gpu_activities_only = True
    no_memory = True
    model_kwargs = {}

    # features_filename = "combined_feature_rank_ab_lr_rf.json"
    features_filename = f"{features_model}"
    if gpu_activities_only:
        features_filename += f"_gpu_kernels"
    if no_memory:
        features_filename += "_nomem"

    # ------------------------------------------------------------------------------

    load_report = (
        True  # if true, load report from file, if false, generate report and save
    )

    # parameters for generating a report, for loading a report, only the first variable
    # needs to be set
    # report_name = "combined_report"
    report_name = features_filename

    folder = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
    model_names = arch_model_names()  # all the models we want to use
    # model_names = ["lr", "knn", "centroid", "nb"]
    num_experiments = 10
    x_axis = [i for i in range(1, 51)]
    x_axis.extend([i for i in range(60, 200, 10)])

    # ------------------------------------------------------------------------------

    # plotting
    plot = True  # whether or not to plot
    plot_model_names = model_names
    plot_datasets = ["val"]  # ['val', 'train', 'test']
    xlim_upper = 30
    plot_save_name = f"{report_name}_{plot_datasets[-1]}"
    title = False

    # ------------------------------------------------------------------------------

    df = getDF(path=folder, gpu_activities_only=gpu_activities_only)
    if no_memory:
        df = remove_cols(df, substrs=["mem"])

    if not load_feature_rank:
        # generate feature rank and save
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
        # load feature rank
        feature_rank = loadReport(features_filename, feature_rank=True)["feature_rank"]

    if not load_report:
        # generate report and save
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
        # load report
        report = loadReport(report_name)

    if plot:
        plotFromReport(
            report=report,
            model_names=plot_model_names,
            datasets=plot_datasets,
            xlim_upper=xlim_upper,
            save_name=plot_save_name,
            title=title,
        )
