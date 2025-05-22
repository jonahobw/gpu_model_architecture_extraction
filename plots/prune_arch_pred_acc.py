"""
Generate a table of accuracy on pruned victim models
top1/top5 columns by architecture prediction model type (rows).

The features used and dataset size are configurable. 
"""

import datetime
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.metrics import top_k_accuracy_score

# plt.style.use('ggplot')

# setting path
sys.path.append("../edge_profile")

from architecture_prediction import (ArchPredBase, arch_model_full_name,
                                     arch_model_names, get_arch_pred_model)
from data_engineering import (all_data, filter_cols, remove_cols,
                              removeColumnsFromOther)
from experiments import predictVictimArchs
from model_manager import VictimModelManager

SAVE_FOLDER = Path(__file__).parent.absolute() / "prune_pred_acc"
if not SAVE_FOLDER.exists():
    SAVE_FOLDER.mkdir(exist_ok=True)

QUADRO_TRAIN = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
QUADRO_PRUNE_TEST = Path.cwd() / "victim_profiles_pruned"


def loadFeatureRank(filename: str):
    if not filename.endswith(".json"):
        filename += ".json"
    report_path = Path(__file__).parent.absolute() / "feature_ranks" / filename
    with open(report_path, "r") as f:
        report = json.load(f)
    return report["feature_rank"]


def getDF(path: Path, to_keep_path: Path = None, df_args: dict = {}):
    df = all_data(path, **df_args)
    if to_keep_path is not None:
        keep_df = all_data(to_keep_path, **df_args)
        # remove cols of df if they aren't in keep_df
        df = removeColumnsFromOther(keep_df, df)
    return df


def generatePruneReport(
    quadro_train: pd.DataFrame,
    config: dict,
    model_names: List[str],
    topk: List[int] = [1, 5],
    train_size: int = None,
):
    # for now, only topk will be evaluated on test dataset.

    columns = [
        "Architecture Prediction Model Type",
        "Train/Test",
        "Train",
        "Test",
        "Test Family Accuracy",
    ]

    for k in topk:
        # columns.append(f"Train Top {k}")
        columns.append(f"Test Top {k}")

    table = pd.DataFrame(columns=columns)

    for model_type in model_names:
        row_data = {
            "Architecture Prediction Model Type": arch_model_full_name()[model_type]
        }
        # train model
        model = get_arch_pred_model(
            model_type=model_type, df=quadro_train, kwargs={"train_size": train_size}
        )
        train_top1 = model.evaluateTrain() * 100
        row_data["Train"] = train_top1

        # test
        test_k_report = predictVictimArchs(
            model, QUADRO_PRUNE_TEST, save=False, topk=max(topk), verbose=False
        )
        k_acc = test_k_report["accuracy_k"]

        row_data["Test Family Accuracy"] = test_k_report["family_accuracy"] * 100

        topk_str = ""
        for k in topk:
            row_data[f"Test Top {k}"] = k_acc[k] * 100
            topk_str += "{:.3g}/".format(k_acc[k] * 100)

        row_data["Test"] = topk_str[:-1]

        row_data["Train/Test"] = "{:.3g}/{:.3g}".format(train_top1, k_acc[1] * 100)

        table = table.append(row_data, ignore_index=True)

    # save table and config
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    table.to_csv(SAVE_FOLDER / f"{time}.csv")
    transpose_path = SAVE_FOLDER / f"{time}_transpose.csv"
    table.T.to_csv(transpose_path)

    config["topk"] = topk
    with open(SAVE_FOLDER / f"{time}.json", "w") as f:
        json.dump(config, f, indent=4)


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

    if feature_ranking_file is not None:
        feature_ranking = loadFeatureRank(feature_ranking_file)
        relevant_features = feature_ranking[:feature_num]

        quadro_train = filter_cols(quadro_train, substrs=relevant_features)

    quadro_train = remove_cols(quadro_train, substrs=df_remove_substrs)

    config = {
        "train_size": train_size,
        "feature_ranking_file": feature_ranking_file,
        "feature_num": feature_num,
        "df_args": df_args,
        "df_remove_substrs": df_remove_substrs,
        "model_names": model_names,
    }

    generatePruneReport(
        quadro_train=quadro_train,
        config=config,
        model_names=model_names,
        topk=topk,
        train_size=train_size,
    )
