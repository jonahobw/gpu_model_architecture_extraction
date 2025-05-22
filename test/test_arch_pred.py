import json
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

# setting path
sys.path.append("../edge_profile")

from architecture_prediction import ArchPredBase, get_arch_pred_model
from config import SYSTEM_SIGNALS
from data_engineering import (add_indicator_cols_to_input, all_data,
                              filter_cols, get_data_and_labels, remove_cols,
                              shared_data)
from experiments import predictVictimArchs
from format_profiles import parse_one_profile
from utils import latest_file


def parse_prof(profile_csv=None, config_file=None, gpu=0):
    if profile_csv is None:
        profile_csv = Path(
            "/Users/jgow98/Library/CloudStorage/OneDrive-UniversityofMassachusetts/2Grad/research/model_extraction/code/edge_profile/models/resnet50/resnet50_20221012-014421/profiles/profile_1802702.csv"
        )
    if config_file is not None:
        with open(config_file, "r") as f:
            conf = json.load(f)
        gpu = conf["gpu"]
    return parse_one_profile(profile_csv, gpu=gpu)


def series_to_df(ser):
    mapping = {ind: val for ind, val in ser.items()}
    res = pd.DataFrame(mapping, index=[0])
    return res


def get_indicator_cols(df):
    for col in df.columns:
        if not col.startswith("indicator_"):
            df.drop(columns=[col], inplace=True)
    return df


def check_indicator_cols_consistency(arch_pred_model, model="resnet50"):
    train_raw = arch_pred_model.data.drop(columns=["model_family", "file"])
    train_raw = train_raw[train_raw["model"] == model]

    indicators = get_indicator_cols(train_raw)
    indicator_std = indicators.std()
    num_indicators = indicator_std.size
    avg = indicator_std.sum()
    print(
        f"model: {model}\tindicator std sum: {avg}\tnumber of indicators: {num_indicators}"
    )
    if avg > 0:
        raise ValueError
    return indicators.mean()


def check_indicator_cols_discrepancy(arch_pred_model, profile_features):
    """For a given profile <profile_features>, checks to see which features are not in both
    <profile_features> and the training data"""

    indicators = series_to_df(check_indicator_cols_consistency(arch_pred_model))
    train_raw = arch_pred_model.data.drop(columns=["model_family", "file"])
    raw_features = add_indicator_cols_to_input(
        train_raw, profile_features, exclude=["model"]
    )
    df = series_to_df(raw_features)
    data_indicators = get_indicator_cols(df)
    compare = (
        (indicators - data_indicators).abs().sort_values(by=0, axis=1, ascending=False)
    )

    for col in compare:
        if compare[col].item() > 0:
            print(
                f"Different\t{col[:50].ljust(50)}\tIn data {int(data_indicators[col].item())}\tIn training data {int(indicators[col].item())}"
            )
        else:
            break
    return compare


def replace_indicators(train_data, x, exclude=[], model="resnet50"):

    model_data = train_data[train_data["model"] == model]
    model_data = model_data.drop(columns=["model"]).mean()

    check_indicator_cols_consistency(model)

    for col in train_data.columns:
        if col not in x.keys() and col != "model":
            x[col] = model_data[col]
            if col.startswith("indicator_"):
                original_name = col.split("indicator_")[1]
                x[original_name] = model_data[original_name]

    for i in x.keys():
        if i not in train_data.columns:
            x = x.drop(i)

    # sort x in order of the keys of the df.
    x = pd.DataFrame([x])
    new_df = pd.concat((train_data, x), ignore_index=True)

    result = new_df.iloc[-1]
    result = result.drop(exclude)

    return result


def model_mean(train_data, model="resnet50"):
    model_data = train_data[train_data["model"] == model]
    model_data = model_data.drop(columns=["model"]).mean()
    return model_data


def compare_normalized(arch_pred_model, profile_features):
    # check normalized data against normalized training data
    full_features = arch_pred_model.preprocessInput(profile_features)
    full_features_normalized = arch_pred_model.model.normalize(full_features)
    train = pd.DataFrame(arch_pred_model.x_tr)
    train_labels = arch_pred_model.label_encoder.inverse_transform(
        arch_pred_model.y_train
    )
    train["label"] = train_labels
    means = train.groupby("label").mean()
    mean_diff = (means - full_features_normalized).abs()
    mean_sum = mean_diff.sum(axis=1).sort_values()
    return mean_sum, mean_diff


def compare_raw(arch_pred_model, profile_features):
    # check raw data against raw training data
    train_raw = arch_pred_model.data.drop(columns=["model_family", "file"])
    raw_mean = train_raw.groupby("model").mean()
    raw_features = add_indicator_cols_to_input(
        train_raw, profile_features, exclude=["model"]
    )
    raw_mean_diff = (raw_mean - raw_features).abs()
    raw_mean_sum = raw_mean_diff.sum(axis=1).sort_values()
    raw_mean_diff_label = raw_mean_diff.loc["resnet50"].sort_values()


def compare_same_indicators(arch_pred_model, train_raw, profile_features):
    # check new datapoint with same indicators
    new_datapoint = replace_indicators(train_raw, profile_features, exclude=["model"])
    new_datapoint = new_datapoint.to_numpy(dtype=np.float32)
    new_datapoint = np.expand_dims(new_datapoint, axis=0)
    arch, conf = arch_pred_model.predict(new_datapoint, preprocess=False)
    print(f"Predicted new datapoint architecture {arch} with confidence {conf}.")


def predict_model_mean(arch_pred_model):
    # check model mean
    resnet50_data = model_mean(
        arch_pred_model.data.drop(columns=["model_family", "file"])
    )
    resnet50_data = resnet50_data.to_numpy(dtype=np.float32)
    resnet50_data = np.expand_dims(resnet50_data, axis=0)
    arch, conf = arch_pred_model.predict(resnet50_data, preprocess=False)
    print(f"Predicted resnet50 datapoint architecture {arch} with confidence {conf}.")


def predict_train_data(arch_pred_model, num_profs=5):
    count = 0
    model_folder = Path.cwd() / "profiles" / "zero_noexe_lots_models"
    for vict_arch in model_folder.glob("*"):
        vict_folder = model_folder / vict_arch
        for prof in vict_arch.glob("*"):
            features = parse_one_profile(vict_folder / prof, gpu=0)
            arch, conf = arch_pred_model.predict(features)
            print(
                f"Profile {prof.name} Predicted architecture {arch} with confidence {conf}."
            )
            count += 1
            if count > num_profs:
                count = 0
                break

        print("\n")


def f(x, arch_model, model="resnet50"):
    # compute the arch_pred_model's confidence that the input x is
    # <model>.  x should be raw data dataframe with all features
    x = x.copy()
    x = x.to_numpy(dtype=np.float32)
    # x = np.expand_dims(x, axis=0)

    y_true = arch_model.label_encoder.transform([model])[0]
    y_preds = arch_model.model.get_preds(x).cpu()

    y_true_conf = y_preds[y_true].item()

    y_pred = y_preds.argmax()
    y_pred_conf = y_preds[y_pred].item()
    y_pred_label = arch_model.label_encoder.inverse_transform(np.array([y_pred]))[0]
    return y_true_conf, y_pred_label, y_pred_conf


def compute_grad(x, step_size, arch_model, model="resnet50"):
    """Computes the gradient of x, perturbing each dimension <step_size>"""

    grad = x.copy()
    new_x = x.copy()

    orig_conf, _, _ = f(x, arch_model, model)

    for col in tqdm(x.columns):
        col_step = step_size[col]
        if col.startswith("indicator"):
            if new_x[col].item() == 1:
                new_x[col] = 0
            else:
                new_x[col] = 1
        else:
            new_x[col] += col_step
        elem_grad, _, _ = f(new_x, arch_model, model)
        grad[col] = (elem_grad - orig_conf) / col_step
        if grad[col].item() != 0:
            print(f"Grad for\n{col}\nis {grad[col].item()}")
        if col.startswith("indicator"):
            if new_x[col].item() == 1:
                new_x[col] = 0
            else:
                new_x[col] = 1
        else:
            new_x[col] -= col_step

    return grad


def compute_grad_model_mean(x, model_mean, arch_model, model="resnet50", eps=1e-4):
    """Computes the gradient of x, changing each dimension to the true class mean.
    Doesn't change indicators"""

    grad = x.copy()
    new_x = x.copy()

    orig_conf, _, _ = f(x, arch_model, model)

    for col in tqdm(x.columns):
        if not col.startswith("indicator"):
            model_col = model_mean[col].item()
            col_step = model_col - new_x[col].item()
            new_x[col] += col_step
            elem_grad, _, _ = f(new_x, arch_model, model)
            divisor = col_step
            if divisor == 0:
                divisor = eps
            grad[col] = (elem_grad - orig_conf) / divisor
            if grad[col].item() != 0:
                print(f"Grad for\n{col}\nis {grad[col].item()}")
            new_x[col] -= col_step
        else:
            grad[col] = 0

    return grad


def transform_input(arch_model, model="resnet50", eps=1e-4, step=0.5, use_means=False):
    """Takes a misclassified input and iteratively perturbs it until
    it is classified correctly.  Returns perturbed input"""
    profile_features = parse_prof()
    # first construct input
    train_raw = arch_model.data.drop(columns=["model_family", "file"])
    model_means = series_to_df(model_mean(train_raw, model=model))
    x = add_indicator_cols_to_input(train_raw, profile_features, exclude=["model"])
    x = series_to_df(x)
    orig = x.copy()

    # step size is proportional to data values
    step_size = (x * step) + eps

    y_true_conf, y_pred_label, y_pred_conf = f(x, arch_model, model)
    iter = 1

    while y_pred_label != model:
        print(
            f"Iter {iter}\t{model} conf: {y_true_conf}\tPredicted {y_pred_label} with conf {y_pred_conf}"
        )
        if use_means:
            grad = compute_grad_model_mean(x, model_means, arch_model, model, eps)
        else:
            grad = compute_grad(x, step_size, arch_model, model)
        whole_grad = grad.sum(axis=1).item()
        print(f"Sum of grad: {whole_grad}")
        x += grad * step_size
        if whole_grad < eps:
            step_size *= 2
            print("Grad was 0, increasing step size by a factor of 2.")
        else:
            y_true_conf, y_pred_label, y_pred_conf = f(x, arch_model, model)
        iter += 1

    return orig, x


def analyze_diff(orig_x, new_x):
    diff = (orig_x - new_x).abs()
    return diff.sort_values(by=0, axis=1, ascending=False)


def replace_non_indicators(train_data, x, model="resnet50"):
    """Replaces non indicators with model mean"""
    train_data = train_data[train_data["model"] == model]
    train_data = train_data.drop(columns=["model"]).mean()
    train_data = series_to_df(train_data)

    for col in train_data.columns:
        if not col.startswith("indicator"):
            x[col] = train_data[col]

    return x


def check_replace_system_sigals(arch_pred_model, model="resnet50"):
    """
    Replaces all system signals with model mean
    """
    profile_features = parse_prof()
    # first construct input
    train_raw = arch_pred_model.data.drop(columns=["model_family", "file"])
    model_means = series_to_df(model_mean(train_raw, model=model))
    x = add_indicator_cols_to_input(train_raw, profile_features, exclude=["model"])
    x = series_to_df(x)
    for col in model_means.columns:
        for sig in SYSTEM_SIGNALS:
            if col.find(sig) >= 0:
                x[col] = model_means[col]
                break
    x = x.to_numpy(dtype=np.float32)
    y_preds = arch_pred_model.model.get_preds(x).cpu()
    y_true = arch_pred_model.label_encoder.transform([model])[0]
    y_true_conf = y_preds[y_true].item()

    y_pred = y_preds.argmax()
    y_pred_conf = y_preds[y_pred].item()
    y_pred_label = arch_pred_model.label_encoder.inverse_transform(np.array([y_pred]))[
        0
    ]
    print(
        f"{model} conf: {y_true_conf}\tPredicted {y_pred_label} with conf {y_pred_conf}"
    )


def predict_non_indicators_replaced(arch_pred_model, model="resnet50"):
    """
    Changes all non indicator columns to have model mean
    """
    profile_features = parse_prof()
    # first construct input
    train_raw = arch_pred_model.data.drop(columns=["model_family", "file"])
    x = add_indicator_cols_to_input(train_raw, profile_features, exclude=["model"])
    x = series_to_df(x)
    x = replace_non_indicators(train_raw, x, model=model)
    x = x.to_numpy(dtype=np.float32)
    y_preds = arch_pred_model.model.get_preds(x).cpu()
    y_true = arch_pred_model.label_encoder.transform([model])[0]
    y_true_conf = y_preds[y_true].item()

    y_pred = y_preds.argmax()
    y_pred_conf = y_preds[y_pred].item()
    y_pred_label = arch_pred_model.label_encoder.inverse_transform(np.array([y_pred]))[
        0
    ]
    print(
        f"{model} conf: {y_true_conf}\tPredicted {y_pred_label} with conf {y_pred_conf}"
    )


def check_specific_cols_model_mean(
    arch_pred_model, model="resnet50", cols=["avg_fan_(%)"]
):
    """
    changes <cols> to have the model mean from the training data
    """
    profile_features = parse_prof()
    # first construct input
    train_raw = arch_pred_model.data.drop(columns=["model_family", "file"])
    model_means = series_to_df(model_mean(train_raw, model=model))
    x = add_indicator_cols_to_input(train_raw, profile_features, exclude=["model"])
    x = series_to_df(x)
    for col in cols:
        x[col] = model_means[col]
    x = x.to_numpy(dtype=np.float32)
    y_preds = arch_pred_model.model.get_preds(x).cpu()
    y_true = arch_pred_model.label_encoder.transform([model])[0]
    y_true_conf = y_preds[y_true].item()

    y_pred = y_preds.argmax()
    y_pred_conf = y_preds[y_pred].item()
    y_pred_label = arch_pred_model.label_encoder.inverse_transform(np.array([y_pred]))[
        0
    ]
    print(
        f"{model} conf: {y_true_conf}\tPredicted {y_pred_label} with conf {y_pred_conf}"
    )


def predict_all_victim_profiles(arch_model: ArchPredBase, exclude_cols=[]):
    data = all_data("zero_noexe_lots_models")
    for col in data.columns:
        for excl in exclude_cols:
            if col.find(excl) >= 0:
                data.drop(columns=[col], inplace=True)
                break

    correct = 0
    tested = 0
    incorrect = []
    model_folder = Path.cwd() / "models"

    for vict_arch in model_folder.glob("*"):
        arch_folder = model_folder / vict_arch
        for vict_model in arch_folder.glob("*"):
            vict_path = arch_folder / vict_model
            profiles_path = vict_path / "profiles"
            if not profiles_path.exists():
                break
            params = [x for x in profiles_path.glob("params*")][0]
            profile = [x for x in profiles_path.glob("profile*")][0]
            # if len(params) == 0 or len(profile) == 0:
            #     break
            features = parse_prof(profile, params)
            arch, conf = arch_model.predict(features)
            print(
                f"{vict_arch.name} Profile {profile.name} Predicted architecture {arch} with confidence {conf}.\n"
            )
            tested += 1
            if arch == vict_arch.name:
                correct += 1
            else:
                incorrect.append((vict_arch.name, profile.name, arch, conf))

    print(
        f"{correct}/{tested} correct, ({100 * (correct/tested)}%)\nIncorrect:\n{incorrect}"
    )
    return arch_model


def feature_correlation(arch_pred_model: ArchPredBase):
    df = arch_pred_model.data.drop(columns=["model_family", "file"])
    transform_y_fn = arch_pred_model.label_encoder.fit_transform
    y = df["model"].to_numpy()
    y = transform_y_fn(y)
    df["model"] = y

    # transform label data

    corr = df.iloc[0]
    for col in df.columns:
        if col in ["model_family", "file", "model"]:
            continue
        col_corr = df["model"].corr(df[col])
        corr[col] = col_corr
    corr = corr.sort_values(ascending=False)
    return corr


def removeColumnsFromOther(keep_cols, remove_df):
    """Given two dataframes keep_cols and remove_df,
    remove each column of remove_df if that column is
    not in keep_cols.
    """
    to_remove = [x for x in remove_df.columns if x not in keep_cols.columns]
    return remove_cols(remove_df, to_remove)


def combineProfiles(folder: Path = None, weight: int = 1):
    """
    Puts all the profiles into one folder so that they can all be put into one csv
    <weight> allows the profiles from the model managers to be copied multiple times
    so that their distribution is accurately represented.
    """
    if folder is None:
        folder = Path.cwd() / "profiles" / "all_profiles"
    assert not folder.exists()
    folder.mkdir()

    # first copy the profile folders
    profile_folder1 = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe"
    profile_folder2 = Path.cwd() / "profiles" / "tesla_t4" / "colab_zero_exe"
    profile_folder3 = Path.cwd() / "profiles" / "zero_noexe_lots_models"

    try:
        profile_folders = [profile_folder1, profile_folder2, profile_folder3]
        count = 0
        for prof_fold in profile_folders:
            for model_arch_folder in prof_fold.glob("*"):
                new_model_arch_folder = (
                    folder / model_arch_folder.name
                )  # this is /<folder>/alexnet for example
                new_model_arch_folder.mkdir(exist_ok=True)
                for profile in model_arch_folder.glob("*"):
                    new_profile_name = f"profile_{count}.csv"
                    new_profile_path = new_model_arch_folder / new_profile_name
                    shutil.copy(profile, new_profile_path)
                    count += 1

        # TODO use the loadProfilestofolder method from model manager instead of the things below.

        # now copy the model manager profiles
        mangers_path1 = Path.cwd() / "models"

        managers_folders = [mangers_path1]
        for manager_fold in managers_folders:  # example ./models
            for model_arch_folder in manager_fold.glob("*"):  # example ./models/alexnet
                new_model_arch_folder = (
                    folder / model_arch_folder.name
                )  # this is /<folder>/alexnet for example
                for model_instance in model_arch_folder.glob(
                    "*"
                ):  # example ./models/alexnet/alexnet_20221209
                    manager_prof_folder = model_instance / "profiles"
                    for profile in manager_prof_folder.glob("profile_*"):
                        for i in range(weight):
                            new_profile_name = f"profile_{count}.csv"
                            new_profile_path = new_model_arch_folder / new_profile_name
                            shutil.copy(profile, new_profile_path)
                            count += 1
    except Exception as e:
        shutil.rmtree(folder)
        raise e


def getDF(path: Path = None, save_path: Path = None):
    to_keep_path = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
    if path is None:
        path = to_keep_path
    df = all_data(path, no_system_data=False)

    keep_df = all_data(to_keep_path)
    # remove cols of df if they aren't in keep_df
    df = removeColumnsFromOther(keep_df, df)

    exclude_cols = SYSTEM_SIGNALS
    exclude_cols.extend(["mem"])
    # exclude_cols.extend(["avg_ms", "time_ms", "max_ms", "min_us"])
    exclude_cols.extend(
        ["memcpy", "Malloc", "malloc", "memset"]
    )  # , "avg_us", "time_ms", "max_ms", "min_us", "indicator"])
    df = remove_cols(df, substrs=exclude_cols)
    # df = filter_cols(df, substrs=["indicator"])
    # df = filter_cols(df, substrs=["num_calls"])
    df = filter_cols(df, substrs=["num_calls", "indicator"])
    # df = filter_cols(df, substrs=["num_calls", "time_percent", "indicator"])
    # df = filter_cols(df, substrs=["gemm", "conv", "volta", "void", "indicator", "num_calls", "time_percent"])
    # df = filter_cols(df, substrs=["void im2col4d_kernel<float, int>(im2col4d_params"])
    print(f"Number of remaining dataframe columns: {len(df.columns)}")
    if save_path is not None:
        df.to_csv(save_path)
    return df


def saveProfileFeatures(df, arch: str):
    profile_folder = Path.cwd() / "victim_profiles"
    profile_csv = latest_file(profile_folder, pattern=f"*{arch}*")
    features = parse_one_profile(profile_csv)
    features = add_indicator_cols_to_input(
        df, features, exclude=["file", "model", "model_family"]
    )
    features = series_to_df(features)
    features.to_csv(Path.cwd() / f"temp_profile_{arch}.csv")


if __name__ == "__main__":
    # arch_model = gen_arch_pred_model()
    # profile_features = parse_prof()
    # predict_from_prof(profile_features, arch_model)

    # try_all_arch_without_signals()
    # predict_non_indicators_replaced(arch_model)
    # check_specific_cols_model_mean(arch_model)
    # check_replace_system_sigals(arch_model)

    # orig, x = transform_input(arch_model=arch_model, use_means=True)
    # diff = analyze_diff(orig, x)

    save_path = None
    # save_path = folder = Path.cwd() / "temp_agg_data.csv"

    # this is the training data
    # folder = Path.cwd() / "profiles" / "all_profiles"
    folder = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
    df = getDF(path=folder, save_path=save_path)

    # saveProfileFeatures(df, "resnet50")

    # model = get_arch_pred_model("nn", df=df)
    model = get_arch_pred_model(
        "nn", df=df, kwargs={"num_layers": 5, "hidden_layer_factor": 1}
    )
    # model = get_arch_pred_model("lr", df=df, kwargs={"multi_class": 'multinomial', "penalty": "l2"})    # these are the default args
    # model = get_arch_pred_model("lr", df=df, kwargs={"multi_class": 'multinomial', "penalty": "none"})
    # model = get_arch_pred_model("lr", df=df, kwargs={"multi_class": 'ovr', "penalty": "none"})
    # model = get_arch_pred_model("lr", df=df, kwargs={"multi_class": 'ovr', "penalty": "l2"})
    # model = get_arch_pred_model("lr_rfe", df=df, kwargs={"rfe_num": 800, "verbose": False})
    # model.printFeatureRank(save_path=Path.cwd(), suppress_output=True)
    model.printFeatures()
    model.evaluateTest()
    a = input(f"Enter anything to continue")

    folder = Path.cwd() / "victim_profiles"

    predictVictimArchs(model, folder, save=False)
