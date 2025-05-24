"""
This tests if the ranked features from RFE are deterministic
Found out that it is deterministic.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# plt.style.use('ggplot')

# setting path
sys.path.append("../edge_profile")

from architecture_prediction import ArchPredBase, get_arch_pred_model
from config import SYSTEM_SIGNALS
from data_engineering import (
    add_indicator_cols_to_input,
    all_data,
    filter_cols,
    get_data_and_labels,
    remove_cols,
    shared_data,
)
from experiments import predictVictimArchs
from format_profiles import parse_one_profile
from utils import latest_file


def removeColumnsFromOther(keep_cols, remove_df):
    """Given two dataframes keep_cols and remove_df,
    remove each column of remove_df if that column is
    not in keep_cols.
    """
    to_remove = [x for x in remove_df.columns if x not in keep_cols.columns]
    return remove_cols(remove_df, to_remove)


def getDF(path: Path = None, to_keep_path: Path = None, save_path: Path = None):
    if to_keep_path is None:
        to_keep_path = (
            Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
        )
    if path is None:
        path = to_keep_path
    df = all_data(path, no_system_data=False)

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


if __name__ == "__main__":

    folder = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
    df = getDF(path=folder)

    model = get_arch_pred_model(
        "lr_rfe", df=df, kwargs={"rfe_num": 1, "verbose": False}
    )
    ranked_features = model.featureRank(suppress_output=True)

    global_features = set(ranked_features)

    model2 = get_arch_pred_model(
        "lr_rfe", df=df, kwargs={"rfe_num": 1, "verbose": False}
    )
    ranked_features2 = model2.featureRank(suppress_output=True)

    assert ranked_features == ranked_features2
