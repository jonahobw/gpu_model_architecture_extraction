"""
Histogram of normalized class importance
from a knockoff transfer set. Can compare
across multiple transfer sets, but they
must all sample from the same dataset.
"""

import datetime
import json
from pathlib import Path
import shutil
from typing import List, Union
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
import matplotlib.pyplot as plt
from matplotlib import rc

# plt.style.use('ggplot')

# setting path
sys.path.append("../edge_profile")

from model_manager import VictimModelManager
from utils import checkDict

rc("font", **{"family": "serif", "serif": ["Times"], "size": 14})
# rc('figure', **{'figsize': (5, 4)})

SAVE_FOLDER = Path(__file__).parent.absolute() / "knockoff_class_importance"
if not SAVE_FOLDER.exists():
    SAVE_FOLDER.mkdir(exist_ok=True)


def plotClassImportance(
    knockoff_params: dict,
    victim_arch: str = "resnet18",
    num_classes: int = 10,
    save: bool = True,
):
    """
    knockoff params is a dict of
    {knockoff_set_name: {parameters for the knockoff}}
    """
    knockoff_names = list(knockoff_params.keys())
    knockoff_dataset = knockoff_params[knockoff_names[0]]["dataset_name"]
    for params in knockoff_params:
        assert knockoff_params[params]["dataset_name"] == knockoff_dataset
    vict_path = VictimModelManager.getModelPaths(architectures=[victim_arch])[0]
    victim_manager = VictimModelManager.load(vict_path)
    idx_to_label = None

    # this will be of the format
    # {knockoff_name: {class_name: importance}}

    data = {}
    for knockoff_name in knockoff_params:
        file, transfer_set = victim_manager.loadKnockoffTransferSet(
            **knockoff_params[knockoff_name], force=True
        )
        if idx_to_label is None:
            idx_to_label = {
                v: k for k, v in transfer_set.train_data.dataset.class_to_idx.items()
            }
        with open(file, "r+") as f:
            conf = json.load(f)
        data[knockoff_name] = {
            idx_to_label[i]: conf["class_importance"][i]
            for i in range(len(conf["class_importance"]))
        }

    # sort the importances of the first knockoff_name
    first_knockoff_data = [(k, v) for k, v in data[knockoff_names[0]].items()]
    first_knockoff_data.sort(reverse=True, key=lambda x: x[1])
    classes = [x[0] for x in first_knockoff_data][:num_classes]

    x = np.arange(len(classes))  # the label locations
    width = 0.8  # the width of all bars for a single architecture
    bar_width = width / len(knockoff_params)  # width of a single bar

    fig, ax = plt.subplots()
    for i, knockoff_name in enumerate(knockoff_names):
        offset = (-1 * width / 2) + (i * bar_width) + (bar_width / 2)
        strategy_data = [data[knockoff_name][class_name] for class_name in classes]
        ax.bar(x - offset, strategy_data, bar_width, label=knockoff_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Class Importance")

    if len(knockoff_params) == 1:
        ax.set_title(
            f"Class Importance for {next(iter(knockoff_params))} Transfer Dataset"
        )
    else:
        ax.set_title("Class Importance by Transfer Set")
        ax.legend()
    ax.set_xticks(x, classes)
    ax.set_xlabel(f"Top {knockoff_dataset} Classes")

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(SAVE_FOLDER / f"{knockoff_dataset}_{timestamp}.png", dpi=500)
    else:
        plt.show()


if __name__ == "__main__":

    # this is a set of args to match with model manager config
    # values.
    knockoff_params = {
        "Entropy": {
            "dataset_name": "cifar100",
            "transfer_size": 10000,
            "sample_avg": 50,
            "random_policy": False,
            "entropy": True,
        },
        # "Confidence": {
        #     "dataset_name": "cifar100",
        #     "transfer_size": 10000,
        #     "sample_avg": 50,
        #     "random_policy": False,
        #     "entropy": True,
        # },
    }

    plotClassImportance(knockoff_params=knockoff_params, num_classes=20)
