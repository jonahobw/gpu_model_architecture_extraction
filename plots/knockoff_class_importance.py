"""
Generate histograms of normalized class importance from knockoff transfer sets.

This module visualizes the importance of different classes in knockoff transfer sets,
allowing comparison across multiple transfer strategies. The visualization shows how
different sampling strategies affect class importance in the transfer dataset.

The analysis includes:
1. Class importance comparison across different transfer strategies
2. Normalized importance scores for each class
3. Visualization of top-k most important classes
4. Comparison of sampling strategies (entropy, confidence, etc.)

Example Usage:
    ```python
    # Generate class importance plot for entropy-based sampling
    knockoff_params = {
        "Entropy": {
            "dataset_name": "cifar100",
            "transfer_size": 10000,
            "sample_avg": 50,
            "random_policy": False,
            "entropy": True,
        }
    }
    plotClassImportance(knockoff_params=knockoff_params, num_classes=20)
    ```
"""

import datetime
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append("../edge_profile")

from model_manager import VictimModelManager
from utils import checkDict

# Configure matplotlib settings
rc("font", **{"family": "serif", "serif": ["Times"], "size": 14})

# Constants
SAVE_FOLDER = Path(__file__).parent.absolute() / "knockoff_class_importance"
if not SAVE_FOLDER.exists():
    SAVE_FOLDER.mkdir(exist_ok=True)


def plotClassImportance(
    knockoff_params: Dict[str, Dict],
    victim_arch: str = "resnet18",
    num_classes: int = 10,
    save: bool = True,
) -> None:
    """
    Generate a bar plot comparing class importance across different knockoff transfer sets.

    Args:
        knockoff_params (Dict[str, Dict]): Dictionary mapping knockoff set names to their parameters.
            Each parameter dict should contain:
            - dataset_name: Name of the dataset
            - transfer_size: Size of the transfer set
            - sample_avg: Number of samples to average
            - random_policy: Whether to use random sampling
            - entropy: Whether to use entropy-based sampling
        victim_arch (str): Architecture of the victim model
        num_classes (int): Number of top classes to display
        save (bool): Whether to save the plot to a file

    The plot shows:
    - Bar chart of class importance for each transfer strategy
    - Comparison of importance scores across strategies
    - Top-k most important classes
    - Normalized importance scores
    """
    knockoff_names = list(knockoff_params.keys())
    knockoff_dataset = knockoff_params[knockoff_names[0]]["dataset_name"]
    
    # Validate that all knockoff sets use the same dataset
    for params in knockoff_params:
        assert knockoff_params[params]["dataset_name"] == knockoff_dataset

    # Load victim model and transfer set
    vict_path = VictimModelManager.getModelPaths(architectures=[victim_arch])[0]
    victim_manager = VictimModelManager.load(vict_path)
    idx_to_label = None

    # Load class importance data for each knockoff set
    data: Dict[str, Dict[str, float]] = {}
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

    # Sort classes by importance in the first knockoff set
    first_knockoff_data = [(k, v) for k, v in data[knockoff_names[0]].items()]
    first_knockoff_data.sort(reverse=True, key=lambda x: x[1])
    classes = [x[0] for x in first_knockoff_data][:num_classes]

    # Set up plot parameters
    x = np.arange(len(classes))  # the label locations
    width = 0.8  # the width of all bars for a single architecture
    bar_width = width / len(knockoff_params)  # width of a single bar

    # Create the plot
    fig, ax = plt.subplots()
    for i, knockoff_name in enumerate(knockoff_names):
        offset = (-1 * width / 2) + (i * bar_width) + (bar_width / 2)
        strategy_data = [data[knockoff_name][class_name] for class_name in classes]
        ax.bar(x - offset, strategy_data, bar_width, label=knockoff_name)

    # Customize the plot
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

    # Adjust layout and save/show
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(SAVE_FOLDER / f"{knockoff_dataset}_{timestamp}.png", dpi=500)
    else:
        plt.show()


if __name__ == "__main__":
    # Configuration for knockoff transfer sets
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

    # Generate the class importance plot
    plotClassImportance(knockoff_params=knockoff_params, num_classes=20)
