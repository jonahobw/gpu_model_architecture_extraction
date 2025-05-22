from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

plt.style.use("ggplot")

import sys

# setting path
sys.path.append("../edge_profile")

from model_manager import VictimModelManager
from config import MODELS

if __name__ == "__main__":
    data = []
    val_acc = []
    train_acc = []
    label = []

    # paths = VictimModelManager.getModelPaths()
    vict_folder = Path.cwd() / "models"

    for arch_folder in vict_folder.glob("*"):
        for model_instance in arch_folder.glob("*"):
            arch = arch_folder.name
            if arch not in MODELS:
                continue
            conf = VictimModelManager.loadConfig(model_instance)
            data.append((arch, conf["val_acc1"] * 100, conf["train_acc1"] * 100))

    data = sorted(data, key=lambda x: x[0])

    label = [x[0] for x in data]
    val_acc = [x[1] for x in data]
    train_acc = [x[2] for x in data]

    x = np.arange(len(label))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, val_acc, width, label="Val Acc")
    rects2 = ax.bar(x + width / 2, train_acc, width, label="Train Acc")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Victim Model Train and Validation Accuracy on Half of CIFAR10")
    ax.set_xticks(x, label)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()

    plt.show()
