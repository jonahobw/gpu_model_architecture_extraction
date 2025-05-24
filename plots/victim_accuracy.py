"""
Visualize training and validation accuracy of victim models.

This module generates bar plots comparing training and validation accuracy
across different model architectures. It analyzes victim model configurations
to extract and visualize accuracy metrics.

Key Features:
1. Comparison of training vs validation accuracy
2. Support for multiple model architectures
3. Configurable model selection
4. Clear visualization of accuracy differences
5. Automatic data collection from model configurations

Example Usage:
    ```python
    # Generate accuracy comparison plot
    data = collectModelAccuracies()
    plotAccuracyComparison(data)
    ```
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Configure plot style
plt.style.use("ggplot")

import sys

# Add parent directory to path for imports
sys.path.append("../edge_profile")

from config import MODELS
from model_manager import VictimModelManager


def collectModelAccuracies() -> List[Tuple[str, float, float]]:
    """
    Collect training and validation accuracy from victim model configurations.

    Returns:
        List[Tuple[str, float, float]]: List of (architecture, val_acc, train_acc) tuples
        where accuracies are in percentage (0-100)

    Example:
        ```python
        data = collectModelAccuracies()
        # Returns: [('resnet18', 95.5, 98.2), ('vgg16', 94.8, 97.9), ...]
        ```
    """
    data = []
    vict_folder = Path.cwd() / "models"

    for arch_folder in vict_folder.glob("*"):
        for model_instance in arch_folder.glob("*"):
            arch = arch_folder.name
            if arch not in MODELS:
                continue
            conf = VictimModelManager.loadConfig(model_instance)
            data.append((arch, conf["val_acc1"] * 100, conf["train_acc1"] * 100))

    return sorted(data, key=lambda x: x[0])


def plotAccuracyComparison(data: List[Tuple[str, float, float]]) -> None:
    """
    Generate a bar plot comparing training and validation accuracy across models.

    Args:
        data (List[Tuple[str, float, float]]): List of (architecture, val_acc, train_acc) tuples
        where accuracies are in percentage (0-100)

    The plot shows:
    - Training accuracy bars
    - Validation accuracy bars
    - Model architecture labels
    - Clear legend and title
    """
    # Extract data for plotting
    labels = [x[0] for x in data]
    val_acc = [x[1] for x in data]
    train_acc = [x[2] for x in data]

    # Configure plot
    x = np.arange(len(labels))
    width = 0.35

    # Create plot
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, val_acc, width, label="Val Acc")
    rects2 = ax.bar(x + width / 2, train_acc, width, label="Train Acc")

    # Customize plot appearance
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Victim Model Train and Validation Accuracy on Half of CIFAR10")
    ax.set_xticks(x, labels)
    ax.legend()
    plt.xticks(rotation=45, ha="right")

    # Adjust layout
    fig.tight_layout()

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Collect accuracy data
    data = collectModelAccuracies()

    # Generate and display plot
    plotAccuracyComparison(data)
