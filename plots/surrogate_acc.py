"""
Different plots:

the first 3 plots are implemented by the function
plotMetricByModelAndStrategy()

Relative metric between surrogate and victim:
    Plot (surrogate model metric) / (victim model metric). This is a 
    histogram of a single datapoint at the end of training. One color 
    will be used for each method of surrogate training, which may 
    include pure knowledge distribution, or transfer set training using 
    transfer sets with different parameters. The metric must exist in
    both the victim and surrogate's config files.
    For example, plotting with 'val_acc1' shows how much of the victim's
    validation accuracy is recouped through the surrogate model's 
    training process. The victim's validation set
    is the default validation set during surrogate model training regardless
    of if a transfer set is used. 

    usage: in plotMetricByModelAndStrategy() set absolute to false and
    include_victim to true.

Absolute metric comparision between surrogate and victim:
    Plot surrogate model metric and victim model metric side-by-side.
    This is the same plot as above but in absolute rather than relative terms.
    For example, for accuracy, this will plot the surrogate and victim accuracy
    side-by-side.

    usage: in plotMetricByModelAndStrategy() set absolute to True and
    include_victim to true.

Absolute metric for surrogate model:
    The same as the plot above but only plotting surrogate model datapoints,
    not victim model datapoints. Used to show data specific to the surrogate
    model, like agreement and transfer attack accuracy.
    Like above, one color is used for each surrogate training strategy.

    usage: in plotMetricByModelAndStrategy() set absolute to True and
    include_victim to False.

The same 3 as above, but averaged over all the models into 1 datapoint per
metric, with bars for multiple metrics and different colors for different
training strategies.

---Plots during training---
Training metrics:
    This is a time series plot of different training metrics where the x axis
    is the training epoch. The metrics are:
    train_loss, train_acc1, train_acc5, train_agreement
    val_loss, val_acc1, val_acc5, val_agreement
    l1_weight_bound,transfer_attack_success.

    Note that everything except the l1_weight_bound and loss is between 0 and 1, so
    they can all be on the same plot, and l1_weight_bound will need to be
    on a different plot. Also, this plot has several options:
        (1) plot a single metric (e.g. train_acc1) compared across different
         surrogate model training strategies, averaged by model architecture
            Usage: plotOneMetricPerModel()
        (2) plot a single metric for the same surrogate model training 
         strategy, compared across model architecture
            Usage: plotSingleMetricbyModel()
        (3) plot multiple metrics averaged by model architecture for a single
        surrogate model training strategy.
            Usage: plotMultipleTrainingMetrics() where the strategies parameter
            has only one entry.
"""

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# Configure matplotlib settings
rc("font", **{"family": "serif", "serif": ["Times"], "size": 16})
rc("figure", **{"figsize": (7, 5)})

import sys

# Add parent directory to path for imports
sys.path.append("../edge_profile")

from config import MODELS
from get_model import model_families, name_to_family
from model_manager import (
    SurrogateModelManager,
    VictimModelManager,
    getModelsFromSurrogateTrainStrategies,
    getVictimSurrogateModels,
)

# Constants
SAVE_FOLDER = Path(__file__).parent.absolute() / "surrogate_acc"
SAVE_FOLDER.mkdir(exist_ok=True)


def plotMetricByModelAndStrategy(
    strategies: Dict[str, Dict],
    models: List[str],
    metric: str,
    absolute: bool = False,
    include_victim: bool = True,
    save: bool = True,
) -> None:
    """
    Note this currently assumes that there is only 1 victim model per architecture.

    Plot (surrogate model metric) / (victim model metric). This is a
    histogram of a single datapoint at the end of training. One color
    will be used for each method of surrogate training, which may
    include pure knowledge distribution, or transfer set training using
    transfer sets with different parameters. The names of these
    strategies are the keys of the <strategies> dict.

    The metric must exist in both the victim and surrogate's config files.
    For example, plotting with 'val_acc1' shows how much of the victim's
    validation accuracy is recouped through the surrogate model's
    training process. The victim's validation set
    is the default validation set during surrogate model training regardless
    of if a transfer set is used.

    If absolute is True:
    Plots absolute surrogate model metric and victim model metric side-by-side.
    This is the same plot as above but in absolute rather than relative terms.
    For example, for accuracy, this will plot the surrogate and victim accuracy
    side-by-side.

    Args:
        strategies (Dict[str, Dict]): Dictionary mapping strategy names to their parameters
        models (List[str]): List of model architectures to include
        metric (str): Metric to plot (e.g., 'val_acc1', 'train_loss')
        absolute (bool): If True, plot absolute values; if False, plot relative to victim
        include_victim (bool): Whether to include victim model metrics
        save (bool): Whether to save the plot to a file
    """
    if not include_victim:
        assert absolute, """Cannot plot surrogate metrics relative to victim model
        if include_victim is False. Either set include_victim to True or set 
        absolute to True."""

    plt.cla()
    # manager_paths is a dict of {strategy name: {architecture_name: path to surrogate model}}
    manager_paths = getModelsFromSurrogateTrainStrategies(
        strategies=strategies, architectures=models
    )
    strategy_labels = list(strategies.keys())

    # will be a 2d list of [[architecture1_name, metric under strategy 1,
    # metric under strategy 2 ...], [architecture2_name, ...], ...]
    data = []
    for model in models:
        model_data = [model]
        try:
            for strategy in strategies:
                path = manager_paths[strategy][model]
                model_data.append(SurrogateModelManager.loadConfig(path.parent)[metric])
        except Exception as e:
            print(f"Strategy: {strategy}\nPath: {path}")
            raise e

        if include_victim:
            # get metric of victim model, assumes only 1 victim
            victim_val_acc = SurrogateModelManager.loadVictimConfig(path.parent)[metric]
            model_data.append(victim_val_acc)

        data.append(model_data)

    # Sort and prepare data for plotting
    data = sorted(data, key=lambda x: x[0])
    x_labels = [x[0] for x in data]
    x = np.arange(len(x_labels))
    width = 0.8
    bar_width = width / len(strategies)

    if absolute and include_victim:
        # account for the victim model bar
        bar_width = width / (len(strategies) + 1)

    # Create the plot
    fig, ax = plt.subplots()
    for i, strategy_name in enumerate(strategy_labels):
        offset = (-1 * width / 2) + (i * bar_width) + (bar_width / 2)
        strategy_data = [x[i + 1] / x[-1] for x in data]
        if absolute:
            strategy_data = [x[i + 1] for x in data]
        ax.bar(x - offset, strategy_data, bar_width, label=strategy_name)
        print(
            f"Data:\n{strategy_data}\nAvg {sum(strategy_data)/len(strategy_data)}\tMin {min(strategy_data)}\tMax {max(strategy_data)}"
        )

    # if absolute, add the victim model metric for comparison
    if absolute and include_victim:
        offset = (width / 2) - bar_width / 2
        victim_data = [x[-1] for x in data]
        ax.bar(x - offset, victim_data, bar_width, label="victim_model")

    # Customize the plot
    y_label = f"{metric}" if absolute else f"Surrogate {metric}/ Victim {metric}"
    ax.set_ylabel(y_label)
    title = f"Surrogate {metric} Relative to Victim {metric}\nby Training Strategy and DNN Architecture"
    if absolute:
        title = f"{metric} by Training Strategy and DNN Architecture"
    ax.set_title(title)
    ax.set_xticks(x, x_labels)
    ax.legend(loc="lower right")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    # Save or show the plot
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(
            SAVE_FOLDER
            / f"{metric}_{'with' if include_victim else 'no'}_victim_{timestamp}.png",
            dpi=500,
        )
    else:
        plt.show()


def plotMultipleTrainingMetrics(
    strategies: Dict[str, Dict],
    models: List[str],
    metrics: List[str],
    y_lim: Optional[Tuple[float, float]] = None,
    save: bool = True,
    normalize: bool = False,
) -> None:
    """
    Plot multiple training metrics averaged across model architectures, for a single
    surrogate model training strategy.

    The number of training epochs must be the same for each model.

    Args:
        strategies (Dict[str, Dict]): Dictionary mapping strategy names to their parameters
        models (List[str]): List of model architectures to include
        metrics (List[str]): List of metrics to plot
        y_lim (Optional[Tuple[float, float]]): Y-axis limits
        save (bool): Whether to save the plot to a file
        normalize (bool): Whether to normalize the metrics

    The plot shows:
    - Multiple metrics over training epochs
    - Averaged values across model architectures
    - Optional normalization
    - Training progress visualization
    """
    plt.cla()
    assert len(strategies) == 1
    # manager_paths is a dict of {strategy name: {architecture_name: path to surrogate model}}
    manager_paths = getModelsFromSurrogateTrainStrategies(
        strategies=strategies, architectures=models
    )
    strategy_name = list(strategies.keys())[0]
    num_epochs = SurrogateModelManager.loadConfig(
        manager_paths[strategy_name][models[0]].parent
    )["epochs_trained"]

    # will be a dict of {metric: [metric per epoch over training, summed over the architectures]}
    data = {metric: [0] * num_epochs for metric in metrics}

    for model in models:
        path = manager_paths[strategy_name][model]
        model_train_df = SurrogateModelManager.loadTrainLog(path.parent)
        try:
            for metric in metrics:
                model_metric_data = model_train_df[metric].values.tolist()
                if normalize:
                    model_metric_data = [
                        x / max(model_metric_data) for x in model_metric_data
                    ]
                data[metric] = [
                    data[metric][i] + model_metric_data[i] for i in range(num_epochs)
                ]

        except Exception as e:
            print(
                f"Model: {model}\nPath: {path}\nEpochs trained: {num_epochs}\nModel metric data: {model_metric_data}\nLen: {len(model_metric_data)}"
            )
            raise e

    # Create the plot
    x_axis = list(range(1, num_epochs + 1))
    for metric in data:
        avg_metric = [data[metric][i] / len(models) for i in range(num_epochs)]
        plt.plot(x_axis, avg_metric, label=metric)

    # Customize the plot
    arch_str = (
        f"{str(models)}"[1:-1].replace("'", "") if len(models) < 4 else str(len(models))
    )
    title = f"Surrogate Model {metrics[0]} \nfor {strategy_name} Training\nAveraged over {arch_str} Architecture{'s' if len(models) > 1 else ''}"
    if len(metrics) > 1:
        title = f"Surrogate Model Metrics \nfor {strategy_name} Training\nAveraged over {arch_str} Architecture{'s' if len(models) > 1 else ''}"
        plt.legend()
    ylabel = "Metric Value" if len(metrics) > 1 else metrics[0]
    if normalize:
        title = "Normalized " + title
        ylabel = "Normalized " + ylabel
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlabel("Training Epoch")
    plt.xticks()
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.tight_layout()

    # Save or show the plot
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(SAVE_FOLDER / f"{strategy_name}_{timestamp}.png", dpi=500)
    else:
        plt.show()


def plotOneMetricPerModel(
    strategies: Dict[str, Dict],
    models: List[str],
    metric: str,
    y_lim: Optional[Tuple[float, float]] = None,
    save: bool = True,
    normalize: bool = False,
    include_victim: bool = False,
) -> None:
    """
    Plot a single metric across different training strategies, averaged by model, with training epochs on the x axis.

    Args:
        strategies (Dict[str, Dict]): Dictionary mapping strategy names to their parameters
        models (List[str]): List of model architectures to include
        metric (str): Metric to plot
        y_lim (Optional[Tuple[float, float]]): Y-axis limits
        save (bool): Whether to save the plot to a file
        normalize (bool): Whether to normalize the metric
        include_victim (bool): Whether to include victim model metrics

    The plot shows:
    - Single metric over training epochs
    - Comparison across training strategies
    - Optional victim model comparison
    - Averaged values across model architectures
    """
    plt.cla()
    # manager_paths is a dict of {strategy name: {architecture_name: path to surrogate model}}
    manager_paths = getModelsFromSurrogateTrainStrategies(
        strategies=strategies, architectures=models
    )
    strategy_name = list(strategies.keys())[0]
    num_epochs = SurrogateModelManager.loadConfig(
        manager_paths[strategy_name][models[0]].parent
    )["epochs_trained"]

    # will be a dict of {metric: [metric per epoch over training, summed over the architectures]}
    data = {strategy: [0] * num_epochs for strategy in strategies}
    if include_victim:
        data["victim"] = [0] * num_epochs

    for model in models:
        try:
            for strategy in strategies:
                path = manager_paths[strategy][model]
                model_train_df = SurrogateModelManager.loadTrainLog(path.parent)
                model_metric_data = model_train_df[metric].values.tolist()
                if normalize:
                    model_metric_data = [
                        x / max(model_metric_data) for x in model_metric_data
                    ]
                data[strategy] = [
                    data[strategy][i] + model_metric_data[i] for i in range(num_epochs)
                ]
            if include_victim:
                victim_data = VictimModelManager.loadTrainLog(path.parent.parent)[
                    metric
                ].values.tolist()
                if normalize:
                    victim_data = [x / max(victim_data) for x in victim_data]
                data["victim"] = [
                    data["victim"][i] + victim_data[i] for i in range(num_epochs)
                ]

        except Exception as e:
            print(f"Model: {model}\nPath: {path}\nEpochs trained: {num_epochs}")
            raise e

    # Create the plot
    x_axis = list(range(1, num_epochs + 1))
    for strategy in data:
        avg_metric = [data[strategy][i] / len(models) for i in range(num_epochs)]
        plt.plot(x_axis, avg_metric, label=strategy)
        print(f"Avg Metric\n{avg_metric}")

    # Customize the plot
    arch_str = (
        f"{str(models)}"[1:-1].replace("'", "") if len(models) < 4 else str(len(models))
    )
    title = f"Surrogate Model {metric} \nby Training Strategy\nAveraged over {arch_str} Architecture{'s' if len(models) > 1 else ''}"
    plt.legend()
    ylabel = metric
    if normalize:
        title = "Normalized " + title
        ylabel = "Normalized " + ylabel
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlabel("Training Epoch")
    plt.xticks()
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.tight_layout()

    # Save or show the plot
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(SAVE_FOLDER / f"{strategy_name}_{timestamp}.png", dpi=500)
    else:
        plt.show()


def plotSingleMetricByModel(
    strategies: Dict[str, Dict],
    models: List[str],
    metric: str,
    y_lim: Optional[Tuple[float, float]] = None,
    save: bool = True,
    normalize: bool = False,
    label_family: bool = True,
) -> None:
    """
    Plot a single metric for different models under one training strategy, with training epochs on the x axis.

    Args:
        strategies (Dict[str, Dict]): Dictionary mapping strategy names to their parameters
        models (List[str]): List of model architectures to include
        metric (str): Metric to plot
        y_lim (Optional[Tuple[float, float]]): Y-axis limits
        save (bool): Whether to save the plot to a file
        normalize (bool): Whether to normalize the metric
        label_family (bool): Whether to label by model family

    The plot shows:
    - Single metric over training epochs
    - One line per model architecture
    - Optional family-based coloring
    - Training progress visualization
    """
    plt.cla()
    assert len(strategies) == 1
    manager_paths = getModelsFromSurrogateTrainStrategies(
        strategies=strategies, architectures=models
    )
    strategy_name = list(strategies.keys())[0]
    num_epochs = SurrogateModelManager.loadConfig(
        manager_paths[strategy_name][models[0]].parent
    )["epochs_trained"]

    # Collect and process data
    data = {arch: [0] * num_epochs for arch in models}

    for model in models:
        try:
            path = manager_paths[strategy_name][model]
            model_train_df = SurrogateModelManager.loadTrainLog(path.parent)
            model_metric_data = model_train_df[metric].values.tolist()
            if normalize:
                model_metric_data = [
                    x / max(model_metric_data) for x in model_metric_data
                ]
            data[model] = model_metric_data

        except Exception as e:
            print(
                f"Model: {model}\nPath: {path}\nEpochs trained: {num_epochs}\nModel metric data: {model_metric_data}\nLen: {len(model_metric_data)}"
            )
            raise e

    # Create the plot
    x_axis = list(range(1, num_epochs + 1))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    families = [x[0] for x in model_families]
    family_to_color = {families[i]: colors[i] for i in range(len(families))}
    
    for model in models:
        if label_family:
            plt.plot(
                x_axis,
                data[model],
                label=name_to_family[model],
                color=family_to_color[name_to_family[model]],
            )
        else:
            plt.plot(x_axis, data[model], label=model)

    # Customize the plot
    plt.legend()
    if label_family:
        # ensure no repeating in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="DNN Family")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    arch_str = (
        f"{str(models)}"[1:-1].replace("'", "") if len(models) < 4 else str(len(models))
    )
    title = f"Surrogate Model {metric} \nusing {strategy_name} Training\nFor {arch_str} Architecture{'s' if len(models) > 1 else ''}"
    ylabel = metric
    if normalize:
        title = "Normalized " + title
        ylabel = "Normalized " + ylabel
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlabel("Training Epoch")
    plt.xticks()
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.tight_layout()

    # Save or show the plot
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(SAVE_FOLDER / f"{strategy_name}_{timestamp}.png", dpi=500)
    else:
        plt.show()


if __name__ == "__main__":

    # this is a set of args to match with model manager config
    # values.
    strategies = {
        # "other_half_cifar10" : {
        #     "pretrained": False,
        #     "knockoff_transfer_set": None,
        # },
        # "other_half_cifar10_pretrained" : {
        #     "pretrained": True,
        #     "knockoff_transfer_set": None,
        # },
        "knockoff_cifar100_pretrained": {
            "pretrained": True,
            "knockoff_transfer_set": {
                "dataset_name": "cifar100",
                "transfer_size": 40000,
                "sample_avg": 50,
                "random_policy": False,
                "entropy": True,
            },
        },
    }

    # Model selection
    models = MODELS
    exclude = [
        "mnasnet0_5",
        "mnasnet0_75",
        "mnasnet1_3",
        "mnasnet1_0",
    ]
    for ex in exclude:
        models.remove(ex)

    # plotMetricByModelAndStrategy(strategies=strategies, models=models, metric = "val_acc1", absolute=True, include_victim=True, save=False)
    # plotMetricByModelAndStrategy(strategies=strategies, models=models, metric = "val_acc1", absolute=True, include_victim=False, save=False)
    # plotMetricByModelAndStrategy(strategies=strategies, models=models, metric = "val_acc1", absolute=False, include_victim=True, save=False)
    # plotMultipleTrainingMetrics(strategies=strategies, models=models, metrics=["val_acc1", "train_acc1"], save=False)
    # plotMultipleTrainingMetrics(strategies=strategies, models=models, metrics=["val_loss", "train_loss"], y_lim=(0.001, 0.2), save=True)
    # plotMultipleTrainingMetrics(strategies=strategies, models=models, metrics=["l1_weight_bound"], save=True, normalize=True)
    # plotMultipleTrainingMetrics(strategies=strategies, models=models, metrics=["transfer_attack_success"], save=True)
    # plotOneMetricPerModel(strategies=strategies, models=models, metric="val_acc1", save=False, include_victim=True)
    # plotOneMetricPerModel(strategies=strategies, models=models, metric="val_loss", save=False, include_victim=True, y_lim=(0.001, 0.075))
    # plotOneMetricPerModel(strategies=strategies, models=models, metric="l1_weight_bound", save=False, normalize=True)
    
    plotOneMetricPerModel(
        strategies=strategies,
        models=models,
        metric="transfer_attack_success",
        save=False,
    )
    # plotOneMetricPerModel(strategies=strategies, models=models, metric="val_agreement", save=False)
    # plotSingleMetricByModel(strategies=strategies, models=models, metric="l1_weight_bound", save=False, normalize=True)
    # plotSingleMetricByModel(strategies=strategies, models=models, metric="transfer_attack_success", save=False, normalize=False)
