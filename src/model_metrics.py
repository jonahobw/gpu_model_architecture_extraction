"""
Metrics for evaluating model performance and accuracy.

This module provides functions for computing various metrics to evaluate model performance,
including top-k accuracy, agreement between models, and loss calculations.

Adapted from https://github.com/jeffrey-xiao/model-extraction/blob/main/model_metrics.py

Example Usage:
    ```python
    # Compute top-1 and top-5 accuracy
    accuracies = accuracy(model, dataloader, topk=(1, 5))
    top1_acc, top5_acc = accuracies

    # Check agreement between two models
    agreement = both_correct(model1_output, model2_output, targets, topk=(1,))
    ```
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm


def correct(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)
) -> List[int]:
    """
    Compute the number of correct predictions for each top-k value.

    This function calculates how many predictions are correct for each specified
    top-k value. A prediction is considered correct if the target label is among
    the top-k highest values in the output.

    Args:
        output (torch.Tensor): Model output predictions of shape (batch_size, num_classes)
        target (torch.Tensor): Target labels of shape (batch_size,)
        topk (Tuple[int, ...]): Tuple of k values to consider for correctness.
            Defaults to (1,).

    Returns:
        List[int]: Number of correct predictions for each top-k value.
            The length of the list matches the length of topk.

    Example:
        ```python
        output = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])  # 2 samples, 3 classes
        target = torch.tensor([1, 0])  # True labels
        correct_preds = correct(output, target, topk=(1, 2))
        # Returns [1, 2] meaning:
        # - 1 correct prediction in top-1
        # - 2 correct predictions in top-2
        ```
    """
    with torch.no_grad():
        maxk = max(topk)
        # Only need to do topk for highest k, reuse for the rest
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res


def both_correct(
    output1: torch.Tensor,
    output2: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> List[int]:
    """
    Compute the number of predictions that both models get correct.

    This function calculates how many predictions are correct for both models
    simultaneously for each specified top-k value. A prediction is considered
    correct if both models have the target label among their top-k highest values.

    Args:
        output1 (torch.Tensor): First model's output predictions of shape (batch_size, num_classes)
        output2 (torch.Tensor): Second model's output predictions of shape (batch_size, num_classes)
        target (torch.Tensor): Target labels of shape (batch_size,)
        topk (Tuple[int, ...]): Tuple of k values to consider for correctness.
            Defaults to (1,).

    Returns:
        List[int]: Number of predictions that both models get correct for each top-k value.
            The length of the list matches the length of topk.

    Example:
        ```python
        output1 = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
        output2 = torch.tensor([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1]])
        target = torch.tensor([1, 0])
        agreement = both_correct(output1, output2, target, topk=(1, 2))
        # Returns [1, 2] meaning:
        # - 1 prediction where both models are correct in top-1
        # - 2 predictions where both models are correct in top-2
        ```
    """
    with torch.no_grad():
        maxk = max(topk)
        # Only need to do topk for highest k, reuse for the rest
        _, pred1 = output1.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred1 = pred1.t()
        correct1 = pred1.eq(target.view(1, -1).expand_as(pred1)).float()

        _, pred2 = output2.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred2 = pred2.t()
        correct2 = pred2.eq(target.view(1, -1).expand_as(pred2)).float()

        res = []
        for k in topk:
            correct_k1 = torch.sum(correct1[:k], dim=0)
            correct_k2 = torch.sum(correct2[:k], dim=0)
            correct_k = correct_k1 * correct_k2
            correct_k = correct_k.sum(0, keepdim=True)
            res.append(correct_k.item())
        return res


def accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    topk: Tuple[int, ...] = (1,),
    seed: Optional[int] = None,
    loss_func: Optional[_Loss] = None,
    debug: Optional[int] = None,
) -> np.ndarray:
    """
    Compute model accuracy and loss over a dataset.

    This function evaluates a model's performance by computing accuracy metrics
    for specified top-k values and optionally computing loss. It supports
    setting random seeds for reproducibility and can be run in debug mode.

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader containing the evaluation dataset
        topk (Tuple[int, ...]): Tuple of k values to compute accuracy for.
            Defaults to (1,).
        seed (Optional[int]): Random seed for reproducibility. If provided, sets
            seeds for Python, PyTorch, and NumPy. Defaults to None.
        loss_func (Optional[_Loss]): Loss function to compute. If provided, loss
            will be appended to the returned accuracies. Defaults to None.
        debug (Optional[int]): If provided, only process this many batches.
            Defaults to None.

    Returns:
        np.ndarray: Array containing accuracies for each top-k value. If loss_func
            is provided, the loss value is appended to the end of the array.

    Example:
        ```python
        # Compute top-1 and top-5 accuracy with cross entropy loss
        metrics = accuracy(
            model=model,
            dataloader=val_loader,
            topk=(1, 5),
            loss_func=torch.nn.CrossEntropyLoss()
        )
        top1_acc, top5_acc, loss = metrics
        ```
    """
    if seed is not None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        random.seed(seed)

        # Numpy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)

    try:
        # Use same device as model
        device = next(model.parameters()).device
    except StopIteration:
        device = "cpu"

    model.eval()

    accs = np.zeros(len(topk))
    total_tested = 0
    loss = 0

    running_accs = {}
    for x in topk:
        running_accs[f"top{x}"] = 0

    epoch_iter = tqdm(dataloader)
    epoch_iter.set_description("Model Accuracy")

    with torch.no_grad():
        for i, (input, target) in enumerate(epoch_iter):
            if debug is not None and i > debug:
                break
            input = input.to(device)
            target = target.to(device)
            output = model(input)

            accs += np.array(correct(output, target, topk))
            if loss_func:
                loss += loss_func(output, target).item()
            total_tested += len(input)
            for i, x in enumerate(topk):
                running_accs[f"top{x}"] = accs[i] / total_tested
            epoch_iter.set_postfix(**running_accs)

    # Normalize over data length
    loss /= total_tested
    accs /= total_tested

    if loss_func:
        accs = np.append(accs, loss)

    return accs
