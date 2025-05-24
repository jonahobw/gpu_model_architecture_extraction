"""
White box profiling of PyTorch models using pyprof.

This module provides functionality to profile PyTorch models in a white box setting,
allowing detailed analysis of model performance and GPU utilization. It supports
various model architectures from torchvision and can run on both CPU and GPU.

Example Usage:
    ```python
    # Profile ResNet18 on GPU 0 for 10 inferences
    python whitebox_pyprof.py -model resnet -n 10 -gpu 0

    # Profile MobileNetV3 on CPU for 5 inferences
    python whitebox_pyprof.py -model mobilenetv3 -n 5 -gpu -1
    ```
"""

import argparse
from typing import Optional, Union

import pyprof
import torch
import torchvision.models as models

# Initialize pyprof for profiling
pyprof.init()


def get_model(model_arch: str) -> torch.nn.Module:
    """
    Load a pre-trained model from torchvision based on architecture name.

    Args:
        model_arch (str): Name of the model architecture to load.
            Supported architectures: "resnet", "googlenet", "mobilenetv3", "vgg"

    Returns:
        torch.nn.Module: The loaded model

    Raises:
        ValueError: If the specified model architecture is not supported

    Example:
        ```python
        model = get_model("resnet")  # Returns ResNet18
        model = get_model("mobilenetv3")  # Returns MobileNetV3 Small
        ```
    """
    model_arch = model_arch.lower()
    if model_arch == "resnet":
        return models.resnet18()
    if model_arch == "googlenet":
        return models.googlenet()
    if model_arch == "mobilenetv3":
        return models.mobilenet_v3_small()
    if model_arch == "vgg":
        return models.vgg11_bn()
    raise ValueError("Model not supported")


def main() -> None:
    """
    Main function to run model profiling.

    This function:
    1. Parses command line arguments
    2. Sets up the device (CPU/GPU)
    3. Loads the specified model
    4. Runs the specified number of inferences
    5. Profiles the model's performance

    Command Line Arguments:
        -model: Model architecture to profile (default: "resnet")
        -n: Number of inferences to run (default: 10)
        -gpu: GPU device to use (-1 for CPU, else GPU number) (default: -1)
    """
    parser = argparse.ArgumentParser(
        description="Profile PyTorch models using pyprof in white box setting"
    )
    parser.add_argument(
        "-model",
        type=str,
        default="resnet",
        required=False,
        help="Model architecture to profile (resnet, googlenet, mobilenetv3, vgg)"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        required=False,
        help="Number of inferences to run"
    )
    parser.add_argument(
        "-gpu",
        type=int,
        default=-1,
        required=False,
        help="GPU device to use (-1 for CPU, else GPU number)"
    )

    args = parser.parse_args()

    # Set up device
    device = torch.device("cpu")
    dev_name = "cpu"
    if args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        dev_name = f"gpu{args.gpu}"

    # Load and prepare model
    model = get_model(args.model)
    model.eval()
    model.to(device)

    print(f"Running {args.n} inferences on {args.model} on {dev_name}...")

    # Generate random input data
    inputs = torch.randn(args.n, 3, 224, 224)
    inputs = inputs.to(device)

    # Run inference
    model(inputs)

    print("Completed.")


if __name__ == "__main__":
    main()
