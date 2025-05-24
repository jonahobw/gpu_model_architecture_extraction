"""
Executes model inference with profiling capabilities.

This script runs multiple inferences on a specified model with configurable inputs
and device placement. It supports various model types including:
- Standard PyTorch models
- Pruned models
- Quantized models
- Victim models (for security analysis)

The script is designed to be run as an executable with profiling enabled,
making it suitable for performance analysis and benchmarking.

Example Usage:
    ```bash
    # Run 10 inferences on a pre-trained ResNet model on GPU 0
    python model_inference.py -model resnet50 -n 10 -gpu 0 -pretrained

    # Run inference on a pruned model with random inputs
    python model_inference.py -model resnet50 -load_path ./pruned_models/resnet50_pruned.pth -input random

    # Run inference on CPU with custom seed
    python model_inference.py -model vgg16 -gpu -1 -seed 123
    ```

Note:
    - Profiling should be enabled when running this script
    - Input types are defined in construct_inputs.py
    - Model loading supports various formats (standard, pruned, quantized)
    - GPU selection can be disabled by setting -gpu to -1
"""

import argparse
from typing import Optional, Union

import torch

from construct_input import construct_input
from get_model import get_model
from model_manager import (
    PruneModelManager,
    QuantizedModelManager,
    VictimModelManager,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for model inference.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run model inference with profiling capabilities"
    )
    parser.add_argument(
        "-model",
        type=str,
        default="resnet",
        required=False,
        help="Model architecture to use",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1,
        required=False,
        help="Number of inferences to run",
    )
    parser.add_argument(
        "-gpu",
        type=int,
        default=-1,
        required=False,
        help="-1 for CPU, else number of GPU to use",
    )
    parser.add_argument(
        "-input",
        type=str,
        default="random",
        help="Input type to pass to model. See construct_inputs.py",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=42,
        help="Random seed for random inputs",
    )
    parser.add_argument(
        "-pretrained",
        action="store_true",
        help="Use a pre-trained model",
    )
    parser.add_argument(
        "-load_path",
        type=str,
        default=None,
        required=False,
        help="Path to a saved model to load",
    )
    return parser.parse_args()


def load_model(
    model_name: str,
    load_path: Optional[str],
    gpu: Optional[int],
    pretrained: bool,
) -> tuple[torch.nn.Module, str]:
    """
    Load a model from either a saved path or create a new one.

    Args:
        model_name: Name of the model architecture
        load_path: Optional path to a saved model
        gpu: GPU device number (None for CPU)
        pretrained: Whether to use pre-trained weights

    Returns:
        tuple[torch.nn.Module, str]: Loaded model and its name
    """
    if load_path is not None:
        if load_path.find(PruneModelManager.FOLDER_NAME) >= 0:
            manager = PruneModelManager.load(model_path=load_path, gpu=gpu)
            return manager.model, manager.model_name
        elif load_path.find(QuantizedModelManager.FOLDER_NAME) >= 0:
            manager = QuantizedModelManager.load(model_path=load_path, gpu=gpu)
            return manager.model, manager.model_name
        else:
            manager = VictimModelManager.load(load_path, gpu)
            return manager.model, manager.model_name
    else:
        model = get_model(model_name, pretrained=pretrained)
        return model, model_name


def main() -> None:
    """Main execution function for model inference."""
    args = parse_args()
    model, model_name = load_model(args.model, args.load_path, args.gpu, args.pretrained)

    # Set up device
    device = torch.device("cpu")
    dev_name = "cpu"
    if args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        dev_name = f"gpu{args.gpu}"

    print(f"Running {args.n} inferences on {model_name} on {dev_name}...")

    # Run inference
    model.eval()
    model.to(device)
    for i in range(args.n):
        inputs = construct_input(type=args.input, number=1, seed=args.seed)
        inputs = inputs.to(device)
        model(inputs)

    print("Completed.")


if __name__ == "__main__":
    main()
