"""Generates an input to feed to a model.

This module provides functionality to generate different types of input tensors for model inference.
It supports random, zero, and one-filled tensors, with configurable dimensions from config.py.
The input size and number of channels are taken from the global configuration.

Dependencies:
    - torch: For tensor operations
    - config: For input size and channel configuration

Example Usage:
    ```python
    from construct_input import construct_input
    
    # Generate random input with seed
    random_input = construct_input('random', number=1, seed=42)
    
    # Generate zero-filled input
    zero_input = construct_input('0', number=1)
    ```
"""

from typing import Dict, Callable, Optional
import torch

import config

# Dictionary mapping input types to their corresponding tensor generation functions
VALID_INPUTS: Dict[str, Callable] = {
    "random": torch.randn,  # Random normal distribution
    "0": torch.zeros,  # Zero-filled tensor
    "1": torch.ones,  # One-filled tensor
}


def construct_input(type: str, number: int, seed: Optional[int] = None) -> torch.Tensor:
    """Construct an input tensor for model inference.

    Args:
        type: Type of input to generate ('random', '0', or '1')
        number: Batch size (number of inputs to generate)
        seed: Optional random seed for reproducible random inputs

    Returns:
        Tensor of shape (number, channels, input_size, input_size)

    Raises:
        ValueError: If the input type is not one of the valid options
    """
    if type not in VALID_INPUTS:
        raise ValueError(
            f"Provided input argument {type} but valid options are {list(VALID_INPUTS.keys())}."
        )

    if seed:
        torch.manual_seed(seed)

    return VALID_INPUTS[type](
        number, config.CHANNELS, config.INPUT_SIZE, config.INPUT_SIZE
    )
