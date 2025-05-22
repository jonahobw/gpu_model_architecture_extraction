"""
Generates an input to feed to a model.  Seed can be specified for random inputs.
Configuration of input size comes from config.py
"""
import config
import torch

VALID_INPUTS = {"random": torch.randn, "0": torch.zeros, "1": torch.ones}


def construct_input(type: str, number: int, seed=None) -> torch.Tensor:
    if type not in VALID_INPUTS:
        raise ValueError(
            f"Provided input argument {type} but valid options are {list(VALID_INPUTS.keys())}."
        )

    if seed:
        torch.manual_seed(seed)

    return VALID_INPUTS[type](
        number, config.CHANNELS, config.INPUT_SIZE, config.INPUT_SIZE
    )
