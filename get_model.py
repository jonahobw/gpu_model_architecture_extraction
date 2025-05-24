"""
Provides utilities for working with PyTorch vision models and their configurations.

This module provides functionality to:
1. Load and configure various PyTorch vision models
2. Support model quantization
3. Modify model architectures for different classification tasks
4. Manage model families and their variants

Dependencies:
    - torch: Core PyTorch library
    - torchvision.models: Pre-trained vision models

Example Usage:
    ```python
    # Load a pre-trained model
    model = get_model("resnet50", pretrained=True)
    
    # Load a model with custom configuration
    model = get_model("vgg16", pretrained=False, kwargs={"num_classes": 10})
    
    # Get quantized version of a model
    quantized_model = get_quantized_model("mobilenet_v2")
    ```

Note:
    - Models are organized into families (e.g., ResNet, VGG, etc.)
    - Each model family has specific architectural variants
    - Some models support quantization
    - Custom configurations can be applied through kwargs
"""

import torch
import torchvision.models as models
from typing import Dict, List, Optional, Any

# Model family definitions with their variants
alexnet = "alexnet", ["alexnet"]
resnet = "resnet", [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]
vgg = "vgg", [
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]
squeezenet = "squeezenet", ["squeezenet1_0", "squeezenet1_1"]
densenet = "densenet", ["densenet121", "densenet169", "densenet201", "densenet161"]
googlenet = "googlenet", ["googlenet"]
mobilenet = "mobilenet", ["mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"]
mnasnet = "mnasnet", ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]
shufflenet = "shufflenet", [
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]

# List of all model families
model_families = [
    alexnet,
    resnet,
    vgg,
    squeezenet,
    densenet,
    googlenet,
    mobilenet,
    mnasnet,
    shufflenet,
]

# Create mapping from model name to family
name_to_family: Dict[str, str] = {}
for family, family_models in model_families:
    for model in family_models:
        name_to_family[model] = family

# List of all supported models
all_models: List[str] = []
for i in model_families:
    all_models.extend(i[1])

# List of models that support quantization
quantized_models: List[str] = [
    x for x in list(name_to_family.keys()) if hasattr(models.quantization, x)
]


def getModelParams(model_arch: str) -> Dict[str, Any]:
    """
    Returns model-specific parameters and configurations.

    Provides default configurations for various models including:
    - Learning rates
    - Input sizes
    - Optimizer choices
    - Model-specific kwargs

    Args:
        model_arch: Name of the model architecture

    Returns:
        Dict[str, Any]: Dictionary containing model parameters and configurations
    """
    model_params = {
        "googlenet": {"kwargs": {"aux_logits": False}},
        "alexnet": {"input_size": 224, "lr": 0.01},
        "resnext50_32x4d": {"lr": 0.01},
        "resnext101_32x8d": {"lr": 0.01},
        "squeezenet1_0": {"lr": 0.0001, "optim": "adam"},
        "squeezenet1_1": {"lr": 0.0001, "optim": "adam"},
        "vgg11": {"lr": 0.01},
        "vgg13": {"lr": 0.01},
        "vgg16": {"lr": 0.01},
        "vgg19": {"lr": 0.01},
        "vgg11_bn": {"lr": 0.01},
        "vgg13_bn": {"lr": 0.01},
        "vgg16_bn": {"lr": 0.01},
        "vgg19_bn": {"lr": 0.01},
        "mnasnet0_5": {"lr": 0.001, "input_size": 224},
        "mnasnet0_75": {"lr": 0.001, "input_size": 224},
        "mnasnet1_0": {"lr": 0.001, "input_size": 224},
        "mnasnet1_3": {"lr": 0.001, "input_size": 224},
    }
    return model_params.get(model_arch, {})


def get_model(
    model_arch: str,
    pretrained: bool = False,
    kwargs: Dict[str, Any] = None,
) -> torch.nn.Module:
    """
    Loads a PyTorch vision model with specified configuration.

    Args:
        model_arch: Name of the model architecture
        pretrained: Whether to load pre-trained weights
        kwargs: Additional arguments to pass to the model constructor

    Returns:
        torch.nn.Module: The constructed model

    Raises:
        ValueError: If the model architecture is not supported

    Note:
        If pretrained is True, kwargs are not passed to the model constructor
        to ensure compatibility with pre-trained weights.
    """

    model_arch = model_arch.lower()
    if model_arch not in name_to_family:
        raise ValueError(f"Model {model_arch} not supported")
    model_params = getModelParams(model_arch)
    if "kwargs" in model_params and not pretrained:
        kwargs.update(model_params["kwargs"])
    if "num_classes" in kwargs and pretrained:
        num_classes = kwargs["num_classes"]
        kwargs.pop("num_classes")
        model = getattr(models, model_arch)(pretrained=pretrained, **kwargs)
        success = fixLastLayer(model, model_arch, num_classes)
        if not success:
            print(
                "Cannot reset number of classes on pretrained model, will default to 1000."
            )
        return model
    print(f"Passing {kwargs} args to torch to construct {model_arch}")
    return getattr(models, model_arch)(pretrained=pretrained, **kwargs)


def get_quantized_model(
    model_arch: str,
    kwargs: Dict[str, Any] = None,
) -> Optional[torch.nn.Module]:
    """
    Loads a quantized version of a PyTorch vision model.

    Args:
        model_arch: Name of the model architecture
        kwargs: Additional arguments to pass to the model constructor

    Returns:
        Optional[torch.nn.Module]: The quantized model, or None if quantization
            is not supported for the given architecture

    Note:
        Only models that support quantization in torchvision.models.quantization
        can be loaded. Check quantized_models list for supported architectures.
    """
    model_arch = model_arch.lower()
    if hasattr(models.quantization, model_arch):
        model_params = getModelParams(model_arch)
        if "kwargs" in model_params:
            kwargs.update(model_params["kwargs"])
        print(f"Passing {kwargs} args to torch to construct {model_arch}")
        return getattr(models.quantization, model_arch)(**kwargs)

    print(
        f"Warning, model architecture {model_arch} is not supported for quanitzation, returning None from get_quantized_model()."
    )
    return None


def fixLastLayer(
    model: torch.nn.Module,
    architecture: str,
    num_classes: int,
    finetune: bool = False,
) -> bool:
    """
    Modifies the last layer of a model for a different number of classes.

    Supports various model architectures and handles their specific layer structures.
    Can optionally freeze all layers except the last one for fine-tuning.

    Args:
        model: The model to modify
        architecture: Name of the model architecture
        num_classes: Number of output classes
        finetune: If True, freezes all layers except the last one

    Returns:
        bool: True if the modification was successful, False otherwise

    Note:
        This function implements the fine-tuning approach described in:
        https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    print(f"Setting number of classes for {architecture} to {num_classes} ...")
    if finetune:
        for param in model.parameters():
            param.requires_grad = False

    supported_models = all_models
    if architecture not in supported_models:
        return False
    family = name_to_family[architecture]
    if architecture == "alexnet":
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
        return True
    if architecture in ["resnet18", "resnet34"]:
        model.fc = torch.nn.Linear(512, num_classes)
        return True
    if architecture in [
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
    ]:
        model.fc = torch.nn.Linear(2048, num_classes)
        return True
    if family == "vgg":
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
        return True
    if family == "squeezenet":
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
        return True
    if family == "densenet":
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
        return True
    if architecture == "googlenet":
        model.fc = torch.nn.Linear(1024, num_classes)
        return True
    if architecture == "mobilenet_v2":
        model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
        return True
    if architecture in ["mobilenet_v3_large", "mobilenet_v3_small"]:
        in_features = model.classifier[0].out_features
        model.classifier[3] = torch.nn.Linear(in_features, num_classes)
        return True
    if family == "mnasnet":
        model.classifier[1] = torch.nn.Linear(1280, num_classes)
        return True
    if family == "shufflenet":
        model.fc = torch.nn.Linear(model._stage_out_channels[-1], num_classes)
        return True
    return False
