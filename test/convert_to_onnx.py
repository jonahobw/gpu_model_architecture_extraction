import json
from pathlib import Path
from typing import List, Tuple, Dict, Union
import numpy as np
import sys
import copy

# setting path
sys.path.append("../edge_profile")

import torch
import onnx
import onnxruntime
from get_model import get_model, get_quantized_model, all_models


def convert_model(
    model_arch: str,
    save_path: Path,
    input_size: int = 224,
    batched: bool = True,
    quantized: bool = False,
    pretrained=True,
    opset_version: int = 14,
    kwargs={},
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Union[str, int, bool]]]:
    """
    Converts model to onnx and saves in the specified file.

    Returns a tuple of (input provided during tracing,
    output of the pytorch model to this input,
    a dictionary of args)
    """
    args = locals()
    args.pop("model_arch")
    config = {model_arch: args}
    if quantized:
        model = get_quantized_model(model_arch=model_arch, kwargs=kwargs)
        if model is None:
            raise ValueError("Model is not supported for quantization")
    else:
        model = get_model(
            model_arch=model_arch, pretrained=pretrained, kwargs=copy.deepcopy(kwargs)
        )

    # convert to onnx
    model.eval()
    # Input to the model
    input = torch.randn(3, input_size, input_size, requires_grad=False)
    dynamic_axes = {}
    if batched:
        input = torch.unsqueeze(input, dim=0)
        dynamic_axes = {
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        }
    torch_out = model(input)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        input,  # model input (or a tuple for multiple inputs)
        save_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes=dynamic_axes,
    )

    return (input, torch_out, config)


def validate_onnx(save_file: Path, input: torch.Tensor, output: torch.Tensor) -> None:
    """
    Checks that
    (1) ONNX computation graph is valid.
    (2) The output of the ONNX model is the same as the
        PyTorch model for the same input.

    Will raise an error if either check fails
    """

    # first check that the graph is valid
    onnx_model = onnx.load(save_file)
    onnx.checker.check_model(onnx_model)

    # then check that the model's output is the same
    ort_session = onnxruntime.InferenceSession(save_file)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)


def convert_models(
    model_archs: List[str],
    save_folder: Path,
    input_size: int = 224,
    batched: bool = True,
    quantized: bool = False,
    pretrained=True,
    opset_version: int = 14,
    kwargs={},
    remove: bool = True,
) -> None:
    """
    If remove is true, will remove the folder if it exists. If remove is false,
    then the will skip any models which have already been converted.
    """
    global_config = {}
    if save_folder.exists():
        if remove:
            print(f"Folder {str(save_folder)} already exists, removing it.")
            for file in save_folder.glob("*"):
                file.unlink()
            save_folder.rmdir()
        else:
            raise print(f"Folder {str(save_folder)} already exists")

    save_folder.mkdir(parents=True)

    for arch in model_archs:
        save_path = Path(save_folder) / f"{arch}.onnx"
        if save_path.exists():
            if remove:
                print(f"Path {str(save_path)} already exists, removing it.")
                save_path.unlink()
            else:
                print(f"Path {str(save_path)} already exists")
                continue

        input, output, config = convert_model(
            model_arch=arch,
            save_path=str(save_path),
            input_size=input_size,
            batched=batched,
            quantized=quantized,
            pretrained=pretrained,
            opset_version=opset_version,
            kwargs=kwargs,
        )
        validate_onnx(save_file=str(save_path), input=input, output=output)
        global_config.update(config)
        config_path = save_folder / "params.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                previous_config = json.load(f)
            previous_config.update(global_config)
            global_config = previous_config
        with open(config_path, "w") as f:
            json.dump(global_config, f, indent=4)


if __name__ == "__main__":
    save_parent_folder = Path(__file__).parent.parent.absolute() / "onnx_models"

    save_name = "ten_output_classes"

    models = all_models
    # models = ['mobilenet_v2', "mobilenet_v3_large", "mobilenet_v3_small", 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
    #   'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']
    convert_models(
        models,
        save_folder=save_parent_folder / save_name,
        pretrained=False,
        kwargs={"num_classes": 10},
    )
