"""Uses pyprof to profile in white box setting."""
import argparse

import torch
import torchvision.models as models

import pyprof

pyprof.init()


def get_model(model_arch: str):
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


parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default="resnet", required=False)
parser.add_argument(
    "-n", type=int, default=10, required=False, help="number of inferences"
)
parser.add_argument(
    "-gpu", type=int, default=-1, required=False, help="-1 for cpu, else number of gpu"
)

args = parser.parse_args()

device = torch.device("cpu")
dev_name = "cpu"
if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
    dev_name = f"gpu{args.gpu}"

model = get_model(args.model)
model.eval()
model.to(device)

print(f"Running {args.n} inferences on {args.model} on {dev_name}...")

inputs = torch.randn(args.n, 3, 224, 224)
inputs = inputs.to(device)

model(inputs)

print("Completed.")
