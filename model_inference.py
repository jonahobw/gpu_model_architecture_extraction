"""
Runs N inferences on a model A (maybe pretrained) on GPU K with inputs X.
This file is turned into an executable and profiling is enabled while running the executable.
"""
import argparse
import torch

from construct_input import construct_input
from get_model import get_model
from model_manager import VictimModelManager, PruneModelManager, QuantizedModelManager

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default="resnet", required=False)
parser.add_argument(
    "-n", type=int, default=1, required=False, help="number of inferences"
)
parser.add_argument(
    "-gpu", type=int, default=-1, required=False, help="-1 for cpu, else number of gpu"
)
parser.add_argument(
    "-input",
    type=str,
    default="random",
    help="Input type to pass to model. See construct_inputs.py",
)
parser.add_argument(
    "-seed", type=int, default=42, help="Random seed for random inputs."
)
parser.add_argument("-pretrained", action="store_true", help="Use a pretrained model")
parser.add_argument(
    "-load_path",
    default=None,
    required=False,
    help="Provide a path to a model to be used.",
)


args = parser.parse_args()

model = args.model
model_name = args.model
if args.load_path is not None:
    gpu = args.gpu if args.gpu >= 0 else None
    if args.load_path.find(PruneModelManager.FOLDER_NAME) >= 0:
        manager = PruneModelManager.load(model_path=args.load_path, gpu=gpu)
        model = manager.model
        model_name = manager.model_name
    elif args.load_path.find(QuantizedModelManager.FOLDER_NAME) >= 0:
        manager = QuantizedModelManager.load(model_path=args.load_path, gpu=gpu)
        model = manager.model
        model_name = manager.model_name
    else:
        manager = VictimModelManager.load(args.load_path, gpu)
        model = manager.model
        model_name = manager.model_name
else:
    model = get_model(args.model, pretrained=args.pretrained)

device = torch.device("cpu")
dev_name = "cpu"
if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
    dev_name = f"gpu{args.gpu}"

print(f"Running {args.n} inferences on {model_name} on {dev_name}...")

model.eval()
model.to(device)
for i in range(args.n):
    inputs = construct_input(type=args.input, number=1, seed=args.seed)
    inputs = inputs.to(device)
    model(inputs)

print("Completed.")
