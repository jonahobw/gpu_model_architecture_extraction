import json
import sys
from multiprocessing import freeze_support
from pathlib import Path

# setting path
sys.path.append("../edge_profile")


import torch

from model_manager import (
    SurrogateModelManager,
    VictimModelManager,
    getVictimSurrogateModels,
)
from model_metrics import correct

if __name__ == "__main__":
    freeze_support()

    a = VictimModelManager(
        architecture="alexnet",
        dataset="cifar10",
        model_name="test_alexnet",
        save_model=False,
        pretrained=True,
    )
    b = VictimModelManager(
        architecture="alexnet",
        dataset="cifar100",
        model_name="test_alexnet",
        save_model=False,
        pretrained=False,
    )
    c = VictimModelManager(
        architecture="resnet18",
        dataset="cifar10",
        model_name="test_alexnet",
        save_model=False,
    )

    # for x, y in a.dataset.train_dl:
    #     print(a.model(x))
    #     print(b.model(x))
    #     print(c.model(x))
    #     break

    print(a.getL1WeightNorm(b))
    print(b.getL1WeightNorm(a))

    print(a.getL1WeightNorm(a))

    try:
        a.getL1WeightNorm(c)
    except:
        print("success")
