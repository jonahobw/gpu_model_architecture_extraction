import sys
from pathlib import Path

# setting path
sys.path.append("../edge_profile")

import datasets
from datasets import Dataset
from model_manager import SurrogateModelManager, VictimModelManager
from utils import latest_file


def validateClassBalance():
    a = Dataset("cifar10", data_subset_percent=0.5)
    a.classBalance(a.train_acc_data)

    a.classBalance(a.train_data)

    a.classBalance(a.val_data)

    b = Dataset("cifar10", data_subset_percent=0.5, idx=1)
    b.classBalance(a.train_acc_data)

    b.classBalance(a.train_data)

    b.classBalance(a.val_data)


def checkAccuracy(gpu: int = -1):
    arch = "mnasnet1_3"

    paths = VictimModelManager.getModelPaths()
    vict_path = [x for x in paths if str(x).find(arch) >= 0][0]
    print(f"Using victim {vict_path}")
    surrogate_path = latest_file(vict_path.parent, pattern="surrogate*")
    surrogate_path = surrogate_path / "checkpoint.pt"

    surrogate_manager = SurrogateModelManager.load(surrogate_path, gpu=gpu)

    print(f"Victim {arch} on its own train set:")
    surrogate_manager.victim_model.topKAcc(
        surrogate_manager.victim_model.dataset.train_acc_dl
    )

    print(f"Surrogate {arch} on victim's train set:")
    surrogate_manager.topKAcc(surrogate_manager.victim_model.dataset.train_acc_dl)

    print(f"Victim {arch} on its own test set:")
    surrogate_manager.victim_model.topKAcc(
        surrogate_manager.victim_model.dataset.val_dl
    )

    print(f"Surrogate {arch} on victim's test set:")
    surrogate_manager.topKAcc(surrogate_manager.victim_model.dataset.val_dl)

    print(f"Victim {arch} on surrogate's train set:")
    surrogate_manager.victim_model.topKAcc(surrogate_manager.dataset.train_acc_dl)

    print(f"Surrogate {arch} on its own train set:")
    surrogate_manager.topKAcc(surrogate_manager.dataset.train_acc_dl)

    print(f"Victim {arch} on surrogate's test set:")
    surrogate_manager.victim_model.topKAcc(surrogate_manager.dataset.val_dl)

    print(f"Surrogate {arch} on its own test set:")
    surrogate_manager.topKAcc(surrogate_manager.dataset.val_dl)


def check_dataset_split():
    a = Dataset("cifar10", data_subset_percent=0.1)
    print(len(a.train_data))
    print(len(a.val_data))
    print(len(a.train_acc_data))

    assert len(a.train_acc_data) == len(a.train_data)
    assert a.train_acc_data.indices == a.train_data.indices

    b = Dataset("cifar10", data_subset_percent=0.1, idx=1)
    print(len(b.train_data))
    print(len(b.val_data))
    print(len(b.train_acc_data))

    assert len(b.train_acc_data) == len(b.train_data)
    assert b.train_acc_data.indices == b.train_data.indices

    c = Dataset("cifar10")
    print(len(c.train_data))
    print(len(c.val_data))
    print(len(c.train_acc_data))

    assert len(c.train_data) == len(a.train_data) + len(b.train_data)
    assert len(c.val_data) == len(a.val_data) + len(b.val_data)
    assert len(c.train_acc_data) == len(a.train_acc_data) + len(b.train_acc_data)

    assert not set(b.train_data.indices).intersection(set(a.train_data.indices))
    assert not set(b.val_data.indices).intersection(set(a.val_data.indices))

    # check for repetition
    d = Dataset("cifar10", data_subset_percent=0.1)
    assert d.train_data.indices == a.train_data.indices
    assert d.val_data.indices == a.val_data.indices

    e = Dataset("cifar10", data_subset_percent=0.1, idx=1)
    assert e.train_data.indices == b.train_data.indices
    assert e.val_data.indices == b.val_data.indices

    print("All checks valid")


if __name__ == "__main__":
    exit(0)
