import json
import sys
from multiprocessing import freeze_support
from pathlib import Path

# setting path
sys.path.append("../edge_profile")


import torch

from model_manager import (SurrogateModelManager, VictimModelManager,
                           getVictimSurrogateModels)
from model_metrics import correct


def load(path):
    return SurrogateModelManager.load(path)


def newSurrogate(victim_path, pretrained=True, save_model=True):
    architecture = vict_path.parent.parent.name
    return SurrogateModelManager(
        victim_path,
        architecture=architecture,
        arch_conf=1.0,
        arch_pred_model_name="rf",
        save_model=save_model,
        pretrained=pretrained,
    )


def testAgreement(surrogate_manager):
    # dl = surrogate_manager.victim_model.dataset.val_dl
    # (x, y) = next(iter(dl))
    # print(x.shape)
    x = torch.rand((5, 3, 224, 224))

    victim_yhat = surrogate_manager.victim_model.model(x)
    print(victim_yhat)
    print(victim_yhat.shape)
    victim_targets = torch.argmax(victim_yhat, dim=1)
    print(victim_targets)
    print(victim_targets.shape)
    yhat = surrogate_manager.model(x)
    print(yhat)
    print(yhat.shape)
    agreement = correct(yhat, victim_targets)
    print(agreement)


if __name__ == "__main__":
    freeze_support()
    manager_paths = getVictimSurrogateModels()
    for vict_path in manager_paths:
        a = newSurrogate(vict_path, save_model=False, pretrained=True)
        for x, y in a.dataset.train_dl:
            print(a.model(x))
            break
    # surrogate_manager = load()
    # l1_weight_bound = surrogate_manager.getL1WeightNorm(surrogate_manager.victim_model)
    # print(l1_weight_bound)
    # testAgreement(surrogate_manager=surrogate_manager)

    # a.loadKnockoffTransferSet(
    #     dataset_name="tiny-imagenet-200",
    #     transfer_size=1000,
    #     sample_avg=5,
    #     force=True,
    # )
    # a.trainModel(3, debug=2)

    # a = getVictimSurrogateModels()
    # b = {str(x): [str(i) for i in a[x]] for x in a}
    # print(json.dumps(b, indent=4, default=lambda x: str(x)))
