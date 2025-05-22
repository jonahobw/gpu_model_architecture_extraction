import json
import sys
from typing import List

import numpy as np
import torch
from torch.nn.utils import prune

# setting path
sys.path.append("../edge_profile")

from get_model import get_model


def pruneModel(pruned_modules, ratio=0.5) -> None:
    # modifies self.model (through self.pruned_params)
    prune.global_unstructured(
        pruned_modules,
        pruning_method=prune.L1Unstructured,
        amount=ratio,
    )


def paramsToPrune(model) -> List[torch.nn.Module]:
    res = []
    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            res.append((module, "weight"))
        # if hasattr(module, "bias"):
        #     res.append((module, "bias"))
    return res


def modelSparsity(model, pruned_modules) -> None:
    sparsity = {"module_sparsity": {}}
    pruned_mods_reformatted = {}
    for mod, name in pruned_modules:
        if mod not in pruned_mods_reformatted:
            pruned_mods_reformatted[mod] = [name]
        else:
            pruned_mods_reformatted[mod].append(name)

    zero_params_count = 0
    for name, module in model.named_modules():
        if module in pruned_mods_reformatted:
            total_params = sum(p.numel() for p in module.parameters())
            zero_params = np.sum(
                [
                    getattr(module, x).detach().cpu().numpy() == 0.0
                    for x in pruned_mods_reformatted[module]
                ]
            )
            zero_params_count += zero_params
            sparsity["module_sparsity"][name] = (
                100.0 * float(zero_params) / float(total_params)
            )
        else:
            sparsity["module_sparsity"][name] = 0.0

    # get total sparisty
    total_params = sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()
    )
    sparsity["total_parameters"] = int(total_params)
    sparsity["zero_parameters"] = int(zero_params_count)
    # the percentage of zero params
    sparsity["total_sparsity"] = 100 * float(zero_params_count) / float(total_params)
    print(json.dumps(sparsity, indent=4))


if __name__ == "__main__":
    arch = "mobilenet_v2"
    model = get_model(arch)
    prune_modules = paramsToPrune(model)
    pruneModel(pruned_modules=prune_modules)
    modelSparsity(model=model, pruned_modules=prune_modules)
