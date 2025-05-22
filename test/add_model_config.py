"""
Use this module when you have updated the code in
model_manager.py such that it is not backwards compatible
with previous config formats.
"""

import json
import sys
from pathlib import Path

# setting path
sys.path.append("../edge_profile")

from model_manager import (SurrogateModelManager, VictimModelManager,
                           getVictimSurrogateModels)


def addConfig(args: dict):
    """For each model, add <args> to its configuration"""
    for path in VictimModelManager.getModelPaths():
        manager = VictimModelManager.load(path)
        manager.saveConfig(args)


def addProfileConfig(args: dict, filters: dict = {}):
    """For each model, and for each profile, add <args> to its configuration"""
    for path in VictimModelManager.getModelPaths():
        manager = VictimModelManager.load(path)
        profiles = manager.getAllProfiles(filters=filters)
        for _, conf in profiles:
            config_file = (
                manager.path / "profiles" / f"params_{conf['profile_number']}.json"
            )
            conf.update(args)
            with open(config_file, "w") as f:
                json.dump(conf, f, indent=4)


def addSurrogateConfig(args: dict, replace: bool = False):
    """
    For each surrogate model, add <args> to its configuration.
    Replace arguments in the config file only if replace = True
    """
    vict_to_surrogate = getVictimSurrogateModels()
    for vict_path in vict_to_surrogate:
        for surrogate_path in vict_to_surrogate[vict_path]:
            SurrogateModelManager.saveConfigFast(
                surrogate_path.parent, args=args, replace=replace
            )


if __name__ == "__main__":
    # addConfig({"pretrained": False})
    # addProfileConfig({"gpu_type": "quadro_rtx_8000"})
    addSurrogateConfig({"knockoff_transfer_set": None})
