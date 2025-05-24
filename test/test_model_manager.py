import sys
from pathlib import Path

# setting path
sys.path.append("../edge_profile")

from model_manager import (
    PruneModelManager,
    QuantizedModelManager,
    SurrogateModelManager,
    VictimModelManager,
    continueVictimTrain,
    trainOneVictim,
)

if __name__ == "__main__":
    arch = "mobilenet_v2"
    # ------------------------------------------------------------------------------------------
    # Test Victim Model
    # ------------------------------------------------------------------------------------------

    # first try training victim and not saving:
    # verify that no folder exists
    #
    # trainOneVictim(model_arch=arch, epochs=1, debug=1, save_model=False)

    # then try training victim and saving
    # verify that folder exists
    #
    # trainOneVictim(model_arch=arch, epochs=1, debug=1)

    # then try loading the victim model and continuing training
    # verify that the logs.csv has the additional epochs and that epochs_trained in config file is updated
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # continueVictimTrain(path, debug=1)

    # ------------------------------------------------------------------------------------------
    # Test Surrogate Model
    # ------------------------------------------------------------------------------------------

    # manually copy profile folder (including profile and params), and modify the path
    # in the params folder to be the actual path to the profile.  Test creating a surrogate
    # model and not saving:
    # verify that no surrogate folder exists
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # surrogate_manager = SurrogateModelManager(victim_model_path=path, save_model=False)
    # surrogate_manager.trainModel(num_epochs=1, debug=1)

    # test creating and saving a surrogate model
    # verify that the folder exists
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # surrogate_manager = SurrogateModelManager(victim_model_path=path, save_model=True)
    # surrogate_manager.trainModel(num_epochs=1, debug=1)

    # test loading a surrogate model and continuing training
    # verify that the logs.csv has the additional epochs and that epochs_trained in config file is updated
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # surrogate_manager = SurrogateModelManager.load(model_path=path)
    # surrogate_manager.trainModel(num_epochs=1, debug=1, replace=True)

    # test transfer attack
    # verify that config file has parameters
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # surrogate_manager = SurrogateModelManager.load(model_path=path)
    # surrogate_manager.transferAttackPGD(eps=8/255, step_size=2/255, iterations=10, debug=1)

    # ------------------------------------------------------------------------------------------
    # Test Quantized Model
    # ------------------------------------------------------------------------------------------

    # test quantizing and not saving
    # verify that there is no folder
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # quant_manager = QuantizedModelManager(victim_model_path=path, save_model=False)

    # test quantizing and saving
    # verify that folder exists
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # quant_manager = QuantizedModelManager(victim_model_path=path, save_model=True)

    # manually copy profile folder (including profile and params) to the quantize folder,
    # and modify the path in the params folder to be the actual path to the profile.
    # test surrogate model with quantized victim, no save.  This also tests the load function of
    # quantization
    # verify that it works and that there is no folder created
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # quant_path = path.parent / QuantizedModelManager.FOLDER_NAME / QuantizedModelManager.MODEL_FILENAME
    # surrogate_manager = SurrogateModelManager(victim_model_path=quant_path, save_model=False)

    # test surrogate model with quantized victim, with save.  This also tests the load function of
    # quantization
    # verify that it works and that folder is created
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # quant_path = path.parent / QuantizedModelManager.FOLDER_NAME / QuantizedModelManager.MODEL_FILENAME
    # surrogate_manager = SurrogateModelManager(victim_model_path=quant_path, save_model=True)
    # surrogate_manager.trainModel(num_epochs=1, debug=1)

    # test surrogate model with quantized victim, with load and transfer attack
    # verify that it works and that config file is correct
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # quant_path = path.parent / QuantizedModelManager.FOLDER_NAME / QuantizedModelManager.MODEL_FILENAME
    # surrogate_manager = SurrogateModelManager.load(model_path=quant_path)
    # surrogate_manager.transferAttackPGD(eps=8/255, step_size=2/255, iterations=10, debug=1)

    # ------------------------------------------------------------------------------------------
    # Test Pruned Model
    # ------------------------------------------------------------------------------------------

    # test pruning and not saving
    # verify that folder does not exist
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # prune_manager = PruneModelManager(victim_model_path=path, save_model=False, finetune_epochs=1, debug=1)

    # test pruning and saving
    # verify folder exists
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # prune_manager = PruneModelManager(victim_model_path=path, save_model=True, finetune_epochs=1, debug=1)

    # test loading a pruned model and continuing to train
    # verify epochs trained
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # prune_path = path.parent / PruneModelManager.FOLDER_NAME / PruneModelManager.MODEL_FILENAME
    # prune_manager = PruneModelManager.load(model_path = prune_path)
    # prune_manager.trainModel(num_epochs=1, debug=1, replace=True)

    # manually copy profile folder (including profile and params) to the prune folder,
    # and modify the path in the params folder to be the actual path to the profile.
    # test surrogate model with pruned victim, with save.  This also tests the load function of
    # prune
    # verify that it works and that folder is created
    #
    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    # prune_path = path.parent / PruneModelManager.FOLDER_NAME / PruneModelManager.MODEL_FILENAME
    # surrogate_manager = SurrogateModelManager(victim_model_path=prune_path, save_model=True)
    # surrogate_manager.trainModel(num_epochs=1, debug=1)

    # test surrogate model with pruned victim, with load and transfer attack
    # verify that it works and that config file is correct
    #
    path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]
    prune_path = (
        path.parent / PruneModelManager.FOLDER_NAME / PruneModelManager.MODEL_FILENAME
    )
    surrogate_manager = SurrogateModelManager.load(model_path=prune_path)
    surrogate_manager.transferAttackPGD(
        eps=8 / 255, step_size=2 / 255, iterations=10, debug=1
    )
