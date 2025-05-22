import datetime
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List

import config
from architecture_prediction import ArchPredBase, get_arch_pred_model
from format_profiles import parse_one_profile
from get_model import (all_models, getModelParams, name_to_family,
                       quantized_models)
from logger import CSVLogger
from model_manager import (PruneModelManager, QuantizedModelManager,
                           StructuredPruneModelManager, SurrogateModelManager,
                           VictimModelManager)
from utils import dict_to_str, latest_file


def trainOneVictim(
    model_arch, epochs=150, gpu: int = -1, debug: int = None, save_model: bool = True
) -> VictimModelManager:
    a = VictimModelManager(
        architecture=model_arch,
        dataset="cifar10",
        model_name=model_arch,
        gpu=gpu,
        save_model=save_model,
    )
    a.trainModel(num_epochs=epochs, debug=debug)
    return a


def continueVictimTrain(
    vict_path: Path, epochs: int = 1, gpu: int = -1, debug: int = None
):
    manager = VictimModelManager.load(model_path=vict_path, gpu=gpu)
    manager.trainModel(num_epochs=epochs, debug=debug, replace=True)


def trainVictimModels(
    epochs=150,
    gpu: int = -1,
    reverse: bool = False,
    debug: int = None,
    repeat: bool = False,
    models: List[str] = None,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = Path.cwd() / f"train_progress_{timestamp}.txt"
    f = open(file_path, "w")

    if models is None:
        models = all_models
    if reverse:
        models.reverse()
    start = time.time()

    for i, model in enumerate(models):
        iter_start = time.time()
        if debug and i > debug:
            break
        model_arch_folder = Path.cwd() / "models" / model
        if not repeat and model_arch_folder.exists():
            continue
        try:
            manager = trainOneVictim(model, epochs=epochs, gpu=gpu, debug=debug)
            f.write(f"{model} success\n")
            config.EMAIL.email_update(
                start=start,
                iter_start=iter_start,
                iter=i,
                total_iters=len(models),
                subject=f"Victim {model} Finished Training",
                params=manager.config,
            )
        except Exception as e:
            print(e)
            f.write(
                f"\n\n{model} failed, error\n{e}\ntraceback:\n{traceback.format_exc()}\n\n"
            )
            config.EMAIL.email(
                f"Failed While Training {model}", f"{traceback.format_exc()}"
            )
    f.close()


def profileAllVictimModels(
    gpu: int = 0,
    prefix: str = None,
    nvprof_args: dict = {},
    count: int = 1,
    add: bool = False,
):
    """Victim models must be trained already."""
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        vict_manager = VictimModelManager.load(model_path=vict_path, gpu=gpu)
        assert vict_manager.config["epochs_trained"] > 0
        if vict_manager.isProfiled() and not add:
            print(f"{vict_manager.model_name} is already profiled, skipping...")
            continue
        for i in range(count):
            vict_manager.runNVProf(**nvprof_args)


def trainSurrogateModels(
    epochs: List[int],
    patience: List[int],
    lr: List[float],
    predict: bool = True,
    arch_pred_model_type: str = "nn",
    model_paths: List[str] = None,
    gpu=0,
    reverse=False,
    debug=None,
    save_model: bool = True,
    df=None,
    average_profiles: bool = False,
    filters: dict = None,
    pretrained: bool = True,
    run_attack: bool = True,
    data_idx: int = 1,
):
    """Victim models must be trained and profiled already."""
    if model_paths is None:
        model_paths = VictimModelManager.getModelPaths()
    if reverse:
        model_paths.reverse()
    start = time.time()

    for i, victim_path in enumerate(model_paths):
        iter_start = time.time()
        vict_manager = VictimModelManager.load(victim_path)
        vict_name = Path(victim_path).parent.name
        if predict:
            arch, conf, model = vict_manager.predictVictimArch(
                model=get_arch_pred_model(model_type=arch_pred_model_type, df=df),
                average=average_profiles,
                filters=filters,
            )
        else:
            print(
                f"Warning, predict is False, not predicting for model {vict_manager.path}"
            )
            arch = vict_manager.architecture
            arch_pred_model_name = None
            conf = 0.0
        try:
            surrogate_model = SurrogateModelManager(
                victim_model_path=victim_path,
                architecture=arch,
                arch_conf=conf,
                arch_pred_model_name=arch_pred_model_name,
                gpu=gpu,
                save_model=save_model,
                pretrained=pretrained,
                data_idx=data_idx,
            )
            for j in range(len(epochs)):
                default_lr = getModelParams(arch).get("lr", 0.1)
                surrogate_model.trainModel(
                    num_epochs=epochs[j],
                    patience=patience[j],
                    debug=debug,
                    lr=default_lr * lr[j],
                    run_attack=run_attack,
                    replace=True,
                )
            config.EMAIL.email_update(
                start=start,
                iter_start=iter_start,
                iter=i,
                total_iters=len(model_paths),
                subject=f"Surrogate Model for Victim {vict_name} Finished Training",
                params=surrogate_model.config,
            )
        except Exception as e:
            print(e)
            config.EMAIL.email(
                f"Failed Training Surrogate model for victim Model {vict_name}",
                f"{traceback.format_exc()}",
            )


def runTransferSurrogateModels(
    prefix: str = None,
    gpu=0,
    eps=0.031372549,
    step_size=0.0078431,
    iterations=10,
    train_data: bool = True,
    debug: int = None,
):
    """Both surrogate and victim models must be trained already."""
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        surrogate_manager = SurrogateModelManager.load(model_path=vict_path, gpu=gpu)
        surrogate_manager.transferAttackPGD(
            eps=eps,
            step_size=step_size,
            iterations=iterations,
            train_data=train_data,
            debug=debug,
        )


def quantizeVictimModels(save: bool = True, prefix: str = None):
    """Victim models must be trained already"""
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        arch = vict_path.parent.parent.name
        if arch in quantized_models:
            print(f"Quantizing {arch}...")
            QuantizedModelManager(victim_model_path=vict_path, save_model=save)


def profileAllQuantizedModels(
    gpu: int = 0,
    prefix: str = None,
    nvprof_args: dict = {},
    count: int = 1,
    add: bool = False,
):
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        quant_path = (
            vict_path.parent
            / QuantizedModelManager.FOLDER_NAME
            / QuantizedModelManager.MODEL_FILENAME
        )
        if quant_path.exists():
            quant_manager = QuantizedModelManager.load(model_path=quant_path, gpu=gpu)
            if quant_manager.isProfiled() and not add:
                print(f"{quant_manager.model_name} is already profiled, skipping...")
                continue
            for i in range(count):
                quant_manager.runNVProf(**nvprof_args)


def pruneOneVictim(
    vict_path: Path,
    ratio: float = 0.5,
    finetune_epochs: int = 20,
    gpu: int = -1,
    save: bool = True,
    structured: bool = False,
    debug: int = None,
):
    """Victim models must be trained already"""
    constructor = StructuredPruneModelManager if structured else PruneModelManager
    prune_manager = constructor(
        victim_model_path=vict_path,
        ratio=ratio,
        finetune_epochs=finetune_epochs,
        gpu=gpu,
        save_model=save,
        debug=debug,
    )


def pruneVictimModels(
    prefix: str = None,
    ratio: float = 0.5,
    finetune_epochs: int = 20,
    gpu: int = -1,
    save: bool = True,
    structured: bool = False,
    debug: int = None,
):
    """Victim models must be trained already"""
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        pruneOneVictim(
            vict_path=vict_path,
            ratio=ratio,
            finetune_epochs=finetune_epochs,
            gpu=gpu,
            save=save,
            structured=structured,
            debug=debug,
        )


def deletePrunedProfiles(
    prefix: str = None,
    structured: bool = False,
):
    constructor = StructuredPruneModelManager if structured else PruneModelManager
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        prune_path = (
            vict_path.parent / constructor.FOLDER_NAME / constructor.MODEL_FILENAME
        )
        if prune_path.exists():
            prune_manager = constructor.load(model_path=prune_path)
            prune_manager.delete()


def profileAllPrunedModels(
    gpu: int = 0,
    prefix: str = None,
    nvprof_args: dict = {},
    count: int = 1,
    add: bool = False,
    structured: bool = False,
):
    """Pruned models must be trained already."""
    constructor = StructuredPruneModelManager if structured else PruneModelManager
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        prune_path = (
            vict_path.parent / constructor.FOLDER_NAME / constructor.MODEL_FILENAME
        )
        if prune_path.exists():
            prune_manager = constructor.load(model_path=prune_path, gpu=gpu)
            assert prune_manager.config["epochs_trained"] > 0
            if prune_manager.isProfiled() and not add:
                print(f"{prune_manager.model_name} is already profiled, skipping...")
                continue
            for i in range(count):
                prune_manager.runNVProf(**nvprof_args)


def loadProfilesToFolder(
    prefix: str = "models",
    folder_name: str = "victim_profiles",
    replace: bool = False,
    filters: dict = None,
    all: bool = True,
):
    """
    For every victim model, loads all the profiles into cwd/prefix/name/
    which is organized by model folder
    folder_name: results will be stored to cwd/folder_name
    Additionally creates a config json file where the keys are the paths to the profiles
    and the values are dicts of information about the profile such as path to actual profile,
    actual model architecture and architecture family, and model name.
    filters: a dict and each argument in the dict must match
        the argument from the config file associated with a profile.
        to get a profile by name, can specify {"profile_number": "2181935"}
    all: if true, loads all the profiles, else loads one per victim model
    """
    config_name = "config.json"
    all_config = {}

    folder = Path.cwd() / folder_name
    if folder.exists():
        if not replace:
            print(
                f"loadProfilesToFolder: folder already exists and replace is false, returning"
            )
            return
            # raise FileExistsError
        shutil.rmtree(folder)
    folder.mkdir(exist_ok=True, parents=True)

    file_count = 0

    vict_model_paths = VictimModelManager.getModelPaths(prefix=prefix)
    print(f"All model paths: {vict_model_paths}")
    for vict_path in vict_model_paths:
        print(f"Getting profiles for {vict_path.parent.name}...")
        manager = VictimModelManager.load(vict_path)
        profiles = manager.getAllProfiles(filters=filters)
        if not all:
            profiles = [manager.getProfile(filters=filters)]
        for profile_path, config in profiles:
            config["model"] = manager.architecture
            config["model_path"] = str(manager.path)
            config["manager_name"] = manager.model_name
            config["model_family"] = name_to_family[manager.architecture]
            new_name = f"profile_{manager.architecture}_{file_count}.csv"
            new_path = folder / new_name
            shutil.copy(profile_path, new_path)
            file_count += 1
            all_config[str(new_name)] = config
            print(f"\tSaved Profile {profile_path.name} to {new_path}")

    # save config file
    config_path = folder / config_name
    with open(config_path, "w") as f:
        json.dump(all_config, f, indent=4)


def loadPrunedProfilesToFolder(
    prefix: str = "models",
    folder_name: str = "victim_profiles_pruned",
    replace: bool = False,
    filters: dict = None,
    all: bool = True,
    structured: bool = False,
):
    """
    Same as loadProfilesToFolder, but for pruned models
    """
    config_name = "config.json"
    all_config = {}

    folder = Path.cwd() / folder_name
    if folder.exists():
        if not replace:
            print(
                f"loadProfilesToFolder: folder already exists and replace is false, returning"
            )
            return
            # raise FileExistsError
        shutil.rmtree(folder)
    folder.mkdir(exist_ok=True, parents=True)

    file_count = 0
    constructor = StructuredPruneModelManager if structured else PruneModelManager

    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        prune_path = (
            vict_path.parent / constructor.FOLDER_NAME / constructor.MODEL_FILENAME
        )
        if prune_path.exists():
            manager = constructor.load(model_path=prune_path)
            # below commented to allow for models that have been pruned
            # but not fine tuned
            # assert manager.config["epochs_trained"] > 0

            profiles = manager.getAllProfiles(filters=filters)
            if not all:
                profiles = [manager.getProfile(filters=filters)]
            for profile_path, config in profiles:
                config["model"] = manager.architecture
                config["model_path"] = str(manager.path)
                config["manager_name"] = manager.model_name
                config["model_family"] = name_to_family[manager.architecture]
                new_name = f"profile_{manager.architecture}_{file_count}.csv"
                new_path = folder / new_name
                shutil.copy(profile_path, new_path)
                file_count += 1
                all_config[str(new_name)] = config
                print(f"\tSaved Profile {profile_path.name} to {new_path}")

    # save config file
    config_path = folder / config_name
    with open(config_path, "w") as f:
        json.dump(all_config, f, indent=4)


def predictVictimArchs(
    model: ArchPredBase,
    folder: Path,
    name: str = "predictions",
    save: bool = True,
    topk=5,
    verbose: bool = True,
):
    """Iterates through the profiles in <folder> which was generated
    by loadProfilesToFolder(), the architecture of each, and storing
    a report in a json file called <name>
    """
    assert folder.exists()

    predictions = {}

    config_path = latest_file(folder, pattern="*.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    total_tested = 0
    total_correctk = {k: 0 for k in range(1, topk + 1)}
    family_correct = 0
    for profile_name in config:
        profile_path = folder / profile_name
        profile_features = parse_one_profile(
            profile_path, gpu=config[profile_name]["gpu"]
        )
        true_arch = config[profile_name]["model"]
        true_family = config[profile_name]["model_family"]

        preds = model.topKConf(profile_features, k=topk)
        total_tested += 1
        for k in range(1, topk + 1):
            top_k_preds = preds[:k]
            correct = true_arch in [x[0] for x in top_k_preds]
            if correct:
                total_correctk[k] += 1
            if k == 1:
                pred_arch, conf = top_k_preds[0]
                # print(
                #     f"Predicted architecture for victim model {config[profile_name]['manager_name']} is {arch} with {conf * 100}% confidence."
                # )
                predictions[profile_name] = {
                    "pred_arch": pred_arch,
                    "conf": conf,
                    "true_arch": true_arch,
                    "true_family": true_family,
                }
                predictions[profile_name]["correct"] = correct
                predictions[profile_name]["family_correct"] = False
                if name_to_family[pred_arch] == true_family:
                    predictions[profile_name]["family_correct"] = True
                    family_correct += 1

        predictions[profile_name]["topk_labels"] = [x[0] for x in top_k_preds]
        predictions[profile_name]["topk_conf"] = [x[1] for x in top_k_preds]

    predictions["total_tested"] = total_tested
    predictions["total_correctk"] = {k: total_correctk[k] for k in total_correctk}
    predictions["family_correct"] = family_correct
    predictions["accuracy_k"] = {
        k: total_correctk[k] / total_tested for k in total_correctk
    }
    predictions["family_accuracy"] = family_correct / total_tested

    if verbose:
        print(json.dumps(predictions, indent=4))

    if save:
        report_path = folder / f"{name}.json"
        with open(report_path, "w") as f:
            json.dump(predictions, f, indent=4)
    return predictions


def testKnockoffTrain(
    model_arch="resnet18",
    dataset="cifar100",
    gpu=-1,
    debug: int = None,
    num_epochs: int = 20,
    run_attack: bool = True,
):
    vict_path = [
        x for x in VictimModelManager.getModelPaths() if str(x).find(model_arch) >= 0
    ][0]
    columns = [
        "dataset",
        "transfer_size",
        "sample_avg",
        "random_policy",
        "entropy",
        "pretrained",
        "epochs_trained",
        "train_loss",
        "val_loss",
        "val_acc1",
        "val_acc5",
        "train_agreement",
        "val_agreement",
        "l1_weight_bound",
    ]
    log_path = Path.cwd() / f"test_knockoff_train_{dataset}.csv"
    # todo if you add a new column but use append mode then the new column
    # will be dropped, so set append = False
    logger = CSVLogger(log_path.parent, columns, name=log_path.name, append=False)
    pretrained = False  # todo fix number of classes

    for transfer_size in [10000, 50000]:
        for sample_avg in [5, 10, 50]:
            for random_policy in [True, False]:
                for entropy in [True, False]:
                    # for pretrained in [True, False]:
                    try:
                        manager = SurrogateModelManager(
                            vict_path,
                            architecture=model_arch,
                            arch_conf=1.0,
                            arch_pred_model_name="rf",
                            pretrained=True,
                            gpu=gpu,
                        )
                        manager.loadKnockoffTransferSet(
                            dataset_name=dataset,
                            transfer_size=transfer_size,
                            sample_avg=sample_avg,
                            random_policy=random_policy,
                            entropy=entropy,
                            force=True,
                        )
                        manager.trainModel(
                            num_epochs=num_epochs, debug=debug, run_attack=run_attack
                        )

                        metrics = {
                            "dataset": dataset,
                            "transfer_size": transfer_size,
                            "sample_avg": sample_avg,
                            "random_policy": random_policy,
                            "entropy": entropy,
                            "pretrained": pretrained,
                            "epochs_trained": num_epochs,
                            "train_loss": manager.config["train_loss"],
                            "val_loss": manager.config["val_loss"],
                            "val_acc1": manager.config["val_acc1"],
                            "val_acc5": manager.config["val_acc5"],
                            "train_agreement": manager.config["train_agreement"],
                            "val_agreement": manager.config["val_agreement"],
                            "l1_weight_bound": manager.config["l1_weight_bound"],
                        }

                        logger.set(**metrics)
                        logger.update()

                        config.EMAIL.email(
                            subject="Surrogate Model Finished Training",
                            content=f"""Results for {dict_to_str(metrics)}\n\n
                            {dict_to_str(manager.config)}""",
                        )
                    except Exception as e:
                        print(e)
                        args = {
                            "dataset": dataset,
                            "transfer_size": transfer_size,
                            "sample_avg": sample_avg,
                            "random_policy": random_policy,
                            "entropy": entropy,
                            "pretrained": pretrained,
                        }
                        config.EMAIL.email(
                            f"Failed Training Surrogate model for victim Model",
                            f"{traceback.format_exc()}\n\n\nFailed with args\n{dict_to_str(args)}",
                        )
    logger.close()


def trainKnockoffSurrogateModels(
    dataset: str,
    transfer_size: int,
    sample_avg: int,
    random_policy: bool,
    entropy: bool,
    pretrained: bool,
    epochs: List[int],
    patience: List[int],
    lr: List[float],
    run_attack: bool,
    gpu=0,
    reverse=False,
    debug=None,
    save_model: bool = True,
    model_paths: List[Path] = None,
):
    """
    Epochs, patience, and lr are lists of the same length
    specifying training in stages. the lr list is a factor
    times the default lr.  So if default is 0.1 and lr =
    [1, 0.5], then the lr will be 0.1*1 in the first stage
    and 0.1 * 0.5 in the second stage.
    """
    if model_paths is None:
        model_paths = VictimModelManager.getModelPaths()
    if reverse:
        model_paths.reverse()
    start = time.time()

    for i, victim_path in enumerate(model_paths):
        iter_start = time.time()
        architecture = victim_path.parent.parent.name
        try:
            surrogate_model = SurrogateModelManager(
                victim_model_path=victim_path,
                architecture=architecture,
                arch_conf=-1,
                arch_pred_model_name=f"knockoff_{architecture}",
                gpu=gpu,
                save_model=save_model,
                pretrained=pretrained,
            )
            surrogate_model.loadKnockoffTransferSet(
                dataset_name=dataset,
                transfer_size=transfer_size,
                sample_avg=sample_avg,
                random_policy=random_policy,
                entropy=entropy,
                force=True,
            )
            for j in range(len(epochs)):
                default_lr = getModelParams(architecture).get("lr", 0.1)
                surrogate_model.trainModel(
                    num_epochs=epochs[j],
                    patience=patience[j],
                    debug=debug,
                    lr=default_lr * lr[j],
                    run_attack=run_attack,
                    replace=True,
                )
            config.EMAIL.email_update(
                start=start,
                iter_start=iter_start,
                iter=i,
                total_iters=len(model_paths),
                subject=f"Surrogate Model {architecture} Finished Training",
                params=surrogate_model.config,
            )
        except Exception as e:
            print(e)
            config.EMAIL.email(
                f"Failed Training Surrogate model {architecture}",
                f"{traceback.format_exc()}",
            )


if __name__ == "__main__":
    ans = input(
        "You are running the experiments file.  Enter yes to continue, anything else to exit."
    )
    if not ans.lower() == "yes":
        exit(0)

    # testKnockoffTrain(dataset="cifar100", gpu=-1, debug=1)
    # testKnockoffTrain(dataset="cifar100", gpu=0)
    # testKnockoffTrain(dataset="tiny-imagenet-200", gpu=1)
    # model_paths = None
    # model_paths = VictimModelManager.getModelPaths(architectures=[
    #     'alexnet',
    #     'resnext50_32x4d',
    #     'resnext101_32x8d',
    #     'wide_resnet50_2',
    #     'wide_resnet101_2',
    #     'vgg11',
    #     'vgg11_bn',
    #     'vgg13',
    #     'vgg13_bn',
    #     'vgg16',
    #     'vgg16_bn',
    #     'vgg19_bn',
    #     'vgg19',
    #     'squeezenet1_0',
    #     'squeezenet1_1',
    #     'densenet121',
    #     'densenet169',
    #     'densenet201',
    #     'densenet161',
    #     'googlenet',
    #     'mobilenet_v2',
    #     "mobilenet_v3_large",
    #     "mobilenet_v3_small",
    #     'mnasnet0_5',
    #     'mnasnet0_75',
    #     'mnasnet1_0',
    #     'mnasnet1_3',
    #     'shufflenet_v2_x0_5',
    #     'shufflenet_v2_x1_0',
    #     'shufflenet_v2_x1_5',
    #     'shufflenet_v2_x2_0'
    # ])
    # trainKnockoffSurrogateModels(
    #     dataset="cifar100",
    #     transfer_size=40000,
    #     sample_avg=50,
    #     random_policy=False,
    #     entropy=True,
    #     pretrained=True,
    #     epochs=[20, 20, 10],
    #     patience=[7, 7, 3],
    #     lr=[1, 0.1, 0.01],
    #     run_attack=True,
    #     gpu=0,
    #     model_paths=model_paths,
    # )
    # trainSurrogateModels(
    #     predict=False,
    #     pretrained=True,
    #     epochs=[20, 20, 10],
    #     patience=[7, 7, 3],
    #     lr=[1, 0.1, 0.01],
    #     run_attack=True,
    #     gpu=0,
    #     model_paths=model_paths,
    # )
    # trainSurrogateModels(
    #     predict=False,
    #     pretrained=True,
    #     epochs=[20, 20, 10],
    #     patience=[7, 7, 3],
    #     lr=[1, 0.1, 0.01],
    #     run_attack=True,
    #     gpu=0,
    #     data_idx=0, # train on the same data as victim
    # )

    # ---------------------------------------
    # Structured pruning workflow
    # ---------------------------------------

    # Step 1: generate structured pruned models
    # pruneVictimModels(
    #     finetune_epochs=0,
    #     structured=True,
    # )
    deletePrunedProfiles(structured=True)

    # Step 2: Profile predicted models
    # profileAllPrunedModels(
    #     gpu=0,
    #     structured=True,
    # )

    # Step 3: Load Profiles to a folder
    # loadPrunedProfilesToFolder(
    #     folder_name= "victim_profiles_pruned_structured",
    #     structured=True,
    # )

    # Step 4: Predict Architecture from Profiles, this is done through
    # a separate function in plots/ which calls the predictVictimArchs function here

    exit(0)
