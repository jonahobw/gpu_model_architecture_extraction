"""Experimental workflows for model training, profiling, and analysis.

This module provides functionality for running various experimental workflows including:
- Training victim and surrogate models
- Model profiling and performance analysis
- Model pruning and quantization
- Architecture prediction and validation
- Knockoff model training and evaluation

The module supports multiple workflows:
1. Victim Model Training:
   - Train individual or multiple victim models
   - Profile model performance
   - Quantize models
   - Prune models (structured and unstructured)

2. Surrogate Model Training:
   - Train surrogate models based on victim models
   - Support for knockoff training
   - Transfer learning capabilities
   - Architecture prediction

3. Model Analysis:
   - Profile collection and organization
   - Architecture prediction validation
   - Performance metrics tracking
   - Model family classification

Dependencies:
    - torch: For model operations
    - torchvision: For model architectures
    - numpy: For numerical operations
    - pandas: For data handling
    - pathlib: For file operations
"""

import datetime
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import config
from architecture_prediction import ArchPredBase, get_arch_pred_model
from format_profiles import parse_one_profile
from get_model import all_models, getModelParams, name_to_family, quantized_models
from logger import CSVLogger
from model_manager import (
    PruneModelManager,
    QuantizedModelManager,
    StructuredPruneModelManager,
    SurrogateModelManager,
    VictimModelManager,
)
from utils import dict_to_str, latest_file


def trainOneVictim(
    model_arch: str,
    epochs: int = 150,
    gpu: int = -1,
    debug: Optional[int] = None,
    save_model: bool = True,
) -> VictimModelManager:
    """Train a single victim model.

    Args:
        model_arch: Architecture name of the model to train
        epochs: Number of training epochs
        gpu: GPU device ID (-1 for CPU)
        debug: Debug level (None for no debugging)
        save_model: Whether to save the trained model

    Returns:
        Trained VictimModelManager instance
    """
    manager = VictimModelManager(
        architecture=model_arch,
        dataset="cifar10",
        model_name=model_arch,
        gpu=gpu,
        save_model=save_model,
    )
    manager.trainModel(num_epochs=epochs, debug=debug)
    return manager


def continueVictimTrain(
    vict_path: Path, epochs: int = 1, gpu: int = -1, debug: Optional[int] = None
) -> None:
    """Continue training a victim model from a saved checkpoint.

    Args:
        vict_path: Path to the saved model
        epochs: Number of additional training epochs
        gpu: GPU device ID (-1 for CPU)
        debug: Debug level (None for no debugging)
    """
    manager = VictimModelManager.load(model_path=vict_path, gpu=gpu)
    manager.trainModel(num_epochs=epochs, debug=debug, replace=True)


def trainVictimModels(
    epochs: int = 150,
    gpu: int = -1,
    reverse: bool = False,
    debug: Optional[int] = None,
    repeat: bool = False,
    models: Optional[List[str]] = None,
) -> None:
    """Train multiple victim models with progress tracking.

    Args:
        epochs: Number of training epochs per model
        gpu: GPU device ID (-1 for CPU)
        reverse: Whether to process models in reverse order
        debug: Debug level (None for no debugging)
        repeat: Whether to retrain existing models
        models: List of model architectures to train (None for all models)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = Path.cwd() / f"train_progress_{timestamp}.txt"

    with open(file_path, "w") as f:
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


def profileAllVictimModels(
    gpu: int = 0,
    prefix: Optional[str] = None,
    nvprof_args: Dict = {},
    count: int = 1,
    add: bool = False,
) -> None:
    """Profile all victim models using NVIDIA profiler.

    Args:
        gpu: GPU device ID
        prefix: Optional prefix to filter model paths
        nvprof_args: Additional arguments for nvprof
        count: Number of profiling runs per model
        add: Whether to add new profiles to existing ones
    """
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
    model_paths: Optional[List[str]] = None,
    gpu: int = 0,
    reverse: bool = False,
    debug: Optional[int] = None,
    save_model: bool = True,
    df: Optional[object] = None,
    average_profiles: bool = False,
    filters: Optional[Dict] = None,
    pretrained: bool = True,
    run_attack: bool = True,
    data_idx: int = 1,
) -> None:
    """Train surrogate models based on victim models.

    Args:
        epochs: List of epochs for each training stage
        patience: List of patience values for each stage
        lr: List of learning rate multipliers for each stage
        predict: Whether to predict architecture before training
        arch_pred_model_type: Type of architecture prediction model
        model_paths: List of victim model paths to use
        gpu: GPU device ID
        reverse: Whether to process models in reverse order
        debug: Debug level (None for no debugging)
        save_model: Whether to save trained models
        df: Optional DataFrame for architecture prediction
        average_profiles: Whether to average multiple profiles
        filters: Optional filters for profile selection
        pretrained: Whether to use pretrained models
        run_attack: Whether to run attack evaluation
        data_idx: Index of dataset split to use
    """
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
    prefix: Optional[str] = None,
    gpu: int = 0,
    eps: float = 0.031372549,
    step_size: float = 0.0078431,
    iterations: int = 10,
    train_data: bool = True,
    debug: Optional[int] = None,
) -> None:
    """Run transfer attacks on surrogate models.

    Args:
        prefix: Optional prefix to filter model paths
        gpu: GPU device ID
        eps: Maximum perturbation size
        step_size: Step size for PGD attack
        iterations: Number of PGD iterations
        train_data: Whether to use training data
        debug: Debug level (None for no debugging)
    """
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        surrogate_manager = SurrogateModelManager.load(model_path=vict_path, gpu=gpu)
        surrogate_manager.transferAttackPGD(
            eps=eps,
            step_size=step_size,
            iterations=iterations,
            train_data=train_data,
            debug=debug,
        )


def quantizeVictimModels(save: bool = True, prefix: Optional[str] = None) -> None:
    """Quantize victim models.

    Args:
        save: Whether to save quantized models
        prefix: Optional prefix to filter model paths
    """
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        arch = vict_path.parent.parent.name
        if arch in quantized_models:
            print(f"Quantizing {arch}...")
            QuantizedModelManager(victim_model_path=vict_path, save_model=save)


def profileAllQuantizedModels(
    gpu: int = 0,
    prefix: Optional[str] = None,
    nvprof_args: Dict = {},
    count: int = 1,
    add: bool = False,
) -> None:
    """Profile all quantized models.

    Args:
        gpu: GPU device ID
        prefix: Optional prefix to filter model paths
        nvprof_args: Additional arguments for nvprof
        count: Number of profiling runs per model
        add: Whether to add new profiles to existing ones
    """
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
    debug: Optional[int] = None,
) -> None:
    """Prune a single victim model.

    Args:
        vict_path: Path to the victim model
        ratio: Pruning ratio
        finetune_epochs: Number of fine-tuning epochs
        gpu: GPU device ID (-1 for CPU)
        save: Whether to save the pruned model
        structured: Whether to use structured pruning
        debug: Debug level (None for no debugging)
    """
    constructor = StructuredPruneModelManager if structured else PruneModelManager
    constructor(
        victim_model_path=vict_path,
        ratio=ratio,
        finetune_epochs=finetune_epochs,
        gpu=gpu,
        save_model=save,
        debug=debug,
    )


def pruneVictimModels(
    prefix: Optional[str] = None,
    ratio: float = 0.5,
    finetune_epochs: int = 20,
    gpu: int = -1,
    save: bool = True,
    structured: bool = False,
    debug: Optional[int] = None,
) -> None:
    """Prune multiple victim models.

    Args:
        prefix: Optional prefix to filter model paths
        ratio: Pruning ratio
        finetune_epochs: Number of fine-tuning epochs
        gpu: GPU device ID (-1 for CPU)
        save: Whether to save pruned models
        structured: Whether to use structured pruning
        debug: Debug level (None for no debugging)
    """
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
    prefix: Optional[str] = None,
    structured: bool = False,
) -> None:
    """Delete profiles of pruned models.

    Args:
        prefix: Optional prefix to filter model paths
        structured: Whether to handle structured pruning profiles
    """
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
    prefix: Optional[str] = None,
    nvprof_args: Dict = {},
    count: int = 1,
    add: bool = False,
    structured: bool = False,
) -> None:
    """Profile all pruned models.

    Args:
        gpu: GPU device ID
        prefix: Optional prefix to filter model paths
        nvprof_args: Additional arguments for nvprof
        count: Number of profiling runs per model
        add: Whether to add new profiles to existing ones
        structured: Whether to handle structured pruning profiles
    """
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
    filters: Optional[Dict] = None,
    all: bool = True,
) -> None:
    """Load model profiles into a centralized folder organized by model.

    Additionally creates a config json file where the keys are the paths to the profiles
    and the values are dicts of information about the profile such as path to actual profile,
    actual model architecture and architecture family, and model name.

    Args:
        prefix: Prefix for model paths
        folder_name: Name of the folder to store profiles
        replace: Whether to replace existing profiles
        filters: Optional filters for profile selection, must match the argument from the
            config file associated with a profile.  To get a profile by name, can specify
            {"profile_number": "2181935"}.
        all: Whether to load all profiles or one per model
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

    # Save config file
    config_path = folder / config_name
    with open(config_path, "w") as f:
        json.dump(all_config, f, indent=4)


def loadPrunedProfilesToFolder(
    prefix: str = "models",
    folder_name: str = "victim_profiles_pruned",
    replace: bool = False,
    filters: Optional[Dict] = None,
    all: bool = True,
    structured: bool = False,
) -> None:
    """Same as loadProfilesToFolder, but for pruned models

    Args:
        prefix: Prefix for model paths
        folder_name: Name of the folder to store profiles
        replace: Whether to replace existing profiles
        filters: Optional filters for profile selection
        all: Whether to load all profiles or one per model
        structured: Whether to handle structured pruning profiles
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

    # Save config file
    config_path = folder / config_name
    with open(config_path, "w") as f:
        json.dump(all_config, f, indent=4)


def predictVictimArchs(
    model: ArchPredBase,
    folder: Path,
    name: str = "predictions",
    save: bool = True,
    topk: int = 5,
    verbose: bool = True,
) -> Dict:
    """Predict architectures from model profiles.

    Iterates through the profiles in <folder> which was generated
    by loadProfilesToFolder(), predicting the architecture of each, and storing
    a report in a json file called <name>.

    Args:
        model: Architecture prediction model
        folder: Folder containing profiles
        name: Name for the prediction results file
        save: Whether to save predictions
        topk: Number of top predictions to consider
        verbose: Whether to print results

    Returns:
        Dictionary containing prediction results and accuracy metrics
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
    model_arch: str = "resnet18",
    dataset: str = "cifar100",
    gpu: int = -1,
    debug: Optional[int] = None,
    num_epochs: int = 20,
    run_attack: bool = True,
) -> None:
    """Test knockoff nets training with different configurations.

    Args:
        model_arch: Architecture of the model to test
        dataset: Dataset to use for training
        gpu: GPU device ID (-1 for CPU)
        debug: Debug level (None for no debugging)
        num_epochs: Number of training epochs
        run_attack: Whether to run attack evaluation
    """
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
    gpu: int = 0,
    reverse: bool = False,
    debug: Optional[int] = None,
    save_model: bool = True,
    model_paths: Optional[List[Path]] = None,
) -> None:
    """Train knockoff surrogate models with specified configurations.

    Epochs, patience, and lr are lists of the same length
    specifying training in stages. the lr list is a factor
    times the default lr.  So if default is 0.1 and lr =
    [1, 0.5], then the lr will be 0.1*1 in the first stage
    and 0.1 * 0.5 in the second stage.

    Args:
        dataset: Dataset to use for training
        transfer_size: Size of transfer set
        sample_avg: Number of samples to average
        random_policy: Whether to use random knockoff nets policy
        entropy: Whether to use entropy-based selection
        pretrained: Whether to use pretrained models
        epochs: List of epochs for each training stage
        patience: List of patience values for each stage
        lr: List of learning rate multipliers for each stage
        run_attack: Whether to run attack evaluation
        gpu: GPU device ID
        reverse: Whether to process models in reverse order
        debug: Debug level (None for no debugging)
        save_model: Whether to save trained models
        model_paths: List of victim model paths to use
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
