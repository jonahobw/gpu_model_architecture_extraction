"""
Model management and training framework for deep neural networks.

This module provides a comprehensive framework for managing, training, and manipulating
deep neural network models. It includes support for:
- Victim models (original models to be attacked, which may be pruned or quantized)
- Surrogate models (models trained to mimic victim models)

The framework supports various operations including:
- Model training and evaluation
- Model pruning and quantization
- Profiling with nvprof
- Model evasion attacks
- Model architecture prediction

Example Usage:
    ```python
    # Create and train a victim model
    victim = VictimModelManager(
        architecture="resnet50",
        dataset="cifar10",
        model_name="victim1",
        pretrained=True
    )
    victim.trainModel(num_epochs=100)

    # Create a surrogate model based on the victim
    surrogate = SurrogateModelManager(
        victim_model_path=victim.model_path,
        architecture="resnet18",
        arch_conf=0.95,
        arch_pred_model_name="nn"
    )
    surrogate.trainModel(num_epochs=50)

    # Create a pruned version of the victim model
    pruned = PruneModelManager(
        victim_model_path=victim.model_path,
        ratio=0.5,
        finetune_epochs=20
    )
    ```

Dependencies:
    - torch: PyTorch deep learning framework
    - numpy: Numerical computing
    - pandas: Data manipulation
    - cleverhans: Adversarial attack library
    - tqdm: Progress bars
    - pathlib: Path manipulation
    - typing: Type hints
"""

import datetime
import json
import random
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from torch.nn.utils import prune
from tqdm import tqdm

from architecture_prediction import ArchPredBase
from collect_profiles import generateExeName, run_command
from datasets import Dataset
from format_profiles import avgProfiles, parse_one_profile
from get_model import (
    get_model,
    get_quantized_model,
    getModelParams,
    quantized_models,
)
from logger import CSVLogger
from model_metrics import accuracy, both_correct, correct
from online import OnlineStats
from utils import checkDict, latest_file, latestFileFromList


def loadModel(
    path: Path, model: torch.nn.Module, device: Optional[torch.device] = None
) -> None:
    """
    Load model parameters from a saved checkpoint.

    Args:
        path: Path to the model checkpoint file
        model: Model to load parameters into
        device: Device to load the model onto (default: None)

    Raises:
        AssertionError: If the model path does not exist
    """
    assert path.exists(), f"Model load path \n{path}\n does not exist."
    if device is not None:
        params = torch.load(path, map_location=device)
    else:
        params = torch.load(path)
    model.load_state_dict(params, strict=False)
    model.eval()


class ModelManagerBase(ABC):
    """
    Base class for managing deep neural network models.

    This abstract base class provides core functionality for model management,
    including training, saving, loading, and configuration management. It serves
    as the foundation for specialized model managers like VictimModelManager,
    SurrogateModelManager, etc.

    Attributes:
        MODEL_FILENAME (str): Default filename for saved model checkpoints
        architecture (str): Model architecture name
        model_name (str): Name of the model instance
        path (Path): Path to model directory
        dataset (Dataset): Dataset for training/evaluation
        model (torch.nn.Module): The actual model instance
        device (torch.device): Device to run model on
        config (dict): Model configuration dictionary
        epochs_trained (int): Number of epochs trained
        train_metrics (List[str]): List of metrics to track during training
    """

    MODEL_FILENAME = "checkpoint.pt"

    def __init__(
        self,
        architecture: str,
        model_name: str,
        path: Path,
        dataset: str,
        data_subset_percent: Optional[float] = None,
        data_idx: int = 0,
        gpu: int = -1,
        save_model: bool = True,
    ) -> None:
        """
        Initialize the model manager.

        Args:
            architecture: Model architecture name
            model_name: Name for this model instance
            path: Path to store model files
            dataset: Name of dataset to use
            data_subset_percent: Percentage of dataset to use (optional)
            data_idx: Index for dataset subset
            gpu: GPU device number (-1 for CPU)
            save_model: Whether to save model checkpoints

        Note:
            Inheriting classes should not modify self.config until after
            calling this constructor as it will overwrite self.config.
            This constructor leaves no footprint on the filesystem.
        """
        self.architecture = architecture
        self.model_name = model_name
        self.path = path
        self.data_subset_percent = data_subset_percent
        self.data_idx = data_idx
        self.dataset = self.loadDataset(dataset)
        self.save_model = save_model
        self.model = None  # set by self.model=self.constructModel()
        self.model_path = None  # set by self.model=self.constructModel()
        self.device = torch.device("cpu")
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu}")
        self.gpu = gpu

        self.config = {
            "path": str(self.path),
            "architecture": self.architecture,
            "dataset": dataset,
            "dataset_config": self.dataset.config,
            "data_subset_percent": data_subset_percent,
            "data_idx": data_idx,
            "model_name": self.model_name,
            "device": str(self.device),
        }

        self.epochs_trained = 0
        if not self.path.exists():
            self.config["epochs_trained"] = self.epochs_trained

        self.train_metrics = [
            "epoch",
            "timestamp",
            "train_loss",
            "train_acc1",
            "train_acc5",
            "val_loss",
            "val_acc1",
            "val_acc5",
            "lr",
            "attack_success",
        ]

    def constructModel(
        self,
        pretrained: bool = False,
        quantized: bool = False,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        """
        Construct a new model instance.

        Args:
            pretrained: Whether to use pretrained weights
            quantized: Whether to use quantized model
            kwargs: Additional arguments for model construction

        Returns:
            torch.nn.Module: Constructed model instance
        """
        if kwargs is None:
            kwargs = {}
        kwargs.update({"num_classes": self.dataset.num_classes})
        print(f"Model Manager - passing {kwargs} args to construct {self.architecture}")
        if not quantized:
            model = get_model(self.architecture, pretrained=pretrained, kwargs=kwargs)
        else:
            model = get_quantized_model(self.architecture, kwargs=kwargs)
        model.to(self.device)
        return model

    def loadModel(self, path: Path) -> None:
        """
        Load model parameters from a checkpoint.

        Args:
            path: Path to model checkpoint

        Raises:
            AssertionError: If the model path does not exist
        """
        assert Path(path).exists(), f"Model load path \n{path}\n does not exist."
        params = torch.load(path, map_location=self.device)
        self.model.load_state_dict(params, strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.model_path = path
        self.saveConfig({"model_path": str(path)})

    def saveModel(
        self,
        name: Optional[str] = None,
        epoch: Optional[int] = None,
        replace: bool = False,
    ) -> None:
        """
        Save model parameters to a checkpoint file.

        Args:
            name: Name for the checkpoint file (default: MODEL_FILENAME)
            epoch: Epoch number to append to filename
            replace: Whether to replace existing file

        Raises:
            FileExistsError: If file exists and replace is False
        """
        if not self.save_model:
            return
        if name is None:
            name = self.MODEL_FILENAME
        if epoch is not None:
            name = name.split(".")[0] + f"_{epoch}." + name.split(".")[1]
        model_file = self.path / name
        if model_file.exists():
            if replace:
                print(f"Replacing model {model_file}")
                model_file.unlink()
            else:
                raise FileExistsError
        torch.save(self.model.state_dict(), model_file)
        self.model_path = model_file
        self.saveConfig({"model_path": str(model_file)})

    def loadDataset(self, name: str) -> Dataset:
        """
        Load a dataset for training/evaluation.

        Args:
            name: Name of dataset to load

        Returns:
            Dataset: Loaded dataset instance
        """
        return Dataset(
            name,
            data_subset_percent=self.data_subset_percent,
            idx=self.data_idx,
            resize=getModelParams(self.architecture).get("input_size", None),
        )

    @staticmethod
    @abstractmethod
    def load(path: Path, gpu: int = -1) -> "ModelManagerBase":
        """
        Load a model manager from a saved checkpoint.

        Args:
            path: Path to model checkpoint
            gpu: GPU device number (-1 for CPU)

        Returns:
            ModelManagerBase: Loaded model manager instance
        """
        pass

    def saveConfig(self, args: Dict[str, Any] = {}) -> None:
        """
        Save configuration parameters to a JSON file.

        Args:
            args: Additional arguments to add to config
        """
        if not self.save_model:
            return
        self.config.update(args)
        config_files = list(self.path.glob("params_*"))
        if len(config_files) > 1:
            raise ValueError(f"Too many config files in path {self.path}")
        if len(config_files) == 1:
            with open(config_files[0], "r") as f:
                conf = json.load(f)
            conf.update(self.config)
            with open(config_files[0], "w") as f:
                json.dump(conf, f, indent=4)
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.path / f"params_{timestamp}.json"
        with open(path, "w+") as f:
            json.dump(self.config, f, indent=4)

    def delete(self) -> None:
        """
        Delete the model directory and all its contents.
        """
        path = Path(self.path)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            print(f"Deleted {str(path.relative_to(Path.cwd()))}")

    @staticmethod
    def loadConfig(path: Path) -> Dict[str, Any]:
        """
        Load configuration from a model manager directory.

        Args:
            path: Path to model manager directory

        Returns:
            Dict[str, Any]: Configuration dictionary

        Raises:
            ValueError: If no config file found or multiple config files exist
        """
        config_files = list(path.glob("params_*"))
        if len(config_files) != 1:
            raise ValueError(
                f"""There are {len(config_files)} config files in path {path}\n
                There should only be one or you are calling loadConfig() with the
                wrong path. The function should be called with a path to the 
                model manager folder."""
            )
        with open(config_files[0], "r") as f:
            conf = json.load(f)
        return conf

    @staticmethod
    def loadTrainLog(path: Path) -> pd.DataFrame:
        """
        Load training log from CSV file.

        Args:
            path: Path to model directory

        Returns:
            pd.DataFrame: Training log data

        Raises:
            AssertionError: If log file does not exist
        """
        train_logfile = path / "logs.csv"
        assert train_logfile.exists()
        return pd.read_csv(train_logfile)

    @staticmethod
    def saveConfigFast(path: Path, args: Dict[str, Any], replace: bool = False) -> None:
        """
        Quickly save configuration arguments to existing config file.

        Provided as a static method to allow config
        alteration without loading modelmanager object.

        Args:
            path: Path to model directory
            args: Arguments to save
            replace: Whether to replace existing values

        Raises:
            ValueError: If no config file found or multiple config files exist
        """
        config_files = list(path.glob("params_*"))
        if len(config_files) != 1:
            raise ValueError(
                f"There are {len(config_files)} config files in path {path}\nThere should only be one."
            )
        with open(config_files[0], "r") as f:
            conf = json.load(f)

        for arg in args:
            if arg not in conf or replace:
                conf[arg] = args[arg]

        with open(config_files[0], "w+") as f:
            json.dump(conf, f, indent=4)

    def trainModel(
        self,
        num_epochs: int,
        lr: Optional[float] = None,
        debug: Optional[int] = None,
        patience: int = 10,
        replace: bool = False,
        run_attack: bool = False,
    ) -> None:
        """
        Train the model on the dataset self.dataset.

        Args:
            num_epochs: Number of training epochs
            lr: Initial learning rate (default: from model params or 0.1)
            debug: Number of iterations to run in debug mode
            patience: Patience for learning rate reduction
            replace: Whether to replace existing model files
            run_attack: Whether to run attack during training

        Note:
            Learning rate is reduced by factor of 0.1 when loss fails to
            decrease by 1e-4 for patience iterations.
        """
        assert self.dataset is not None
        assert self.model is not None, "Must call constructModel() first"

        if num_epochs == 0:
            self.model.eval()
            print("Training ended, saving model.")
            self.saveModel(replace=replace)
            self.config["epochs_trained"] = self.epochs_trained
            if "initialLR" not in self.config:
                self.config["initialLR"] = lr
            return

        if lr is None:
            lr = getModelParams(self.architecture).get("lr", 0.1)

        self.epochs = num_epochs
        if self.save_model:
            logger = CSVLogger(self.path, self.train_metrics)

        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-4,
        )
        if getModelParams(self.architecture).get("optim", "") == "adam":
            optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        loss_func = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=patience
        )

        since = time.time()
        try:
            for epoch in range(1, num_epochs + 1):
                if debug is not None and epoch > debug:
                    break

                metrics = self.collectEpochMetrics(
                    epoch_num=epoch,
                    optimizer=optim,
                    loss_fn=loss_func,
                    lr_scheduler=lr_scheduler,
                    debug=debug,
                    run_attack=run_attack,
                )

                self.epochs_trained += 1
                if self.save_model:
                    logger.futureWrite(
                        {
                            "timestamp": time.time() - since,
                            "epoch": self.epochs_trained,
                            **metrics,
                        }
                    )

        except KeyboardInterrupt:
            print(f"\nInterrupted at epoch {epoch}. Tearing Down")

        self.model.eval()
        print("Training ended, saving model.")
        self.saveModel(replace=replace)
        if self.save_model:
            logger.flush()
            logger.close()
        self.config["epochs_trained"] = self.epochs_trained
        if "initialLR" not in self.config:
            self.config["initialLR"] = lr
        self.config["finalLR"] = optim.param_groups[0]["lr"]
        self.config.update(metrics)
        self.saveConfig()

    def collectEpochMetrics(
        self,
        epoch_num: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        debug: Optional[int] = None,
        run_attack: bool = False,
    ) -> Dict[str, float]:
        """
        Collect metrics for a training epoch.

        Args:
            epoch_num: Current epoch number
            optimizer: Model optimizer
            loss_fn: Loss function
            lr_scheduler: Learning rate scheduler
            debug: Number of iterations to run in debug mode
            run_attack: Whether to run attack during training

        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        loss, acc1, acc5 = self.runEpoch(
            train=True,
            epoch=epoch_num,
            optim=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            debug=debug,
        )
        val_loss, val_acc1, val_acc5 = self.runEpoch(
            train=False,
            epoch=epoch_num,
            optim=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            debug=debug,
        )

        metrics = {
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "lr": optimizer.param_groups[0]["lr"],
        }
        return metrics

    def runEpoch(
        self,
        train: bool,
        epoch: int,
        optim: torch.optim.Optimizer,
        loss_fn: Callable,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        debug: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Run a single training or validation epoch.

        Args:
            train: Whether this is a training epoch
            epoch: Current epoch number
            optim: Model optimizer
            loss_fn: Loss function
            lr_scheduler: Learning rate scheduler
            debug: Number of iterations to run in debug mode

        Returns:
            Tuple[float, float, float]: (loss, top1 accuracy, top5 accuracy)
        """
        self.model.eval()
        prefix = "val"
        dl = self.dataset.val_dl
        if train:
            self.model.train()
            prefix = "train"
            dl = self.dataset.train_dl

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()
        step_size = OnlineStats()
        step_size.add(optim.param_groups[0]["lr"])

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(
            f"{prefix.capitalize()} Epoch {epoch if train else '1'}/{self.epochs if train else '1'}"
        )

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                if debug and i > debug:
                    break
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = loss_fn(yhat, y)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / len(x))
                acc1.add(c1 / len(x))
                acc5.add(c5 / len(x))

                epoch_iter.set_postfix(
                    loss=total_loss.mean,
                    top1=acc1.mean,
                    top5=acc5.mean,
                    step_size=step_size.mean,
                )

        loss = total_loss.mean
        top1 = acc1.mean
        top5 = acc5.mean

        if train and debug is None:
            lr_scheduler.step(loss)
            # get actual train accuracy/loss after weights update
            top1, top5, loss = accuracy(
                model=self.model,
                dataloader=self.dataset.train_acc_dl,
                loss_func=loss_fn,
                topk=(1, 5),
            )

        return loss, top1, top5

    def runPGD(
        self,
        x: torch.Tensor,
        eps: float,
        step_size: float,
        iterations: int,
        norm: float = np.inf,
    ) -> torch.Tensor:
        """
        Run Projected Gradient Descent attack.

        Args:
            x: Input tensor
            eps: Maximum perturbation
            step_size: Step size for gradient updates
            iterations: Number of iterations
            norm: Norm for projection (default: inf)

        Returns:
            torch.Tensor: Adversarial examples
        """
        return projected_gradient_descent(
            self.model, x, eps=eps, eps_iter=step_size, nb_iter=iterations, norm=norm
        )

    def topKAcc(
        self, dataloader: torch.utils.data.DataLoader, topk: Tuple[int, ...] = (1, 5)
    ) -> None:
        """
        Calculate top-k accuracy on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            topk: Tuple of k values for top-k accuracy
        """
        self.model.eval()
        online_stats = {}
        for k in topk:
            online_stats[k] = OnlineStats()

        data_iter = tqdm(dataloader)
        for x, y in data_iter:
            x = x[:1]
            y = y[:1]
            x, y = x.to(self.device), y.to(self.device)
            yhat = self.model(x)
            topk_correct = correct(yhat, y, topk)
            for k, k_correct in zip(topk, topk_correct):
                online_stats[k].add(k_correct)

            for k in topk:
                data_iter.set_postfix(
                    **{str(k): online_stats[k].mean for k in online_stats}
                )

        print({k: online_stats[k].mean for k in online_stats})

    def getL1WeightNorm(self, other: "ModelManagerBase") -> float:
        """
        Calculate L1 norm of weight differences between two models.

        Args:
            other: Another model manager to compare with

        Returns:
            float: L1 norm of weight differences

        Raises:
            ValueError: If models have different architectures
        """
        if not isinstance(other.model, type(self.model)):
            raise ValueError

        l1_norm = 0.0
        for x, y in zip(
            self.model.state_dict().values(), other.model.state_dict().values()
        ):
            if x.shape == y.shape:
                l1_norm += (x - y).abs().sum().item()
        return l1_norm

    def __repr__(self) -> str:
        """Return string representation of model manager."""
        return str(self.path.relative_to(Path.cwd()))


class ProfiledModelManager(ModelManagerBase):
    """
    Extends ModelManagerBase to include support for profiling neural network models.

    This class adds functionality for profiling model performance using NVIDIA's profiling tools,
    managing profile data, and using profiles for architecture prediction.

    Attributes:
        gpu (int): GPU device number (-1 for CPU)
        model_path (Optional[Path]): Path to the model checkpoint file
    """

    def runNVProf(
        self, use_exe: bool = True, seed: int = 47, n: int = 10, input: str = "0"
    ) -> None:
        """
        Run NVIDIA profiler on the model and save the results.

        Creates a subfolder self.path/profiles, and adds a profile file profile_{pid}.csv and
        associated params_{pid}.json file to this subfolder if the profile succeeds.
        There is support for multiple profiles.

        Args:
            use_exe (bool): Whether to use executable for profiling
            seed (int): Random seed for profiling
            n (int): Number of iterations to profile
            input (str): Input data specification

        Raises:
            AssertionError: If GPU is not available or model path is not set
            RuntimeError: If profiling fails after 5 retries
        """
        assert self.gpu >= 0
        assert self.model_path is not None
        profile_folder = self.path / "profiles"
        profile_folder.mkdir(exist_ok=True)
        prefix = profile_folder / "profile_"
        executable = generateExeName(use_exe)
        print(f"Using executable {executable} for nvprof")
        command = (
            f"nvprof --csv --log-file {prefix}%p.csv --system-profiling on "
            f"--profile-child-processes {executable} -gpu {self.gpu} -load_path {self.model_path}"
            f" -seed {seed} -n {n} -input {input}"
        )

        print(f"\nCommand being run:\n{command}\n\n")

        success, file = run_command(profile_folder, command)
        retries = 0
        print(f"{'Success' if success else 'Failure'} on file {file}")
        while not success:
            print("\nNvprof retrying ... \n")
            time.sleep(10)
            profile_file = latest_file(profile_folder, "profile_")
            if profile_file is not None and profile_file.exists():
                profile_file.unlink()
            success, file = run_command(profile_folder, command)
            retries += 1
            if retries > 5:
                print("Reached 5 retries, exiting...")
                break
        if not success:
            latest_file(profile_folder, "profile_").unlink()
            raise RuntimeError("Nvprof failed 5 times in a row.")
        profile_num = str(file.name).split("_")[1].split(".")[0]
        params = {
            "file": str(file),
            "profile_number": profile_num,
            "use_exe": use_exe,
            "seed": seed,
            "n": n,
            "input": input,
            "success": success,
            "gpu": self.gpu,
            "gpu_type": torch.cuda.get_device_name(0).lower().replace(" ", "_"),
        }
        with open(profile_folder / f"params_{profile_num}.json", "w") as f:
            json.dump(params, f, indent=4)
        assert self.isProfiled()

    def isProfiled(self) -> bool:
        """
        Check if the model has been profiled.

        Returns:
            bool: True if there is a subfolder self.path/profiles with at least one
                  profile_{pid}.csv and associated params_{pid}.json file.
        """
        profile_folder = self.path / "profiles"
        profile_config = list(profile_folder.glob("params_*"))
        if len(profile_config) == 0:
            return False
        with open(profile_config[0], "r") as f:
            conf = json.load(f)
        # instead of taking the path directly from the config file, use the name
        # this allows the model to be profiled on one machine and then downloaded
        # to another machine, and this method will still return true.
        profile_path = self.path / "profiles" / Path(conf["file"]).name
        return profile_path.exists()

    def getProfile(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Get a specific profile based on filter criteria.

        Args:
            filters (Optional[Dict[str, Any]]): Dictionary of filter criteria. Each argument
                must match the argument from the config file associated with a profile.
                To get a profile by name, specify {"profile_number": "2181935"}.  If there
                are multiple profiles which fit the filters, return the latest one.

        Returns:
            Tuple[Path, Dict[str, Any]]: Tuple containing:
                - Path to profile_{pid}.csv
                - Dictionary from params_{pid}.json

        Raises:
            ValueError: If no profiles match the filters
            AssertionError: If no profile configs exist
        """
        if filters is None:
            filters = {}
        profile_folder = self.path / "profiles"
        # get config files
        profile_config = list(profile_folder.glob("params_*"))
        assert len(profile_config) > 0

        fit_filters = {}

        for config_path in profile_config:
            with open(config_path, "r") as f:
                conf = json.load(f)
            matched_filter = True
            for arg in filters:
                if arg not in conf or filters[arg] != conf[arg]:
                    matched_filter = False
                    break
            if matched_filter:
                prof_num = conf["profile_number"]
                profile_path = profile_folder / f"profile_{prof_num}.csv"
                assert profile_path.exists()
                fit_filters[profile_path] = conf

        if len(fit_filters) == 0:
            raise ValueError(
                f"No profiles with filters {filters} found in {profile_folder}"
            )
        latest_valid_path = latestFileFromList(list(fit_filters.keys()))
        conf = fit_filters[latest_valid_path]
        return latest_valid_path, conf

    def getAllProfiles(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        Get all profiles matching filter criteria.

        Args:
            filters (Optional[Dict[str, Any]]): Dictionary of filter criteria. Each argument
                must match the argument from the config file associated with a profile.
                To get profiles by name, specify {"profile_number": "2181935"}.

        Returns:
            List[Tuple[Path, Dict[str, Any]]]: List of tuples containing:
                - Path to profile_{pid}.csv
                - Dictionary from params_{pid}.json
        """
        if filters is None:
            filters = {}
        result = []
        profile_folder = self.path / "profiles"
        profile_configs = list(profile_folder.glob("params_*"))
        for config_file in profile_configs:
            with open(config_file, "r") as f:
                conf = json.load(f)
            matched_filters = True
            for arg in filters:
                if filters[arg] != conf[arg]:
                    matched_filters = False
                    break
            if matched_filters:
                prof_num = conf["profile_number"]
                profile_path = profile_folder / f"profile_{prof_num}.csv"
                if profile_path.exists():
                    result.append((profile_path, conf))
        return result

    def predictVictimArch(
        self,
        arch_pred_model: ArchPredBase,
        average: bool = False,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float, ArchPredBase]:
        """
        Predict the architecture of the victim model using a trained architecture prediction model.

        Args:
            arch_pred_model (ArchPredBase): Trained architecture prediction model
            average (bool): If True, average features from all profiles before prediction
            filters (Optional[Dict[str, Any]]): Filter criteria for selecting profiles

        Returns:
            Tuple[str, float, ArchPredBase]: Tuple containing:
                - Predicted architecture name
                - Confidence score
                - Architecture prediction model used

        Raises:
            AssertionError: If model has not been profiled
        """
        assert self.isProfiled()

        profile_csv, config = self.getProfile(filters=filters)
        profile_features = parse_one_profile(profile_csv, gpu=config["gpu"])
        if average:
            all_profiles = self.getAllProfiles(filters=filters)
            gpu = all_profiles[0][1]["gpu"]
            for _, config in all_profiles:
                assert config["gpu"] == gpu
            profile_features = avgProfiles(
                profile_paths=[x[0] for x in all_profiles], gpu=gpu
            )

        arch, conf = arch_pred_model.predict(profile_features)
        print(
            f"""Predicted surrogate model architecture for victim model\n
            {self.path}\n is {arch} with {conf * 100}% confidence."""
        )
        # format is to store results in self.config as such:
        # {
        #    "pred_arch": {
        #        "nn (model type)": {
        #            "profile_4832947.csv: [
        #                {"pred_arch": "resnet18", "conf": 0.834593048}
        #            ]
        #        }
        #    }
        # }
        #
        # this way there is support for multiple predictions from the
        # same model type on the same profile
        prof_name = str(profile_csv.name)
        arch_conf = {"pred_arch": arch, "conf": conf}
        results = {prof_name: [arch_conf]}
        if "pred_arch" not in self.config:
            self.config["pred_arch"] = {arch_pred_model.name: results}
        else:
            if arch_pred_model.name not in self.config["pred_arch"]:
                self.config["pred_arch"][arch_pred_model.name] = results
            else:
                if prof_name not in self.config["pred_arch"][arch_pred_model.name]:
                    self.config["pred_arch"][arch_pred_model.name][prof_name] = [
                        arch_conf
                    ]
                else:
                    self.config["pred_arch"][arch_pred_model.name][prof_name].append(
                        arch_conf
                    )
        self.saveConfig()
        return arch, conf, arch_pred_model


class VictimModelManager(ProfiledModelManager):
    """
    Manages victim models for model extraction attacks.

    This class handles the creation, training, and management of victim models that will be
    targeted by model extraction attacks. It extends ProfiledModelManager to include
    functionality specific to victim models.

    Attributes:
        pretrained (bool): Whether the model uses pretrained weights
        trained (bool): Whether the model has been trained
    """

    def __init__(
        self,
        architecture: str,
        dataset: str,
        model_name: str,
        load: Optional[str] = None,
        gpu: int = -1,
        data_subset_percent: float = 0.5,
        pretrained: bool = False,
        save_model: bool = True,
    ) -> None:
        """
        Initialize a victim model manager.

        Model files are stored in a folder ./models/{model_architecture}/{self.name}_{date_time}/
        This includes the model file, a csv documenting training, and a config file.

        Args:
            architecture (str): The exact string representation of the model architecture.
                See get_model.py for valid architectures.
            dataset (str): The name of the dataset (all lowercase).
            model_name (str): The name of the model. Do not use underscores.
            load (Optional[str]): If provided, should be the absolute path to the model folder,
                {cwd}/models/{model_architecture}/{self.name}{date_time}. This will load the
                model stored there.
            gpu (int): GPU device number (-1 for CPU).
            data_subset_percent (float): The fraction of the dataset to use. This will be
                generated deterministically using torch.utils.data.random_split.
            pretrained (bool): Whether to use pretrained weights.
            save_model (bool): Whether to save model checkpoints.

        Note:
            The data_idx parameter is set to 0 for victim models.
        """
        path = self.generateFolder(load, architecture, model_name)
        super().__init__(
            architecture=architecture,
            model_name=model_name,
            path=path,
            dataset=dataset,
            data_subset_percent=data_subset_percent,
            gpu=gpu,
            save_model=save_model,
        )

        self.pretrained = pretrained
        self.config["pretrained"] = pretrained

        if load is None:
            # creating the object for the first time
            self.model = self.constructModel(pretrained=pretrained)
            self.trained = False
            if save_model:
                assert not path.exists()
                path.mkdir(parents=True)
        else:
            # load from previous run
            self.model = self.constructModel(pretrained=False)
            self.trained = True
            self.loadModel(load)
            # this causes stored parameters to overwrite new ones
            self.config.update(self.loadConfig(self.path))
            # update with the new device
            self.config["device"] = str(self.device)
            self.epochs_trained = self.config["epochs_trained"]

        self.saveConfig()

    @staticmethod
    def load(model_path: Path, gpu: int = -1) -> "VictimModelManager":
        """
        Create a VictimModelManager from a saved model checkpoint.

        Args:
            model_path (Path): Path to the model checkpoint file
            gpu (int): GPU device number (-1 for CPU)

        Returns:
            VictimModelManager: Loaded victim model manager instance
        """
        folder_path = Path(model_path).parent
        conf = ModelManagerBase.loadConfig(folder_path)
        print(
            f"Loading {conf['architecture']} trained on {conf['dataset']} from path {model_path}"
        )
        model_manager = VictimModelManager(
            architecture=conf["architecture"],
            dataset=conf["dataset"],
            model_name=conf["model_name"],
            load=Path(model_path),
            gpu=gpu,
            data_subset_percent=conf["data_subset_percent"],
            pretrained=conf["pretrained"],
        )
        model_manager.config = conf
        return model_manager

    def generateFolder(
        self, load: Optional[Path], architecture: str, model_name: str
    ) -> Path:
        """
        Generate the model folder path.

        Args:
            load (Optional[Path]): Path to existing model folder if loading
            architecture (str): Model architecture name
            model_name (str): Model instance name

        Returns:
            Path: Path to the model folder
        """
        if load:
            return Path(load).parent
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_folder = Path.cwd() / "models" / architecture / f"{model_name}_{time}"
        return model_folder

    @staticmethod
    def getModelPaths(
        prefix: Optional[str] = None, architectures: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Get paths to all victim models in a directory.

        Args:
            prefix (Optional[str]): Directory prefix to search in. Defaults to "models".
            architectures (Optional[List[str]]): List of architectures to filter by.

        Returns:
            List[Path]: List of paths to victim model checkpoints

        Note:
            The directory must be organized by model architecture folders whose subfolders
            are victim model folders and contain a model stored in 'checkpoint.pt'.
        """
        if prefix is None:
            prefix = "models"

        models_folder = Path.cwd() / prefix
        arch_folders = [i for i in models_folder.glob("*") if i.is_dir()]
        model_paths = []
        for arch in arch_folders:
            if architectures is not None and arch.name not in architectures:
                continue
            for model_folder in [i for i in arch.glob("*") if i.is_dir()]:
                victim_path = (
                    models_folder
                    / arch
                    / model_folder
                    / VictimModelManager.MODEL_FILENAME
                )
                if victim_path.exists():
                    model_paths.append(victim_path)
                else:
                    print(f"Warning, no model found {victim_path}")
        return model_paths

    def getSurrogateModelPaths(self) -> List[Path]:
        """
        Get paths to surrogate models trained on this victim model.

        Returns:
            List[Path]: List of paths to surrogate model checkpoints
        """
        res = []
        surrogate_paths = list(self.path.glob("surrogate*"))
        for path in surrogate_paths:
            res.append(path / ModelManagerBase.MODEL_FILENAME)
        return res

    def generateKnockoffTransferSet(
        self,
        dataset_name: str,
        transfer_size: int,
        sample_avg: int = 10,
        random_policy: bool = False,
        entropy: bool = True,
    ) -> None:
        """
        Uses adaptations of the Knockoff Nets paper https://arxiv.org/pdf/1812.02766.pdf
        to generate a transfer set for this victim model.  A transfer set is a subset of
        a dataset not used to train the victim model. For example, if the victim is
        trained on CIFAR10, the transfer set could be a subset of TinyImageNet. The
        transfer set can then be used to train a surrogate model.

        Two strategies are considered: random (takes a random subset), and adaptive.
        Adaptive is not a faithful implementation of the paper.  Instead, the victim
        model's outputs on samples of each class are averaged over some number of samples,
        then the entropy of this average is taken to get a measure of how influential
        these samples are.  Low entropy means that the samples are influential, and
        vice versa.  There is one entropy value per class. We want to sample more
        from influential classes, so we make a vector of (1 - entropy) for each class,
        normalize it, and then use it as a multinomial distribution from which to
        sample elements for the transfer set.

        When using the adaptive method, a confidence based measure of class influence
        can be used instead of entropy by setting entropy to false.  The confidence is
        metric is the max value of the averaged samples.

        Right now, this function queries the victim model to calculate the entropy of
        each class, but these queries are discarded, and the the victim model predictions
        are generated at training time.  So the actual query budget is
        transfer_size + (num_classes * sample_avg) but if the adversary wanted to run the
        attack efficiently they could save these queries, making the query budget only of
        size transfer_size.

        The resulting dataset will be stored in a json format under
        self.path/transfer_sets/<dataset>_<datetime>.  This file will include the
        parameters passed to this algorithm as well as the indices of the dataset
        which are included in the transfer set.

        Args:
            dataset_name (str): Name of dataset to generate transfer set from (should
            not be the dataset on which the victim model was trained)
            transfer_size (int): Size of transfer set to generate
            sample_avg (int): Number of samples per class for entropy estimation,
            higher number means better entropy estimation.  Only used if random_policy=0
            random_policy (bool): If True, use random sampling strategy
            entropy (bool): If True, use entropy for adaptive sampling, else use confidence

        Raises:
            AssertionError: If dataset is same as victim training dataset
            AssertionError: If transfer size exceeds dataset size
            AssertionError: If sample_avg * num_classes exceeds dataset size
            AssertionError: If sample_avg * num_classes exceeds transfer size
        """
        config = {
            "dataset_name": dataset_name,
            "transfer_size": transfer_size,
            "sample_avg": sample_avg,
            "random_policy": random_policy,
            "entropy": entropy,
        }
        assert (
            dataset_name != self.dataset.name
        ), "Don't use the same dataset for training and transfer set"
        # dataset is a torchvision.datasets.ImageFolder object
        dataset = Dataset(
            dataset_name,
            resize=getModelParams(self.architecture).get("input_size", None),
        ).train_data
        num_classes = len(dataset.classes)

        assert transfer_size <= len(
            dataset
        ), f"""Requested transfer set size of {transfer_size} but {dataset_name} 
        dataset has only {len(dataset)} samples."""

        if not random_policy:
            assert sample_avg * num_classes < len(
                dataset
            ), f"""Requested {sample_avg} samples per class, with {num_classes} classes this 
            is {sample_avg * num_classes} samples but {dataset_name} dataset has only 
            {len(dataset)} samples."""

            assert (
                sample_avg * num_classes <= transfer_size
            ), f"""Requested {sample_avg} samples per class, with {num_classes} classes this is 
            {sample_avg * num_classes} samples, but the transfer size (budget) is only 
            {transfer_size}.  Either decrease the sample_avg or increase the transfer budget"""

        print(
            f"""Generating a transfer dataset for victim model {self} with configuration\n
            {json.dumps(config, indent=4)}\n"""
        )

        sample_indices = None
        if random_policy:
            sample_indices = random.sample(range(len(dataset)), transfer_size)
        else:
            # adaptive policy

            # THE FOLLOWING DOESN'T WORK BECAUSE THE SAMPLES ARE NOT
            # SORTED IN ALL DATASETS, FOR EXAMPLE CIFAR100
            # if the dataset samples are sorted, we need to find
            # the start and end indices of each class and store as
            # [(start index for class 0, end index for class 0),
            # (start index for class 1, end index for class 1), ...]
            # this implementation does not assume a balanced dataset
            # use binary search
            # transitions = []
            # start = 0
            # for class_idx in range(num_classes - 1):
            #     # look for point in dataset.targets that changes from
            #     # class_idx to class_idx + 1, this is the end index
            #     hi = len(dataset) - 1
            #     lo = start
            #     while hi >= lo:
            #         mid = (hi + lo) // 2
            #         if dataset.targets[mid] == class_idx:
            #             lo = mid + 1
            #         else:
            #             if mid > start and dataset.targets[mid - 1] == class_idx:
            #                 # found
            #                 transitions.append((start, mid - 1))
            #                 start = mid
            #                 break
            #             hi = mid - 1
            # # need to account for last class
            # transitions.append((start, len(dataset) - 1))
            #
            # assert len(transitions) == num_classes
            #
            # def sampleClass(n, class_idx) -> List[int]:
            #     # returns n samples from class <class_idx> without replacement
            #     # need to generate n random numbers between the start and end
            #     # indices of this class, inclusive
            #     return random.sample(
            #         range(transitions[class_idx][0], transitions[class_idx][1] + 1), n
            #     )

            # first collect <sample_avg> samples of each class, store as a
            # list [[index of sample 1 of class 1, index of sample 2 of class 1, ...],
            # [index of sample 1 of class 2, index of sample 2 of class 2, ...], ...]
            # the samples of dataset are not necessarily sorted by class
            print("Generating a mapping from indices to labels ...")
            samples = [[] for _ in range(num_classes)]
            for idx in range(len(dataset)):
                samples[dataset.targets[idx]].append(idx)
            # check that there are enough samples per class for the sample average
            for class_name in range(num_classes):
                assert (
                    len(samples[class_name]) >= sample_avg
                ), f"""Could only find {len(samples[class_name])} samples for class {class_name}, 
                but {sample_avg} samples were requested"""

            # get <sample_avg> samples per class
            class_samples = [
                random.sample(samples[class_idx], sample_avg)
                for class_idx in range(num_classes)
            ]
            # now, for each class, get the average of the victim model's output on the samples
            # then compute (1-entropy of average output)
            print(f"Calculating class {'entropies' if entropy else 'confidences'} ...")
            class_influence = []
            for class_idx in tqdm(range(num_classes)):
                sample_indices = class_samples[class_idx]
                x = torch.stack([dataset[i][0] for i in sample_indices]).to(self.device)
                y = self.model(x).cpu()
                y_prob = torch.softmax(
                    y, dim=1
                )  # dimensions [# of samples, # of classes in victim dataset]
                if entropy:
                    # get rid of 0 probability so we can take the log
                    y_prob = torch.where(y_prob == 0.0, 1e-9, y_prob)
                    log_avg = torch.log(y_prob) / torch.log(torch.tensor(num_classes))
                    sample_entropy = -1 * torch.sum(torch.mul(log_avg, y_prob), dim=1)
                    avg_entropy = torch.mean(sample_entropy).item()
                    class_influence.append(1 - avg_entropy)
                else:
                    conf = torch.max(y_prob, dim=1)[0]
                    avg_conf = torch.mean(conf).item()
                    class_influence.append(avg_conf)

            # now, normalize the samples array to a multinomial distribution and use that to
            # sample from the dataset. note that we are going to use the class_samples variable as
            # the starting point, so each class starts with sample_avg samples.
            samples_sum = sum(class_influence)
            multinomial_dist = [x / samples_sum for x in class_influence]
            config["class_importance"] = multinomial_dist

            # check to see if the requested transfer size is all of the data
            if len(dataset) == transfer_size:
                config["samples_per_class"] = [
                    len(dataset) // num_classes for x in range(num_classes)
                ]
                sample_indices = list(range(len(dataset)))

            else:  # need to randomly sample according to class importances
                samples_per_class = np.random.multinomial(
                    transfer_size - (sample_avg * num_classes), multinomial_dist
                ).tolist()
                samples_per_class = [
                    samples_per_class[i] + sample_avg for i in range(num_classes)
                ]
                # check to see if we sampled some classes more than we have data for
                overflow_samples = 0
                unsaturated_classes = [0 for _ in range(num_classes)]
                for class_idx in range(num_classes):
                    if len(samples[class_idx]) < samples_per_class[class_idx]:
                        diff = samples_per_class[class_idx] - len(samples[class_idx])
                        samples_per_class[class_idx] -= diff
                        overflow_samples += diff
                    else:
                        unsaturated_classes[class_idx] += (
                            len(samples[class_idx]) - samples_per_class[class_idx]
                        )
                # sample overflows randomly
                for _ in range(overflow_samples):
                    # normalize unsaturated classes
                    unsaturated_classes_norm = [
                        x / sum(unsaturated_classes) for x in unsaturated_classes
                    ]
                    # choose from unsaturated classes
                    class_idx = np.random.multinomial(
                        1, unsaturated_classes_norm
                    ).argmax()
                    unsaturated_classes[class_idx] -= 1
                    samples_per_class[class_idx] += 1
                config["samples_per_class"] = samples_per_class

                # now generate samples per class and add them to a main list
                print("Sampling classes and writing to file ...")
                sample_indices = []
                for class_idx, num_samples in enumerate(samples_per_class):
                    sample_indices.extend(
                        random.sample(samples[class_idx], num_samples)
                    )

        config["sample_indices"] = sample_indices
        transfer_folder = self.path / "transfer_sets"
        transfer_folder.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = transfer_folder / f"{dataset_name}_{timestamp}.json"
        with open(save_path, "w+") as f:
            json.dump(config, f, indent=4)
        print("Completed generating the transfer set.")

    def loadKnockoffTransferSet(
        self,
        dataset_name: str,
        transfer_size: int,
        sample_avg: int = 10,
        random_policy: bool = False,
        entropy: bool = True,
        force: bool = False,
    ) -> Tuple[Path, Dataset]:
        """
        Loads a dataset from a json file produced by self.generateKnockoffTransferSet().
        This json file will be stored in self.path/transfer_sets/

        This function will search all of the json files in this folder,
        looking for a transfer set that fits the provided arguments.
        If none exists and force is false, raises a filenotfound error.

        If force is true, generates the transfer set first.

        Note that the labels of the transfer set are still the original dataset labels,
        so when training a surrogate model, the victim model needs to be queried to get the labels.

        Return value is a tuple of (path to transfer set json file, Dataset object).

        Args:
            dataset_name (str): Name of dataset used for transfer set
            transfer_size (int): Size of transfer set
            sample_avg (int): Number of samples per class used
            random_policy (bool): Whether random sampling was used
            entropy (bool): Whether entropy was used for sampling
            force (bool): If True, generate new transfer set if none exists

        Returns:
            Tuple[Path, Dataset]: Tuple containing:
                - Path to transfer set JSON file
                - Dataset object for transfer set
        Raises:
            FileNotFoundError: If no matching transfer set found and force=False
        """
        config = {
            "dataset_name": dataset_name,
            "transfer_size": transfer_size,
            "sample_avg": sample_avg,
            "random_policy": random_policy,
            "entropy": entropy,
        }
        transfer_folder = self.path / "transfer_sets"
        valid = transfer_folder.exists()
        for file in transfer_folder.glob("*.json"):
            with open(file, "r+") as f:
                transfer_set_args = json.load(f)
            valid = True
            for arg in config:
                if config[arg] != transfer_set_args[arg]:
                    valid = False
                    break
            if valid:
                break
        if not valid:
            if force:
                self.generateKnockoffTransferSet(
                    dataset_name=dataset_name,
                    transfer_size=transfer_size,
                    sample_avg=sample_avg,
                    random_policy=random_policy,
                    entropy=entropy,
                )
                return self.loadKnockoffTransferSet(
                    dataset_name=dataset_name,
                    transfer_size=transfer_size,
                    sample_avg=sample_avg,
                    random_policy=random_policy,
                    entropy=entropy,
                )
            raise FileNotFoundError(
                f"""No transfer sets found in {transfer_folder} matching configuration\n
                {json.dumps(config, indent=4)}\nCall generateKnockoffTransferSet to make
                 one or set force=True to automatically generate one."""
            )

        # only need to add indices for the training data
        transfer_set = Dataset(
            dataset=transfer_set_args["dataset_name"],
            indices=(
                transfer_set_args["sample_indices"],
                [],
            ),  # 2nd element is for validation data
            deterministic=True,
            resize=getModelParams(self.architecture).get("input_size", None),
        )
        return file, transfer_set


class QuantizedModelManager(ProfiledModelManager):
    """
    Manages quantized versions of victim models.

    This class handles the creation and management of quantized versions of victim models,
    supporting both FBGEMM and QNNPACK backends for quantization.

    Attributes:
        FOLDER_NAME (str): Name of the folder for quantized models
        MODEL_FILENAME (str): Name of the quantized model checkpoint file
        backend (str): Quantization backend to use ('fbgemm' or 'qnnpack')
        victim_manager (VictimModelManager): The original victim model manager
    """

    FOLDER_NAME = "quantize"
    MODEL_FILENAME = "quantized.pt"

    def __init__(
        self,
        victim_model_path: Path,
        backend: str = "fbgemm",
        load_path: Optional[Path] = None,
        gpu: int = -1,
        save_model: bool = True,
    ) -> None:
        """
        Initialize a quantized model manager.

        Args:
            victim_model_path (Path): Path to the victim model checkpoint
            backend (str): Quantization backend to use ('fbgemm' or 'qnnpack')
            load_path (Optional[Path]): Path to existing quantized model if loading
            gpu (int): GPU device number (-1 for CPU)
            save_model (bool): Whether to save model checkpoints

        Raises:
            AssertionError: If victim model is not trained
            AssertionError: If victim model architecture is not supported for quantization
        """
        self.victim_manager = VictimModelManager.load(victim_model_path)
        assert self.victim_manager.config["epochs_trained"] > 0
        assert self.victim_manager.architecture in quantized_models
        path = self.victim_manager.path / self.FOLDER_NAME
        super().__init__(
            architecture=self.victim_manager.architecture,
            model_name=f"quantized_{self.victim_manager.architecture}",
            path=path,
            dataset=self.victim_manager.dataset.name,
            data_subset_percent=self.victim_manager.data_subset_percent,
            data_idx=self.victim_manager.data_idx,
            gpu=gpu,
            save_model=save_model,
        )
        self.backend = backend
        self.model = self.constructModel(quantized=True)
        if load_path is None:
            # construct this object for the first time
            self.quantize()
            if save_model:
                path.mkdir()
                self.saveModel(self.MODEL_FILENAME)
        else:
            self.loadQuantizedModel(load_path)
            # this causes stored parameters to overwrite new ones
            self.config.update(self.loadConfig(self.path))
            # update with the new device
            self.config["device"] = str(self.device)
            self.epochs_trained = self.config["epochs_trained"]

        self.saveConfig(
            {"victim_model_path": str(victim_model_path), "backend": backend}
        )

    def prepare_for_quantization(self) -> None:
        """
        Prepare the model for quantization.

        Sets up the quantization configuration and fuses model layers.
        The configuration depends on the selected backend:
        - FBGEMM: Uses per-channel weight quantization
        - QNNPACK: Uses default weight quantization
        """
        torch.backends.quantized.engine = self.backend
        self.model.eval()
        # Make sure that weight qconfig matches that of the serialized models
        if self.backend == "fbgemm":
            self.model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_per_channel_weight_observer,
            )
        elif self.backend == "qnnpack":
            self.model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_weight_observer,
            )

        self.model.fuse_model()
        torch.quantization.prepare(self.model, inplace=True)

    def quantize(self) -> None:
        """
        Quantize the model using the prepared configuration.

        This method:
        1. Prepares the model for quantization
        2. Calibrates the model using the victim model's training data
        3. Converts the model to a quantized version
        """
        self.prepare_for_quantization()
        # calibrate model
        dl_iter = tqdm(self.victim_manager.dataset.train_acc_dl)
        dl_iter.set_description("Calibrating model for quantization")
        for i, (x, y) in enumerate(dl_iter):
            self.model(x)
        torch.quantization.convert(self.model, inplace=True)

    def loadQuantizedModel(self, path: Path) -> None:
        """
        Load a quantized model from a checkpoint.

        Args:
            path (Path): Path to the quantized model checkpoint
        """
        self.prepare_for_quantization()
        torch.quantization.convert(self.model, inplace=True)
        self.loadModel(path)

    @staticmethod
    def load(model_path: Path, gpu: int = -1) -> "QuantizedModelManager":
        """
        Create a QuantizedModelManager from a saved quantized model checkpoint.

        Args:
            model_path (Path): Path to the quantized model checkpoint
            gpu (int): GPU device number (-1 for CPU)

        Returns:
            QuantizedModelManager: Loaded quantized model manager instance
        """
        quantized_folder = Path(model_path).parent
        victim_model_path = quantized_folder.parent / VictimModelManager.MODEL_FILENAME
        conf = ModelManagerBase.loadConfig(quantized_folder)
        return QuantizedModelManager(
            victim_model_path=victim_model_path,
            backend=conf["backend"],
            load_path=model_path,
            gpu=gpu,
        )


class PruneModelManager(ProfiledModelManager):
    """
    Manages pruned versions of victim models.

    This class handles the creation and management of pruned versions of victim models,
    supporting both unstructured and structured pruning strategies.

    Attributes:
        FOLDER_NAME (str): Name of the folder for pruned models
        MODEL_FILENAME (str): Name of the pruned model checkpoint file
        ratio (float): Pruning ratio (fraction of weights to prune)
        finetune_epochs (int): Number of epochs to finetune after pruning
        victim_manager (VictimModelManager): The original victim model manager
        pruned_modules (List[Tuple[torch.nn.Module, str]]): List of modules to prune
    """

    FOLDER_NAME = "prune"
    MODEL_FILENAME = "pruned.pt"

    def __init__(
        self,
        victim_model_path: Path,
        ratio: float = 0.5,
        finetune_epochs: int = 20,
        gpu: int = -1,
        load_path: Optional[Path] = None,
        save_model: bool = True,
        debug: Optional[int] = None,
    ) -> None:
        """
        Initialize a pruned model manager.

        Args:
            victim_model_path (Path): Path to the victim model checkpoint
            ratio (float): Fraction of weights to prune (0.0 to 1.0)
            finetune_epochs (int): Number of epochs to finetune after pruning
            gpu (int): GPU device number (-1 for CPU)
            load_path (Optional[Path]): Path to existing pruned model if loading
            save_model (bool): Whether to save model checkpoints
            debug (Optional[int]): Number of iterations to run in debug mode

        Raises:
            AssertionError: If victim model is not trained
        """
        self.victim_manager = VictimModelManager.load(victim_model_path, gpu=gpu)
        assert self.victim_manager.config["epochs_trained"] > 0
        path = self.victim_manager.path / self.FOLDER_NAME
        super().__init__(
            architecture=self.victim_manager.architecture,
            model_name=f"pruned_{self.victim_manager.architecture}",
            path=path,
            dataset=self.victim_manager.dataset.name,
            data_subset_percent=self.victim_manager.data_subset_percent,
            data_idx=self.victim_manager.data_idx,
            gpu=gpu,
            save_model=save_model,
        )
        self.ratio = ratio
        self.finetune_epochs = finetune_epochs
        self.model = self.constructModel()
        self.pruned_modules = self.paramsToPrune()

        if load_path is None:
            if save_model:
                # this must be called before self.prune() because
                # self.prune() calls updateConfigSparsity() which
                # assumes the path exists.
                path.mkdir()
            # construct this object for the first time
            self.prune()  # modifies self.model
            # finetune
            self.trainModel(finetune_epochs, debug=debug)
        else:
            self.loadModel(load_path)
            # this causes stored parameters to overwrite new ones
            self.config.update(self.loadConfig(self.path))
            # update with the new device
            self.config["device"] = str(self.device)
            self.epochs_trained = self.config["epochs_trained"]

        self.saveConfig(
            {
                "victim_model_path": str(victim_model_path),
                "ratio": ratio,
                "finetune_epochs": finetune_epochs,
            }
        )

    def prune(self) -> None:
        """
        Prune the model using L1 unstructured pruning.

        This method:
        1. Applies L1 unstructured pruning to the selected modules
        2. Updates the configuration with sparsity information
        """
        prune.global_unstructured(
            self.pruned_modules,
            pruning_method=prune.L1Unstructured,
            amount=self.ratio,
        )
        self.updateConfigSparsity()

    def paramsToPrune(
        self, min_dims: Optional[int] = None, conv_only: bool = False
    ) -> List[Tuple[torch.nn.Module, str]]:
        """
        Get list of parameters to prune.

        Args:
            min_dims (Optional[int]): If provided, only add modules whose weight
                parameter has at least min_dims dimensions. Used for structured pruning.
            conv_only (bool): If True, only prune convolution layers.

        Returns:
            List[Tuple[torch.nn.Module, str]]: List of (module, parameter_name) pairs to prune
        """
        res = []
        for name, module in self.model.named_modules():
            if name.startswith("classifier"):
                continue
            if name.startswith("fc"):
                continue
            if hasattr(module, "weight"):
                if min_dims == None or module.weight.ndim > min_dims:
                    if not conv_only or isinstance(
                        module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
                    ):
                        res.append((module, "weight"))
        self.pruned_modules = res
        return res

    def updateConfigSparsity(self) -> None:
        """
        Update configuration with sparsity information.

        Calculates and stores:
        - Per-module sparsity percentages
        - Total number of parameters
        - Number of zero parameters
        - Total sparsity percentage
        """
        sparsity = {"module_sparsity": {}}

        pruned_mods_reformatted = {}
        for mod, name in self.pruned_modules:
            if mod not in pruned_mods_reformatted:
                pruned_mods_reformatted[mod] = [name]
            else:
                pruned_mods_reformatted[mod].append(name)

        zero_params_count = 0
        for name, module in self.model.named_modules():
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

        # get total sparsity
        # getting total number of parameters from
        # https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model/62764464#62764464
        total_params = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters()).values()
        )
        sparsity["total_parameters"] = int(total_params)
        sparsity["zero_parameters"] = int(zero_params_count)
        # the percentage of zero params
        sparsity["total_sparsity"] = (
            100 * float(zero_params_count) / float(total_params)
        )
        self.config["sparsity"] = sparsity
        self.saveConfig()

    @staticmethod
    def load(model_path: Path, gpu: int = -1) -> "PruneModelManager":
        """
        Create a PruneModelManager from a saved pruned model checkpoint.

        Args:
            model_path (Path): Path to the pruned model checkpoint
            gpu (int): GPU device number (-1 for CPU)

        Returns:
            PruneModelManager: Loaded pruned model manager instance
        """
        load_folder = Path(model_path).parent  # the prune folder
        victim_path = load_folder.parent / VictimModelManager.MODEL_FILENAME
        conf = ModelManagerBase.loadConfig(load_folder)
        return PruneModelManager(
            victim_model_path=victim_path,
            ratio=conf["ratio"],
            finetune_epochs=conf["finetune_epochs"],
            gpu=gpu,
            load_path=model_path,
        )


class StructuredPruneModelManager(PruneModelManager):
    """
    Manages structured pruning of victim models.

    This class extends PruneModelManager to implement structured pruning, which removes
    entire channels or filters from convolutional layers. This can lead to more efficient
    models than unstructured pruning.

    Attributes:
        FOLDER_NAME (str): Name of the folder for structured pruned models
        MODEL_FILENAME (str): Name of the structured pruned model checkpoint file
    """

    FOLDER_NAME = "structured_prune"
    MODEL_FILENAME = "structured_pruned.pt"

    def prune(self) -> None:
        """
        Prune the model using L2 structured pruning.

        This method:
        1. Applies L2 structured pruning to the selected modules
        2. Updates the configuration with sparsity information

        Note:
            Uses L2 norm for pruning and operates on the channel dimension (dim=1)
            of convolutional layers.
        """
        for module, name in self.pruned_modules:
            prune.ln_structured(
                module=module,
                name=name,
                amount=self.ratio,
                n=2,
                dim=1,  # dim 1 is the channel dimension in PyTorch
            )
        self.updateConfigSparsity()

    def paramsToPrune(
        self, min_dims: int = 2, conv_only: bool = True
    ) -> List[Tuple[torch.nn.Module, str]]:
        # override superclass implementation and change default args
        return super().paramsToPrune(min_dims, conv_only)

    @staticmethod
    def load(model_path: Path, gpu: int = -1) -> "StructuredPruneModelManager":
        """
        Create a StructuredPruneModelManager from a saved structured pruned model checkpoint.

        Args:
            model_path (Path): Path to the structured pruned model checkpoint
            gpu (int): GPU device number (-1 for CPU)

        Returns:
            StructuredPruneModelManager: Loaded structured pruned model manager instance
        """
        load_folder = Path(model_path).parent  # the prune folder
        victim_path = load_folder.parent / VictimModelManager.MODEL_FILENAME
        conf = ModelManagerBase.loadConfig(load_folder)
        return StructuredPruneModelManager(
            victim_model_path=victim_path,
            ratio=conf["ratio"],
            finetune_epochs=conf["finetune_epochs"],
            gpu=gpu,
            load_path=model_path,
        )


class SurrogateModelManager(ModelManagerBase):
    """
    Manages surrogate models for model extraction attacks.

    This class handles the creation and training of surrogate models that mimic victim models.
    It supports both knowledge distillation and knockoff transfer set training strategies.
    The surrogate model is trained to match the victim model's predictions.

    Attributes:
        FOLDER_NAME (str): Name of the folder for surrogate models
        victim_model (Union[VictimModelManager, PruneModelManager, QuantizedModelManager]):
            The victim model being mimicked
        arch_pred_model_name (str): Name of the architecture prediction model used
        arch_confidence (float): Confidence score of architecture prediction
        pretrained (bool): Whether the model uses pretrained weights
        knockoff_transfer_set (Optional[Tuple[Path, Dataset]]): Transfer set for knockoff training
        train_with_transfer_set (bool): Whether to use transfer set for training
    """

    FOLDER_NAME = "surrogate"

    def __init__(
        self,
        victim_model_path: Path,
        architecture: str,
        arch_conf: float,
        arch_pred_model_name: str,
        pretrained: bool = False,
        load_path: Optional[Path] = None,
        gpu: int = -1,
        save_model: bool = True,
        data_idx: int = 1,
    ) -> None:
        """
        Initialize a surrogate model manager.

        Args:
            victim_model_path (Path): Path to the victim model checkpoint
            architecture (str): Architecture of the surrogate model
            arch_conf (float): Confidence score of architecture prediction
            arch_pred_model_name (str): Name of the architecture prediction model
            pretrained (bool): Whether to use pretrained weights
            load_path (Optional[Path]): Path to existing surrogate model if loading
            gpu (int): GPU device number (-1 for CPU)
            save_model (bool): Whether to save model checkpoints
            data_idx (int): Index for dataset subset

        Note:
            The self.config only accounts for keeping track of one history of
            victim model architecture prediction. Assumes victim model has been
            profiled/predicted.
        """
        self.victim_model = self.loadVictim(victim_model_path, gpu=gpu)
        if isinstance(self.victim_model, VictimModelManager):
            assert self.victim_model.config["epochs_trained"] > 0
        self.arch_pred_model_name = arch_pred_model_name
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.victim_model.path / f"{self.FOLDER_NAME}_{timestamp}"
        self.pretrained = pretrained
        if load_path is None:
            # creating this object for the first time
            if not self.victim_model.isProfiled():
                print(
                    f"""Warning, victim model {self.victim_model.path} has not been 
                    profiled and a surrogate model is being created."""
                )
            self.arch_confidence = arch_conf
            if save_model:
                assert not path.exists()
                path.mkdir(parents=True)
        else:
            # load from a previous run
            folder = load_path.parent
            config = self.loadConfig(folder)
            self.arch_confidence = config["arch_confidence"]
            # TODO the following line is a hack, should be addressed.  The name of
            # the surrogate model including the timestamp is not included in the config
            path = self.victim_model.path / Path(config["path"]).name
        super().__init__(
            architecture=architecture,
            model_name=f"surrogate_{self.victim_model.model_name}_{architecture}",
            path=path,
            dataset=self.victim_model.dataset.name,
            data_subset_percent=1 - self.victim_model.data_subset_percent,
            data_idx=data_idx,
            gpu=gpu,
            save_model=save_model,
        )
        self.model = self.constructModel(
            pretrained=self.pretrained and load_path is None
        )
        if load_path is not None:
            self.loadModel(load_path)
            # this causes stored parameters to overwrite new ones
            self.config.update(self.loadConfig(self.path))
            # update with the new device
            self.config["device"] = str(self.device)
            self.epochs_trained = self.config["epochs_trained"]
            self.pretrained = self.config["pretrained"]
        self.saveConfig(
            {
                "victim_model_path": str(victim_model_path),
                "arch_pred_model_name": arch_pred_model_name,
                "arch_confidence": self.arch_confidence,
                "pretrained": self.pretrained,
                "knockoff_transfer_set_path": "",
                "knockoff_transfer_set": None,
            }
        )
        self.train_metrics.extend(
            [
                "train_agreement",
                "val_agreement",
                "l1_weight_bound",
                "transfer_attack_success",
            ]
        )
        self.knockoff_transfer_set = None
        self.train_with_transfer_set = (
            False  # if true, the transfer set will be used for training
        )

    def loadKnockoffTransferSet(
        self,
        dataset_name: str,
        transfer_size: int,
        sample_avg: int = 10,
        random_policy: bool = False,
        entropy: bool = True,
        force: bool = False,
    ) -> None:
        """
        Load a knockoff transfer set for training the surrogate model.

        Args:
            dataset_name (str): Name of dataset to use for transfer set
            transfer_size (int): Size of transfer set to generate
            sample_avg (int): Number of samples per class for entropy estimation
            random_policy (bool): If True, use random sampling strategy
            entropy (bool): If True, use entropy for adaptive sampling
            force (bool): If True, generate new transfer set if none exists

        Note:
            The transfer set is used to train the surrogate model using the
            knockoff nets approach. This can be more efficient than pure
            knowledge distillation.
        """
        self.knockoff_transfer_set = self.victim_model.loadKnockoffTransferSet(
            dataset_name=dataset_name,
            transfer_size=transfer_size,
            sample_avg=sample_avg,
            random_policy=random_policy,
            entropy=entropy,
            force=force,
        )
        self.train_with_transfer_set = True
        self.saveConfig(
            {
                "knockoff_transfer_set": {
                    "path": str(self.knockoff_transfer_set[0]),
                    "dataset_name": dataset_name,
                    "transfer_size": transfer_size,
                    "sample_avg": sample_avg,
                    "random_policy": random_policy,
                    "entropy": entropy,
                },
                "knockoff_transfer_set_path": str(self.knockoff_transfer_set[0]),
            }
        )

    @staticmethod
    def load(model_path: str, gpu: int = -1) -> "SurrogateModelManager":
        """
        Create a SurrogateModelManager from a saved surrogate model checkpoint.

        Args:
            model_path (str): Path to the surrogate model checkpoint
            gpu (int): GPU device number (-1 for CPU)

        Returns:
            SurrogateModelManager: Loaded surrogate model manager instance

        Note:
            The model path should be under {victim_model_path}/surrogate_{time}/checkpoint.pt
        """
        model_path = Path(model_path)
        vict_model_path = model_path.parent.parent / VictimModelManager.MODEL_FILENAME
        load_folder = model_path.parent
        conf = ModelManagerBase.loadConfig(load_folder)
        surrogate_manager = SurrogateModelManager(
            victim_model_path=vict_model_path,
            architecture=conf["architecture"],
            arch_conf=conf["arch_confidence"],
            arch_pred_model_name=conf["arch_pred_model_name"],
            data_idx=conf["data_idx"],
            pretrained=conf["pretrained"],
            load_path=model_path,
            gpu=gpu,
        )
        print(f"Loaded surrogate model\n{model_path}\n")
        if conf["knockoff_transfer_set"] is not None:
            surrogate_manager.loadKnockoffTransferSet(
                dataset_name=conf["knockoff_transfer_set"]["dataset_name"],
                transfer_size=conf["knockoff_transfer_set"]["transfer_size"],
                sample_avg=conf["knockoff_transfer_set"]["sample_avg"],
                random_policy=conf["knockoff_transfer_set"]["random_policy"],
                entropy=conf["knockoff_transfer_set"]["entropy"],
                force=False,
            )
        return surrogate_manager

    @staticmethod
    def loadVictimConfig(path: Path) -> Dict[str, Any]:
        """
        Load configuration from the associated victim model.

        Args:
            path (Path): Path to surrogate model manager folder

        Returns:
            Dict[str, Any]: Configuration dictionary from victim model
        """
        return VictimModelManager.loadConfig(path.parent)

    def collectEpochMetrics(
        self,
        epoch_num: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        debug: Optional[int] = None,
        run_attack: bool = True,
    ) -> Dict[str, float]:
        """
        Collect metrics for a training epoch.

        Args:
            epoch_num (int): Current epoch number
            optimizer (torch.optim.Optimizer): Model optimizer
            loss_fn (Callable): Loss function
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
            debug (Optional[int]): Number of iterations to run in debug mode
            run_attack (bool): Whether to run transfer attack during training

        Returns:
            Dict[str, float]: Dictionary of metrics including:
                - Training/validation loss and accuracy
                - Agreement with victim model
                - L1 weight norm difference
                - Transfer attack success rate
        """
        loss, acc1, acc5, agreement = self.runEpoch(
            train=True,
            epoch=epoch_num,
            optim=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            debug=debug,
        )
        val_loss, val_acc1, val_acc5, val_agreement = self.runEpoch(
            train=False,
            epoch=epoch_num,
            optim=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            debug=debug,
        )

        l1_weight_bound = self.getL1WeightNorm(self.victim_model)

        transfer_attack_success = -1
        if run_attack:
            # run transfer on victim validation data
            results = self.transferAttackPGD(
                self.victim_model.dataset, dataloader_name="val_dl", debug=debug
            )
            transfer_attack_success = 1 - results["victim_correct1"]

        metrics = {
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "lr": optimizer.param_groups[0]["lr"],
            "train_agreement": agreement,
            "val_agreement": val_agreement,
            "l1_weight_bound": l1_weight_bound,
            "transfer_attack_success": transfer_attack_success,
        }
        return metrics

    def runEpoch(
        self,
        train: bool,
        epoch: int,
        optim: torch.optim.Optimizer,
        loss_fn: Callable,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        debug: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Run a single training or validation epoch.

        Args:
            train (bool): Whether this is a training epoch
            epoch (int): Current epoch number
            optim (torch.optim.Optimizer): Model optimizer
            loss_fn (Callable): Loss function
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
            debug (Optional[int]): Number of iterations to run in debug mode

        Returns:
            Tuple[float, float, float, float]: Tuple containing:
                - Loss value
                - Top-1 accuracy
                - Top-5 accuracy
                - Agreement percentage with victim model

        Note:
            Uses L1 loss between victim model predictions and surrogate model predictions.
            Validation is done on victim's validation dataset.
            Agreement is the percentage of samples where victim and surrogate's
            top-1 predictions match.
        """
        self.model.eval()
        prefix = "val"
        dl = self.victim_model.dataset.val_dl
        if train:
            self.model.train()
            prefix = "train"
            if self.train_with_transfer_set:
                dl = self.knockoff_transfer_set[1].train_dl
            else:
                dl = self.dataset.train_dl
        train_loss = torch.nn.L1Loss()

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()
        agreement = OnlineStats()
        step_size = OnlineStats()
        step_size.add(optim.param_groups[0]["lr"])

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(
            f"{prefix.capitalize()} Epoch {epoch if train else '1'}/{self.epochs if train else '1'}"
        )

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                if debug and i > debug:
                    break
                x, y = x.to(self.device), y.to(self.device)
                victim_yhat = self.victim_model.model(x)
                victim_yhat = torch.autograd.Variable(victim_yhat, requires_grad=False)
                yhat = self.model(x)
                loss = train_loss(yhat, victim_yhat)
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                c1, c5 = correct(yhat, y, (1, 5))
                agreement.add(
                    correct(yhat, torch.argmax(victim_yhat, dim=1))[0] / len(x)
                )
                total_loss.add(loss.item() / len(x))
                acc1.add(c1 / len(x))
                acc5.add(c5 / len(x))

                epoch_iter.set_postfix(
                    loss=total_loss.mean,
                    top1=acc1.mean,
                    top5=acc5.mean,
                    agreement=agreement.mean,
                    step_size=step_size.mean,
                )

        loss = total_loss.mean
        top1 = acc1.mean
        top5 = acc5.mean
        agreement = agreement.mean

        if train and debug is None:
            lr_scheduler.step(loss)

        return loss, top1, top5, agreement

    def transferAttackPGD(
        self,
        dataset: Dataset,
        dataloader_name: str,
        eps: float = 8 / 255,
        step_size: float = 2 / 255,
        iterations: int = 10,
        norm: float = np.inf,
        debug: Optional[int] = None,
    ) -> Dict[str, Union[float, int]]:
        """
        Run a transfer attack using Projected Gradient Descent.

        Generates adversarial examples on the surrogate model and tests their
        transferability to the victim model.

        Args:
            dataset (Dataset): Dataset to run attack on
            dataloader_name (str): Name of dataloader to use (e.g. "train_dl", "val_dl")
            eps (float): Maximum perturbation size
            step_size (float): Step size for gradient updates
            iterations (int): Number of PGD iterations
            norm (float): Norm for projection (default: inf)
            debug (Optional[int]): Number of iterations to run in debug mode

        Returns:
            Dict[str, Union[float, int]]: Dictionary containing:
                - Number of inputs tested
                - Accuracy metrics for both models
                - Transfer attack success rate
                - Runtime and parameters

        Note:
            Code adapted from cleverhans tutorial:
        https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/torch/cifar10_tutorial.py
        """
        topk = (1, 5)

        since = time.time()
        self.model.eval()
        self.model.to(self.device)
        self.victim_model.model.to(self.device)
        self.victim_model.model.eval()

        results = {
            "inputs_tested": 0,
            "both_correct1": 0,
            "both_correct5": 0,
            "surrogate_correct1": 0,
            "surrogate_correct5": 0,
            "victim_correct1": 0,
            "victim_correct5": 0,
        }

        surrogate_acc1 = OnlineStats()
        surrogate_acc5 = OnlineStats()
        victim_acc1 = OnlineStats()
        victim_acc5 = OnlineStats()

        dl = getattr(dataset, dataloader_name)
        epoch_iter = tqdm(dl)
        epoch_iter.set_description(
            f"Running PGD Transfer attack from {self} to {self.victim_model}"
        )

        for i, (x, y) in enumerate(epoch_iter, start=1):
            x, y = x.to(self.device), y.to(self.device)

            # generate adversarial examples using the surrogate model
            x_adv_surrogate = self.runPGD(x, eps, step_size, iterations, norm)
            # get predictions from surrogate and victim model
            y_pred_surrogate = self.model(x_adv_surrogate)
            y_pred_victim = self.victim_model.model(x_adv_surrogate)

            results["inputs_tested"] += y.size(0)

            adv_c1_surrogate, adv_c5_surrogate = correct(y_pred_surrogate, y, topk)
            results["surrogate_correct1"] += adv_c1_surrogate
            results["surrogate_correct5"] += adv_c5_surrogate
            surrogate_acc1.add(adv_c1_surrogate / dl.batch_size)
            surrogate_acc5.add(adv_c5_surrogate / dl.batch_size)

            adv_c1_victim, adv_c5_victim = correct(y_pred_victim, y, topk)
            # if the attack is successful, then these values should be both
            # (1) low values, meaning low accuracy on the adversarial images
            # and (2) similar values to the surrogate values
            results["victim_correct1"] += adv_c1_victim
            results["victim_correct5"] += adv_c5_victim
            victim_acc1.add(adv_c1_victim / dl.batch_size)
            victim_acc5.add(adv_c5_victim / dl.batch_size)

            both_correct1, both_correct5 = both_correct(
                y_pred_surrogate, y_pred_victim, y, topk
            )
            results["both_correct1"] += both_correct1
            results["both_correct5"] += both_correct5

            epoch_iter.set_postfix(
                surrogate_acc1=surrogate_acc1.mean,
                surrogate_acc5=surrogate_acc5.mean,
                victim_acc1=victim_acc1.mean,
                victim_acc5=victim_acc5.mean,
            )

            if debug is not None and i == debug:
                break

        results["both_correct1"] = results["both_correct1"] / results["inputs_tested"]
        results["both_correct5"] = results["both_correct5"] / results["inputs_tested"]

        results["surrogate_correct1"] = (
            results["surrogate_correct1"] / results["inputs_tested"]
        )
        results["surrogate_correct5"] = (
            results["surrogate_correct5"] / results["inputs_tested"]
        )

        results["victim_correct1"] = (
            results["victim_correct1"] / results["inputs_tested"]
        )
        results["victim_correct5"] = (
            results["victim_correct5"] / results["inputs_tested"]
        )

        results["transfer_runtime"] = time.time() - since
        results["parameters"] = {
            "eps": eps,
            "step_size": step_size,
            "iterations": iterations,
            "dataset": dataset.name,
            "dataloader": dataloader_name,
        }

        print(json.dumps(results, indent=4))
        if debug is None:
            if "transfer_results" not in self.config:
                self.config["transfer_results"] = {f"{dataset.name}_results": results}
            else:
                self.config["transfer_results"][f"{dataset.name}_results"] = results
            self.saveConfig()
        return results

    def loadVictim(
        self, victim_model_path: str, gpu: int
    ) -> Union[VictimModelManager, PruneModelManager, QuantizedModelManager]:
        """
        Load the victim model based on its type.

        Args:
            victim_model_path (str): Path to victim model checkpoint
            gpu (int): GPU device number (-1 for CPU)

        Returns:
            Union[VictimModelManager, PruneModelManager, QuantizedModelManager]:
                The appropriate type of victim model manager based on the model path
        """
        victim_folder = Path(victim_model_path).parent
        if victim_folder.name == PruneModelManager.FOLDER_NAME:
            return PruneModelManager.load(model_path=victim_model_path, gpu=gpu)
        if victim_folder.name == QuantizedModelManager.FOLDER_NAME:
            return QuantizedModelManager.load(model_path=victim_model_path, gpu=gpu)
        return VictimModelManager.load(model_path=victim_model_path, gpu=gpu)


def getVictimSurrogateModels(
    architectures: Optional[List[str]] = None,
    victim_args: Dict[str, Any] = {},
    surrogate_args: Dict[str, Any] = {},
) -> Dict[Path, List[Path]]:
    """
    Get paths to victim models and their associated surrogate models that match specified criteria.

    This function searches for victim models and their surrogate models that match the provided
    architecture and configuration arguments. It returns a mapping from victim model paths to
    lists of associated surrogate model paths.

    Args:
        architectures (Optional[List[str]]): List of model architectures to filter by.
            If None, all architectures are considered.
        victim_args (Dict[str, Any]): Dictionary of configuration parameters that victim
            models must match. Keys should match configuration parameters in the victim
            model's config file.
        surrogate_args (Dict[str, Any]): Dictionary of configuration parameters that surrogate
            models must match. Keys should match configuration parameters in the surrogate
            model's config file.

    Returns:
        Dict[Path, List[Path]]: Dictionary mapping victim model paths to lists of associated
            surrogate model paths. Only includes victim models that have at least one matching
            surrogate model.

    Example:
        ```python
        # Find all ResNet18 victim models and their surrogate models trained with
        # knowledge distillation
        models = getVictimSurrogateModels(
            architectures=["resnet18"],
            victim_args={"dataset": "cifar10"},
            surrogate_args={"knockoff_transfer_set": None}
        )
        ```
    """

    def validManager(
        victim_path: Path,
    ) -> Dict[Path, List[Path]]:
        """
        Check if a victim model and its surrogate models match the specified criteria.

        This helper function checks if a victim model's configuration matches the provided
        victim_args and if it has any surrogate models that match the surrogate_args.

        Args:
            victim_path (Path): Path to the victim model checkpoint file.

        Returns:
            Dict[Path, List[Path]]: Dictionary containing the victim path as key and a list
                of matching surrogate model paths as value. Returns empty dict if victim
                model doesn't match criteria.

        Note:
            The function checks both the victim model's configuration and the configurations
            of all surrogate models in the victim model's directory.
        """
        vict_config = VictimModelManager.loadConfig(victim_path.parent)
        # check victim
        if not checkDict(vict_config, victim_args):
            return {}
        result = {vict_path: []}
        # check surrogate
        surrogate_paths = list(
            vict_path.parent.glob(f"{SurrogateModelManager.FOLDER_NAME}*")
        )
        for surrogate_path in surrogate_paths:
            surrogate_config = SurrogateModelManager.loadConfig(surrogate_path)
            if checkDict(surrogate_config, surrogate_args):
                result[vict_path].append(
                    surrogate_path / SurrogateModelManager.MODEL_FILENAME
                )

        return result

    victim_paths = VictimModelManager.getModelPaths(architectures=architectures)
    result = {}
    for vict_path in victim_paths:
        result.update(validManager(victim_path=vict_path))
    return result


def getModelsFromSurrogateTrainStrategies(
    strategies: Dict[str, Dict[str, Any]],
    architectures: List[str],
    latest_file: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    Get paths to surrogate models that match specified training strategies and architectures.

    This function searches for surrogate models that match the provided training strategies
    and architectures. It returns a nested dictionary mapping strategy names to architecture
    names to model paths.

    Args:
        strategies (Dict[str, Dict[str, Any]]): Dictionary mapping strategy names to their
            configuration parameters. Each strategy configuration should include:
            - pretrained (bool): Whether the model uses pretrained weights
            - knockoff_transfer_set (Optional[Dict[str, Any]]): Configuration for knockoff
                training. If None, indicates pure knowledge distillation. If specified,
                should include:
                - dataset_name (str): Name of dataset used
                - transfer_size (int): Size of transfer set
                - sample_avg (int): Number of samples per class
                - random_policy (bool): Whether to use random sampling
                - entropy (bool): Whether to use entropy for sampling
        architectures (List[str]): List of model architectures to search for
        latest_file (bool): If True, return the most recent model file when multiple
            matches are found. If False, return the oldest file.

    Returns:
        Dict[str, Dict[str, Path]]: Nested dictionary where:
            - Outer key is strategy name
            - Inner key is architecture name
            - Value is path to matching surrogate model

    Raises:
        AssertionError: If no matching surrogate model is found for any
            architecture-strategy combination

    Example:
        ```python
        strategies = {
            "knowledge_dist": {
            "pretrained": False,
            "knockoff_transfer_set": None,
        },
            "knockoff1": {
            "pretrained": False,
            "knockoff_transfer_set": {
                "dataset_name": "cifar100",
                "transfer_size": 20000,
                "sample_avg": 100,
                "random_policy": False,
                "entropy": False,
            },
        },
    }
        architectures = ["resnet18", "googlenet"]

        models = getModelsFromSurrogateTrainStrategies(
            strategies=strategies,
            architectures=architectures,
            latest_file=True
        )
        # Returns:
        # {
        #     "knowledge_dist": {
        #         "resnet18": <path to resnet18 surrogate model>,
        #         "googlenet": <path to googlenet surrogate model>,
        #     },
        #     "knockoff1": {
        #         "resnet18": <path to resnet18 surrogate model>,
        #         "googlenet": <path to googlenet surrogate model>,
        #     },
        # }
        ```
    """
    result = {}
    for strategy in strategies:
        strategy_result = {}
        # formatted as {path to victim manager: [list of surrogate paths]}
        # we only want 1 surrogate path
        models_satisfying_strategy = getVictimSurrogateModels(
            surrogate_args=strategies[strategy], architectures=architectures
        )
        for arch in architectures:
            found = False
            for vict_path in models_satisfying_strategy:
                if (
                    vict_path.parent.parent.name == arch
                ):  # str(vict_path).find(arch) >= 0:
                    found = True
                    assert (
                        len(models_satisfying_strategy[vict_path]) > 0
                    ), f"Could not find any surrogate models with arch {arch} and strategy {strategies[strategy]}"
                    strategy_result[arch] = latestFileFromList(
                        models_satisfying_strategy[vict_path], oldest=not latest_file
                    )
                    break
            assert (
                found
            ), f"Could not find any surrogate models with arch {arch} and strategy {strategies[strategy]}"
        result[strategy] = strategy_result
    return result
