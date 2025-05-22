"""
Generic model training from Pytorch model zoo, used for the victim model.

Assuming that the victim model architecture has already been found/predicted,
the surrogate model can be trained using labels from the victim model.
"""

import datetime
import json
import random
import shutil
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from cleverhans.torch.attacks.projected_gradient_descent import \
    projected_gradient_descent
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from tqdm import tqdm

from architecture_prediction import ArchPredBase
from collect_profiles import generateExeName, run_command
from datasets import Dataset
from format_profiles import avgProfiles, parse_one_profile
from get_model import (get_model, get_quantized_model, getModelParams,
                       quantized_models)
from logger import CSVLogger
from model_metrics import accuracy, both_correct, correct
from online import OnlineStats
from utils import checkDict, latest_file, latestFileFromList


def loadModel(path: Path, model: torch.nn.Module, device: torch.device = None) -> None:
    assert path.exists(), f"Model load path \n{path}\n does not exist."
    if device is not None:
        params = torch.load(path, map_location=device)
    else:
        params = torch.load(path)
    model.load_state_dict(params, strict=False)
    model.eval()


class ModelManagerBase(ABC):
    """
    Generic model manager class.
    Can train a model on a dataset and save/load a model.

    Note- inheriting classes should not modify self.config until
    after calling this constructor because this constructor will overwrite
    self.config

    This constructor leaves no footprint on the filesystem.
    """

    MODEL_FILENAME = "checkpoint.pt"

    def __init__(
        self,
        architecture: str,
        model_name: str,
        path: Path,
        dataset: str,
        data_subset_percent: float = None,
        data_idx: int = 0,
        gpu: int = -1,
        save_model: bool = True,
    ) -> None:
        """
        path: path to folder
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

        self.epochs_trained = 0  # if loading a model, this gets updated in inheriting classes' constructors
        # when victim models are loaded, they call self.loadModel, which calls
        # self.saveConfig().  If config["epochs_trained"] = 0 when this happens,
        # then the actual epochs_trained is overwritten.  To prevent this, we
        # have the check below.
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
        self, pretrained: bool = False, quantized: bool = False, kwargs=None
    ) -> torch.nn.Module:
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
        assert Path(path).exists(), f"Model load path \n{path}\n does not exist."
        # the epochs trained should already be done from loading the config.
        # if "_" in str(path.name):
        #     self.epochs_trained = int(str(path.name).split("_")[1].split(".")[0])
        params = torch.load(path, map_location=self.device)
        self.model.load_state_dict(params, strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.model_path = path
        self.saveConfig({"model_path": str(path)})

    def saveModel(
        self, name: str = None, epoch: int = None, replace: bool = False
    ) -> None:
        """
        If epoch is passed, will append '_<epoch>' before the file extension
        in <name>.  Example: if name="checkpoint.pt" and epoch = 10, then
        will save as "checkpoint_10.pt".
        If replace is true, then if the model file already exists it will be replaced.
        If replace is false and the model file already exists, an error is raised.
        """
        if not self.save_model:
            return
        # todo add remove option
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
        return Dataset(
            name,
            data_subset_percent=self.data_subset_percent,
            idx=self.data_idx,
            resize=getModelParams(self.architecture).get("input_size", None),
        )

    @staticmethod
    @abstractmethod
    def load(self, path: Path, gpu: int = -1):
        """Abstract method"""

    def saveConfig(self, args: dict = {}) -> None:
        """
        Write parameters to a json file.  If file exists already, then will be
        appended to/overwritten.  If args are provided, they are added to the config file.
        """
        if not self.save_model:
            return
        self.config.update(args)
        # look for config file
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

    def delete(self):
        """
        Deletes the folder for this model.
        """
        path = Path(self.path)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            print(f"Deleted {str(path.relative_to(Path.cwd()))}")

    @staticmethod
    def loadConfig(path: Path) -> dict:
        """
        Given a path to a model manager folder, return its config file as a dict.
        path is not hardcoded as self.path so that this method can be called
        by inheriting classes before self.path is set (which occurs when invoking
        this class's constructor).
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
        train_logfile = path / "logs.csv"
        assert train_logfile.exists()
        return pd.read_csv(train_logfile)

    @staticmethod
    def saveConfigFast(path: Path, args: dict, replace: bool = False):
        """
        Given a path to a model manager folder and config arguments,
        save the arguments in the config folder, rewriting existing arguments if
        replace is true.  Provided as a static method to allow config
        alteration without loading modelmanager object.
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
        lr: float = None,
        debug: int = None,
        patience: int = 10,
        replace: bool = False,
        run_attack: bool = False,
    ):
        """Trains the model using dataset self.dataset.

        Args:
            num_epochs (int): number of training epochs
            lr (float): initial learning rate.  This function decreases the learning rate
                by a factor of 0.1 when the loss fails to decrease by 1e-4 for 10 iterations.
                If not passed, will default to learning rate of model from get_model.py, and
                if there is no learning rate specified there, defaults to 0.1.
            save_freq: how often to save model, default is only at the end. models are overwritten.

        Returns:
            Nothing, only sets the self.model class variable.
        """
        # todo add checkpoint freq, also make note to flush logger when checkpointing
        assert self.dataset is not None
        assert self.model is not None, "Must call constructModel() first"

        if num_epochs == 0:
            self.model.eval()
            print("Training ended, saving model.")
            self.saveModel(
                replace=replace
            )  # this function already checks self.save_model
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
                    # we only write to the log file once the model is saved,
                    # see below after self.saveModel()
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
        self.saveModel(replace=replace)  # this function already checks self.save_model
        if self.save_model:
            logger.flush()  # this saves all the futureWrite() calls
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
        debug: int = None,
        run_attack: bool = False,
    ):
        # todo implement run_attack like it is implemented for surrogate models
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
        debug: int = None,
    ) -> tuple[int]:
        """Run a single epoch."""

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
        norm=np.inf,
    ) -> torch.Tensor:
        return projected_gradient_descent(
            self.model, x, eps=eps, eps_iter=step_size, nb_iter=iterations, norm=norm
        )

    def topKAcc(self, dataloader: torch.utils.data.DataLoader, topk=(1, 5)):
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

    def getL1WeightNorm(self, other) -> float:
        """
        Return the sum of elementwise differences between parameters of
        2 models of the same architecture.
        """
        if not isinstance(other.model, type(self.model)):
            raise ValueError
            # return float("inf")

        l1_norm = 0.0
        for x, y in zip(
            self.model.state_dict().values(), other.model.state_dict().values()
        ):
            if x.shape == y.shape:
                l1_norm += (x - y).abs().sum().item()
        return l1_norm

    def __repr__(self) -> str:
        return str(self.path.relative_to(Path.cwd()))


class ProfiledModelManager(ModelManagerBase):
    """Extends ModelManagerBase to include support for profiling"""

    def runNVProf(
        self, use_exe: bool = True, seed: int = 47, n: int = 10, input: str = "0"
    ):
        """
        Creates a subfolder self.path/profiles, and adds a profile file profile_{pid}.csv and
        associated params_{pid}.json file to this subfolder, if the profile succeeded.
        There is support for multiple profiles.
        Note - this function does not check for collisions in pid.
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
        Checks if the model has been profiled. Returns True if there
        is a subfolder self.path/profiles with at least one profile_{pid}.csv
        and associated params_{pid}.csv.
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

    def getProfile(self, filters: dict = None) -> Tuple[Path, Dict]:
        """
        Return a tuple of (path to profile_{pid}.csv,
        dictionary obtained from reading params_{pid}.json).
        filters: a dict and each argument in the dict must match
            the argument from the config file associated with a profile.
            to get a profile by name, can specify {"profile_number": "2181935"}
        If there are multiple profiles which fit the filters, return the latest one.
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

    def getAllProfiles(self, filters: dict = None) -> List[Tuple[Path, Dict]]:
        """
        Returns a list of tuples (path to profile_{pid}.csv,
        dictionary obtained from reading params_{pid}.json) for
        every profile in self.path/profiles
        filters: a dict and each argument in the dict must match
            the argument from the config file associated with a profile.
            to get a profile by name, can specify {"profile_number": "2181935"}
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
        self, arch_pred_model: ArchPredBase, average: bool = False, filters: dict = None
    ) -> Tuple[str, float]:
        """
        Given an architecture prediction model, use it to predict the architecture
        of the model associated with this model manager.
        average: if true, will average the features from all of the profiles on this
        model and then pass the features to the architecture prediction model.
        filters: a dict and each argument in the dict must match
            the argument from the config file associated with a profile.
            to get a profile by name, can specify {"profile_number": "2181935"}
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
    def __init__(
        self,
        architecture: str,
        dataset: str,
        model_name: str,
        load: str = None,
        gpu: int = -1,
        data_subset_percent: float = 0.5,
        pretrained: bool = False,
        save_model: bool = True,
    ):
        """
        Models files are stored in a folder
        ./models/{model_architecture}/{self.name}_{date_time}/

        This includes the model file, a csv documenting training, and a
        config file.

        Args:
            architecture (str): the exact string representation of
                the model architecture. See get_model.py.
            dataset (str): the name of the dataset all lowercase.
            model_name (str): The name of the model, can be anything except
                don't use underscores.
            load (str, optional): If provided, should be the absolute path to
                the model folder, {cwd}/models/{model_architecture}/{self.name}{date_time}.
                This will load the model stored there.
            data_subset_percent (float, optional): If provided, should be the
                fraction of the dataset to use.  This will be generated determinisitcally.
                Uses torch.utils.data.random_split (see datasets.py)
            idx (int): the index into the subset of the dataset.  0
                for victim model and 1 for surrogate.
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
    def load(model_path: Path, gpu: int = -1):
        """Create a ModelManager Object from a path to a model file."""
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

    def generateFolder(self, load: Path, architecture: str, model_name: str) -> str:
        """
        Generates the model folder as ./models/model_architecture/{self.name}_{date_time}/
        """
        if load:
            return Path(load).parent
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_folder = Path.cwd() / "models" / architecture / f"{model_name}_{time}"
        return model_folder

    @staticmethod
    def getModelPaths(
        prefix: str = None, architectures: List[str] = None
    ) -> List[Path]:
        """
        Return a list of paths to all victim models
        in directory "./<prefix>".  This directory must be organized
        by model architecture folders whose subfolders are victim model
        folders and contain a model stored in 'checkpoint.pt'.
        Default prefix is ./models/
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
        """Returns a list of paths to surrogate models of this victim model."""
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

        dataset_name: name of the dataset from which to generate the transfer set (should
            not be the dataset on which the victim model was trained)
        transfer_size: the size of the transfer set
        sample_avg: this is the number of samples to take per class before averaging,
            higher number means better entropy estimation.  Only used if random_policy=0
        random_policy: if true, generates the transfer set randomly. If false, uses the
            adaptive method.
        entropy: if true, uses entropy as a class influence measure, else uses confidence
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
    FOLDER_NAME = "quantize"
    MODEL_FILENAME = "quantized.pt"

    def __init__(
        self,
        victim_model_path: Path,
        backend: str = "fbgemm",
        load_path: Path = None,
        gpu: int = -1,
        save_model: bool = True,
    ) -> None:
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
        self.prepare_for_quantization()
        # calibrate model
        dl_iter = tqdm(self.victim_manager.dataset.train_acc_dl)
        dl_iter.set_description("Calibrating model for quantization")
        for i, (x, y) in enumerate(dl_iter):
            self.model(x)
        torch.quantization.convert(self.model, inplace=True)

    def loadQuantizedModel(self, path: Path):
        self.prepare_for_quantization()
        torch.quantization.convert(self.model, inplace=True)
        self.loadModel(path)

    @staticmethod
    def load(model_path: Path, gpu: int = -1):
        """model_path is a path to the quantized model checkpoint"""
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
    FOLDER_NAME = "prune"
    MODEL_FILENAME = "pruned.pt"

    def __init__(
        self,
        victim_model_path: Path,
        ratio: float = 0.5,
        finetune_epochs: int = 20,
        gpu: int = -1,
        load_path: Path = None,
        save_model: bool = True,
        debug: int = None,
    ) -> None:
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
        # modifies self.model (through self.pruned_params)
        prune.global_unstructured(
            self.pruned_modules,
            pruning_method=prune.L1Unstructured,
            amount=self.ratio,
        )
        self.updateConfigSparsity()

    def paramsToPrune(
        self, min_dims=None, conv_only=False
    ) -> List[Tuple[torch.nn.Module, str]]:
        """
        min_dims (int, default None) - if provided, will only add modules
            whose weight parameter has at least min_dims dimensions.  This
            is used for structured pruning.
        conv_only (bool, default False) - if enabled, will only prune
            convolution layers.
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
    def load(model_path: Path, gpu: int = -1):
        """model_path is a path to the pruned model checkpoint"""

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
    FOLDER_NAME = "structured_prune"
    MODEL_FILENAME = "structured_pruned.pt"

    def prune(self) -> None:
        # modifies self.model (through self.pruned_params)
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
        self, min_dims=2, conv_only=True
    ) -> List[Tuple[torch.nn.Module, str]]:
        # override superclass implementation and change default args
        return super().paramsToPrune(min_dims, conv_only)

    @staticmethod
    def load(model_path: Path, gpu: int = -1):
        """model_path is a path to the pruned model checkpoint"""

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
    Constructs the surrogate model with a paired victim model,
    trains using from the labels from victim model.
    """

    FOLDER_NAME = "surrogate"

    def __init__(
        self,
        victim_model_path: Path,
        architecture: str,
        arch_conf: float,
        arch_pred_model_name: str,
        pretrained: bool = False,
        load_path: Path = None,
        gpu: int = -1,
        save_model: bool = True,
        data_idx: int = 1,
    ):
        """
        If load_path is not none, it should be a path to model.
        See SurrogateModelManager.load().

        Note - the self.config only accounts for keeping track of one history of
        victim model architecture prediction.
        Assumes victim model has been profiled/predicted.
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
    ):
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
    def load(model_path: str, gpu: int = -1):
        """
        model_path is a path to a surrogate models checkpoint,
        they are stored under {victim_model_path}/surrogate_{time}/checkpoint.pt
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
    def loadVictimConfig(path: Path) -> dict:
        """
        Given a path to a surrogate model manager folder,
        load and return the associated victim model's config.
        """
        return VictimModelManager.loadConfig(path.parent)

    def collectEpochMetrics(
        self,
        epoch_num: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        debug: int = None,
        run_attack: bool = True,
    ):
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
        debug: int = None,
    ) -> tuple[int]:
        """
        Run a single epoch.
        Uses loss between vitim model predictions and surrogate model predictions.
        validation is done on victim's validation dataset.
        agreement is the percent of samples for which the victim and surrogate's
        top1 prediction is the same.
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
        norm=np.inf,
        debug: int = None,
    ) -> Dict[str, Union[float, int]]:
        """
        Run a transfer attack, generating adversarial inputs on
        surrogate model and applying them to the victim model.
        Code adapted from cleverhans tutorial
        https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/torch/cifar10_tutorial.py
        dataset name is the name of the class attribute of the dataset,
        for example "train_dl" corresponds to dataset.train_dl,
        see datasets.Dataset class
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

    def loadVictim(self, victim_model_path: str, gpu: int):
        victim_folder = Path(victim_model_path).parent
        if victim_folder.name == PruneModelManager.FOLDER_NAME:
            return PruneModelManager.load(model_path=victim_model_path, gpu=gpu)
        if victim_folder.name == QuantizedModelManager.FOLDER_NAME:
            return QuantizedModelManager.load(model_path=victim_model_path, gpu=gpu)
        return VictimModelManager.load(model_path=victim_model_path, gpu=gpu)


def getVictimSurrogateModels(
    architectures: List[str] = None,
    victim_args: dict = {},
    surrogate_args: dict = {},
) -> Dict[Path, List[Path]]:
    """
    Given args for victim and surrogate models, return a
    dictionary of {victim_model_path: [paths of surrogate models associated
    with this victim model]}.

    Only the victim and surrogate models whose args match those
    provided will be returned.
    """

    def validManager(
        victim_path: Path,
    ) -> Dict[VictimModelManager, List[SurrogateModelManager]]:
        """
        Given a path to a victim model manger object, determine if
        its configuration matches the provided args and if it has a surrogate
        model that matches the provided args.

        Returns [(path to victim, path to surrogate)] if config matches
        and [] if not.
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
    strategies: dict,
    architectures: List[str],
    latest_file: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    architectures is a list of DNN architecture strings, like config.MODELS

    The strategies input is keyed by an arbitrary name given to a surrogate model
    training strategy.  The value associated with that key is dict of arguments
    to be matched with the surrogate model's config. Notable elements are:
        knockoff_transfer_set - will be None for pure knowledge distillation
            training, otherwise is a dict specifying the arguments: dataset,
            transfer_size, sample_avg, random_policy, and entropy.
        pretrained - boolean

    an example of the strategies input would be:
    {
        "knowledge_dist" : {
            "pretrained": False,
            "knockoff_transfer_set": None,
        },
        "knowledge_dist_pretrained" : {
            "pretrained": True,
            "knockoff_transfer_set": None,
        },
        "knockoff1" : {
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

    The return value of this function is a dict with the same keys as
    the strategies input.  The value for a key is another dict keyed by model
    architecture (only including the provided architectures).  The values are a path to
    a surrogate model which matches the args to the surrogate model training
    strategy. An error is raised if there is not a valid surrogate model for any
    of the provided architectures.

    For example, if the input is the strategies example as above, and the
    architectures input is ["resnet18", "googlenet"] then the output would be
    {
        "knowledge_dist" : {
            "resnet18": <path to resnet18 surrogate model satisfying knowledge_dist>,
            "googlenet": <path to googlenet surrogate model satisfying knowledge_dist>,
        },
        "knowledge_dist_pretrained" : {
            "resnet18": <path to resnet18 surrogate model satisfying knowledge_dist_pretrained>,
            "googlenet": <path to googlenet surrogate model satisfying knowledge_dist_pretrained>,
        },
        "knockoff1" : {
            "resnet18": <path to resnet18 surrogate model satisfying knockoff1>,
            "googlenet": <path to googlenet surrogate model satisfying knockoff1>,
        },
    }
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


if __name__ == "__main__":
    sys.exit(0)
