"""Dataset loading and preprocessing utilities for deep learning models.

Adapted from https://github.com/jonahobw/shrinkbench/blob/master/datasets/datasets.py

This module provides functionality for loading and preprocessing common image datasets
including MNIST, CIFAR10, CIFAR100, ImageNet, and TinyImageNet. It supports:
- Dataset loading with proper preprocessing and normalization
- Train/validation splits
- Data augmentation
- Lazy loading of datasets
- Dataset partitioning
- Class balance analysis

Dependencies:
    - torch: For tensor operations and data loading
    - torchvision: For dataset implementations and transforms
    - numpy: For numerical operations

Example Usage:
    ```python
    from datasets import Dataset
    
    # Create a dataset with lazy loading
    dataset = Dataset("cifar10", batch_size=128)
    
    # Get training data loader
    train_loader = dataset.train_dl
    
    # Get validation data loader
    val_loader = dataset.val_dl
    ```
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable

from torch import Generator
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset


def nameToDataset() -> Dict[str, Callable]:
    """Get mapping of dataset names to their loader functions.
    
    Returns:
        Dictionary mapping dataset names to their respective dataset classes
    """
    return {
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
        "ImageNet": datasets.ImageNet,
        "tiny-imagenet-200": TinyImageNet200,
    }


class TinyImageNet200(datasets.ImageFolder):
    """Dataset class for TinyImageNet-200.
    
    This class extends ImageFolder to handle the TinyImageNet-200 dataset,
    Adapted from https://github.com/tribhuvanesh/knockoffnets
    
    Args:
        root: Root directory of the dataset
        train: If True, loads training data, else validation data
        transform: Optional transform to apply to images
    """

    def __init__(self, root: Union[str, Path], train: bool = True, transform: Optional[Callable] = None):
        root = Path(root) / "train" if train else "val"
        super().__init__(root=root, transform=transform)
        self.root = root
        self._load_meta()

    def _load_meta(self) -> None:
        """Replace class names (synsets) with more descriptive labels.
        
        Loads the words.txt file to map synset IDs to human-readable class names.
        """
        synset_to_desc = dict()
        fpath = Path(self.root.parent) / "words.txt"
        with open(fpath, "r") as rf:
            for line in rf:
                synset, desc = line.strip().split(maxsplit=1)
                synset_to_desc[synset] = desc

        for i in range(len(self.classes)):
            self.classes[i] = synset_to_desc[self.classes[i]]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}


def dataset_path(dataset: str, path: Optional[str] = None) -> Path:
    """Get the path to a specified dataset.
    
    Args:
        dataset: Name of the dataset (MNIST, CIFAR10, CIFAR100, ImageNet)
        path: Semicolon-separated list of paths to look for dataset folders
        
    Returns:
        Path to the dataset directory
        
    Raises:
        ValueError: If no path is provided and DATAPATH is not set
        LookupError: If the dataset cannot be found in any of the provided paths
    """
    p = Path.cwd() / "datasets" / dataset

    if p.exists():
        return p
    print(f"Path does not exist:\n{p}\n")

    paths = [Path(p) for p in path.split(":")]

    for p in paths:
        p = (p / dataset).resolve()
        if p.exists():
            return p
    raise LookupError(f"Could not find {dataset}")


def dataset_builder(
    dataset: str,
    train: bool = True,
    normalize: Optional[transforms.Normalize] = None,
    preproc: Optional[List[transforms.Transform]] = None,
    path: Optional[str] = None,
    resize: Optional[int] = None
) -> VisionDataset:
    """Build a dataset with proper preprocessing.
    
    Args:
        dataset: Name of the dataset
        train: Whether to return training or validation set
        normalize: Transform to normalize data channel-wise
        preproc: List of preprocessing operations
        path: Semicolon-separated list of paths to look for dataset folders
        resize: Optional size to resize images to
        
    Returns:
        Dataset object with transforms and normalization
    """
    if preproc is not None:
        preproc += [transforms.ToTensor()]
        if resize is not None:
            preproc += [transforms.Resize(resize)]
        if normalize is not None:
            preproc += [normalize]
        preproc = transforms.Compose(preproc)

    kwargs = {"transform": preproc}
    if dataset == "ImageNet":
        kwargs["split"] = "train" if train else "val"
    else:
        kwargs["train"] = train

    path = dataset_path(dataset, path)

    return nameToDataset()[dataset](path, **kwargs)


def MNIST(
    train: bool = True,
    path: Optional[str] = None,
    resize: Optional[int] = None,
    normalize: Optional[Tuple[List[float], List[float]]] = None,
) -> VisionDataset:
    """Create MNIST dataset with preprocessing.
    
    Args:
        train: Whether to return training or validation set
        path: Path to dataset directory
        resize: Optional size to resize images to
        normalize: Optional (mean, std) for normalization
        
    Returns:
        MNIST dataset with specified preprocessing
    """
    mean, std = 0.1307, 0.3081
    normalize = transforms.Normalize(mean=(mean,), std=(std,))
    dataset = dataset_builder("MNIST", train, normalize, [], path, resize=resize)
    dataset.shape = (1, 28, 28)
    return dataset


def CIFAR10(
    train: bool = True,
    path: Optional[str] = None,
    deterministic: bool = False,
    resize: Optional[int] = None,
    normalize: Optional[Tuple[List[float], List[float]]] = None,
) -> VisionDataset:
    """Create CIFAR10 dataset with preprocessing.
    
    Args:
        train: Whether to return training or validation set
        path: Path to dataset directory
        deterministic: If True, disable data augmentation
        resize: Optional size to resize images to
        normalize: Optional (mean, std) for normalization
        
    Returns:
        CIFAR10 dataset with specified preprocessing
    """
    mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.262]
    if normalize is not None:
        mean, std = normalize
    normalize = transforms.Normalize(mean=mean, std=std)
    if train and not deterministic:
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder("CIFAR10", train, normalize, preproc, path, resize=resize)
    dataset.shape = (3, 32, 32)
    return dataset


def CIFAR100(
    train: bool = True,
    path: Optional[str] = None,
    resize: Optional[int] = None,
    normalize: Optional[Tuple[List[float], List[float]]] = None,
    deterministic: bool = False,
) -> VisionDataset:
    """Create CIFAR100 dataset with preprocessing.
    
    Args:
        train: Whether to return training or validation set
        path: Path to dataset directory
        resize: Optional size to resize images to
        normalize: Optional (mean, std) for normalization
        deterministic: If True, disable data augmentation
        
    Returns:
        CIFAR100 dataset with specified preprocessing
    """
    mean, std = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
    if normalize is not None:
        mean, std = normalize
    normalize = transforms.Normalize(mean=mean, std=std)
    if train and not deterministic:
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder(
        "CIFAR100", train, normalize, preproc, path, resize=resize
    )
    dataset.shape = (3, 32, 32)
    return dataset


def TinyImageNet(
    train: bool = True,
    path: Optional[str] = None,
    resize: Optional[int] = None,
    normalize: Optional[Tuple[List[float], List[float]]] = None,
    deterministic: bool = False,
) -> VisionDataset:
    """Create TinyImageNet dataset with preprocessing.
    
    Args:
        train: Whether to return training or validation set
        path: Path to dataset directory
        resize: Optional size to resize images to
        normalize: Optional (mean, std) for normalization
        deterministic: If True, disable data augmentation
        
    Returns:
        TinyImageNet dataset with specified preprocessing
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if normalize is not None:
        mean, std = normalize
    normalize = transforms.Normalize(mean=mean, std=std)
    if train and not deterministic:
        preproc = [transforms.RandomCrop(56, 4), transforms.RandomHorizontalFlip()]
    else:
        preproc = []
    dataset = dataset_builder(
        "tiny-imagenet-200", train, normalize, preproc, path, resize=resize
    )
    dataset.shape = (3, 64, 64)
    return dataset


def ImageNet(
    train: bool = True,
    path: Optional[str] = None,
    resize: Optional[int] = None,
    normalize: Optional[Tuple[List[float], List[float]]] = None,
    deterministic: bool = False,
) -> VisionDataset:
    """Create ImageNet dataset with preprocessing.
    
    Args:
        train: Whether to return training or validation set
        path: Path to dataset directory
        resize: Optional size to resize images to
        normalize: Optional (mean, std) for normalization
        deterministic: If True, disable data augmentation
        
    Returns:
        ImageNet dataset with specified preprocessing
    """
    import warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if normalize is not None:
        mean, std = normalize
    normalize = transforms.Normalize(mean=mean, std=std)
    if train and not deterministic:
        preproc = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder(
        "ImageNet", train, normalize, preproc, path, resize=resize
    )
    dataset.shape = (3, 224, 224)
    return dataset


class Dataset:
    """Dataset class for managing data loading and preprocessing.
    
    This class provides a unified interface for loading and preprocessing datasets,
    with support for lazy loading, data subsetting, and custom preprocessing.
    
    Attributes:
        name_mapping: Dictionary mapping dataset names to their loader functions
        num_classes_map: Dictionary mapping dataset names to their number of classes
    """

    name_mapping = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "imagenet": ImageNet,
        "tiny-imagenet-200": TinyImageNet,
    }
    num_classes_map = {
        "mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "imagenet": 1000,
        "tiny-imagenet-200": 200,
    }

    def __init__(
        self,
        dataset: str,
        batch_size: int = 128,
        workers: int = 8,
        data_subset_percent: Optional[float] = None,
        seed: int = 42,
        idx: int = 0,
        resize: Optional[int] = None,
        normalize: Optional[Tuple[List[float], List[float]]] = None,
        lazy_load: bool = True,
        indices: Optional[Tuple[List[int], List[int]]] = None,
        deterministic: bool = False,
    ) -> None:
        """Initialize Dataset.
        
        Args:
            dataset: Name of the dataset to load
            batch_size: Batch size for data loaders
            workers: Number of worker processes for data loading
            data_subset_percent: Percentage of data to use (splits into two portions)
            idx: Which portion of the split to use (0 or 1)
            seed: Random seed for reproducibility
            resize: Optional size to resize images to
            normalize: Optional (mean, std) for normalization
            lazy_load: If True, datasets are loaded only when accessed
            indices: Tuple of (train_indices, val_indices) for custom splits
            deterministic: If True, disable data augmentation
        """
        self.name = dataset.lower()
        self.num_classes = self.num_classes_map[self.name]
        self.batch_size = batch_size
        self.workers = workers
        self.data_subset_percent = data_subset_percent
        self.seed = seed
        self.idx = idx
        self.resize = resize
        self.normalize = normalize
        self.lazy_load = lazy_load
        self.indices = indices
        self.deterministic = deterministic

        # Lazy loading attributes
        self._train_data = None
        self._train_dl = None
        self._val_data = None
        self._val_dl = None
        self._train_acc_data = None
        self._train_acc_dl = None

        if not lazy_load:
            assert self.train_data is not None
            assert self.train_dl is not None
            assert self.val_data is not None
            assert self.val_dl is not None
            assert self.train_acc_data is not None
            assert self.train_acc_dl is not None

        self.config = {
            "dataset": self.name,
            "batch_size": batch_size,
            "workers": workers,
            "data_subset_percent": data_subset_percent,
            "seed": seed,
            "idx": idx,
            "resize": resize,
            "normalize": normalize,
            "lazy_load": lazy_load,
            "indices": indices,
        }

    @property
    def train_data(self) -> Union[VisionDataset, Subset]:
        """Get training dataset.
        
        Returns:
            Training dataset with specified preprocessing
        """
        if self._train_data is None:
            # this executes the first time the property is used
            self._train_data = self.name_mapping[self.name](
                resize=self.resize,
                normalize=self.normalize,
                deterministic=self.deterministic,
            )
            if self.data_subset_percent is not None:
                first_amount = int(len(self._train_data) * self.data_subset_percent)
                second_amount = len(self._train_data) - first_amount
                self._train_data = random_split(
                    self._train_data,
                    [first_amount, second_amount],
                    generator=Generator().manual_seed(self.seed),
                )[self.idx]
            if self.indices is not None:
                self._train_data = Subset(self._train_data, self.indices[0])
        return self._train_data

    @property
    def train_dl(self) -> DataLoader:
        """Get training data loader.
        
        Returns:
            DataLoader for training data
        """
        if self._train_dl is None:
            # this executes the first time the property is used
            self._train_dl = DataLoader(
                self.train_data,
                shuffle=True,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.workers,
            )
        return self._train_dl

    @property
    def val_data(self) -> Union[VisionDataset, Subset]:
        """Get validation dataset.
        
        Returns:
            Validation dataset with specified preprocessing
        """
        if self._val_data is None:
            # this executes the first time self.train_data is used
            self._val_data = self.name_mapping[self.name](
                train=False,
                resize=self.resize,
                normalize=self.normalize,
                deterministic=self.deterministic,
            )
            if self.data_subset_percent is not None:
                first_amount = int(len(self._val_data) * self.data_subset_percent)
                second_amount = len(self._val_data) - first_amount
                self._val_data = random_split(
                    self._val_data,
                    [first_amount, second_amount],
                    generator=Generator().manual_seed(self.seed),
                )[self.idx]
            if self.indices is not None:
                self._val_data = Subset(self._val_data, self.indices[1])
        return self._val_data

    @property
    def val_dl(self) -> DataLoader:
        """Get validation data loader.
        
        Returns:
            DataLoader for validation data
        """
        if self._val_dl is None:
            # this executes the first time the property is used
            self._val_dl = DataLoader(
                self.val_data,
                shuffle=False,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.workers,
            )
        return self._val_dl

    @property
    def train_acc_data(self) -> Union[VisionDataset, Subset]:
        """Get training dataset for accuracy computation.
        
        Returns:
            Training dataset with deterministic preprocessing
        """
        if self._train_acc_data is None:
            # this executes the first time the property is used
            self._train_acc_data = self.name_mapping[self.name](
                deterministic=True, resize=self.resize, normalize=self.normalize
            )
            if self.data_subset_percent is not None:
                first_amount = int(len(self._train_acc_data) * self.data_subset_percent)
                second_amount = len(self._train_acc_data) - first_amount
                self._train_acc_data = random_split(
                    self._train_acc_data,
                    [first_amount, second_amount],
                    generator=Generator().manual_seed(self.seed),
                )[self.idx]
            if self.indices is not None:
                self._train_acc_data = Subset(self._train_acc_data, self.indices[0])
        return self._train_acc_data

    @property
    def train_acc_dl(self) -> DataLoader:
        """Get training data loader for accuracy computation.
        
        Returns:
            DataLoader for training data with deterministic preprocessing
        """
        if self._train_acc_dl is None:
            # this executes the first time the property is used
            self._train_acc_dl = DataLoader(
                self.train_acc_data,
                shuffle=False,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.workers,
            )
        return self._train_acc_dl

    def classBalance(
        self, dataset: Union[VisionDataset, Subset], show: bool = True
    ) -> Dict[int, int]:
        """Compute class balance statistics for a dataset.
        
        Args:
            dataset: Dataset to analyze
            show: If True, print the class distribution
            
        Returns:
            Dictionary mapping class indices to their counts
        """
        if isinstance(dataset, Subset):
            result = dict(Counter(dataset.dataset.targets))
        else:
            result = dict(Counter(dataset.targets))
        if show:
            print(json.dumps(result, indent=4))
        return result


def datasetPartition(
    dataset: str,
    batch_size: int = 128,
    workers: int = 8,
    data_subset_percent: Optional[float] = None,
    seed: int = 42,
    resize: Optional[int] = None,
) -> List[Dataset]:
    """Create two datasets from a single dataset with different splits.
    
    Args:
        dataset: Name of the dataset to load
        batch_size: Batch size for data loaders
        workers: Number of worker processes for data loading
        data_subset_percent: Percentage of data to use (splits into two portions)
        seed: Random seed for reproducibility
        resize: Optional size to resize images to
        
    Returns:
        List of two Dataset objects with different data splits
    """
    first_dataset = Dataset(
        dataset,
        batch_size=batch_size,
        workers=workers,
        data_subset_percent=data_subset_percent,
        seed=seed,
        idx=0,
        resize=resize,
    )
    second_dataset = Dataset(
        dataset,
        batch_size=batch_size,
        workers=workers,
        data_subset_percent=data_subset_percent,
        seed=seed,
        idx=1,
        resize=resize,
    )
    return [first_dataset, second_dataset]
