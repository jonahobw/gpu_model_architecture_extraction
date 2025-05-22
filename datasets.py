"""
adapted from https://github.com/jonahobw/shrinkbench/blob/master/datasets/datasets.py
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
from collections import Counter

from torchvision import transforms, datasets
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, random_split, Subset
from torch import Generator


def nameToDataset():
    return {
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
        "ImageNet": datasets.ImageNet,
        "tiny-imagenet-200": TinyImageNet200,
    }


class TinyImageNet200(datasets.ImageFolder):
    """
    Dataset for TinyImageNet200.
    Adapted from https://github.com/tribhuvanesh/knockoffnets
    """

    def __init__(self, root, train=True, transform=None):

        # Initialize ImageFolder
        root = Path(root) / "train" if train else "val"
        super().__init__(root=root, transform=transform)
        self.root = root
        self._load_meta()

    def _load_meta(self):
        """Replace class names (synsets) with more descriptive labels"""
        # Load mapping
        synset_to_desc = dict()
        fpath = Path(self.root.parent) / "words.txt"
        with open(fpath, "r") as rf:
            for line in rf:
                synset, desc = line.strip().split(maxsplit=1)
                synset_to_desc[synset] = desc

        # Replace
        for i in range(len(self.classes)):
            self.classes[i] = synset_to_desc[self.classes[i]]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}


def dataset_path(dataset, path=None):
    """Get the path to a specified dataset

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet

    Keyword Arguments:
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        dataset_path -- pathlib.Path for the first match

    Raises:
        ValueError -- If no path is provided and DATAPATH is not set
        LookupError -- If the given dataset cannot be found
    """
    p = Path.cwd() / "datasets" / dataset

    if p.exists():
        return p
    print(f"Path does not exist:\n{p}\n")

    paths = [Path(p) for p in path.split(":")]

    for p in paths:
        p = (p / dataset).resolve()
        if p.exists():
            # print(f"Found {dataset} under {p}")
            return p
    raise LookupError(f"Could not find {dataset}")


def dataset_builder(
    dataset, train=True, normalize=None, preproc=None, path=None, resize: int = None
):
    """Build a torch.utils.Dataset with proper preprocessing

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet, Places365

    Keyword Arguments:
        train {bool} -- Whether to return train or validation set (default: {True})
        normalize {torchvision.Transform} -- Transform to normalize data channel wise (default: {None})
        preproc {list(torchvision.Transform)} -- List of preprocessing operations (default: {None})
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        torch.utils.data.Dataset -- Dataset object with transforms and normalization
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
    train=True,
    path=None,
    resize=None,
    normalize: Tuple[List[float], List[float]] = None,
    deterministic: bool = False,
):
    """Thin wrapper around torchvision.datasets.CIFAR10"""
    mean, std = 0.1307, 0.3081
    normalize = transforms.Normalize(mean=(mean,), std=(std,))
    dataset = dataset_builder("MNIST", train, normalize, [], path, resize=resize)
    dataset.shape = (1, 28, 28)
    return dataset


def CIFAR10(
    train=True,
    path=None,
    deterministic=False,
    resize=None,
    normalize: Tuple[List[float], List[float]] = None,
):
    """Thin wrapper around torchvision.datasets.CIFAR10"""
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
    train=True,
    path=None,
    resize=None,
    normalize: Tuple[List[float], List[float]] = None,
    deterministic: bool = False,
):
    """Thin wrapper around torchvision.datasets.CIFAR100"""
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
    train=True,
    path=None,
    resize=None,
    normalize: Tuple[List[float], List[float]] = None,
    deterministic: bool = False,
):
    """Sets up the tiny imagenet dataset."""
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
    train=True,
    path=None,
    resize=None,
    normalize: Tuple[List[float], List[float]] = None,
    deterministic: bool = False,
):
    """Thin wrapper around torchvision.datasets.ImageNet"""
    # ImageNet loading from files can produce benign EXIF errors
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
    # todo implement lazy loading of datasets

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
        batch_size=128,
        workers=8,
        data_subset_percent: float = None,
        seed: int = 42,
        idx: int = 0,
        resize: int = None,
        normalize: Tuple[List[float], List[float]] = None,
        lazy_load: bool = True,
        indices: Tuple[List[int], List[int]] = None,
        deterministic: bool = False,
    ) -> None:
        """
        data_subset_percent will divide the dataset into 2 pieces, and the first will have <data_subset_percent>% of the data.
            which portion of the data is allocated depends on <idx> which is either 0 or 1
        resize will resize the data when retrieving it.  Some models need this, most do not.  Resizing significantly slows training.
        lazy_load: if true, won't load underlying datasets until they are referenced
        indices: a tuple of 2 lists of indices to the dataset to be used as a subset.  The first list is for the training data and
            the second list is for the test data
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

        # lazy loading attributes, each is associated
        # with a property of this class with the same name but
        # without a '_' at the beginning, these are only loaded
        # when the property is referenced.
        self._train_data = None
        self._train_dl = None
        self._val_data = None
        self._val_dl = None
        self._train_acc_data = None
        self._train_acc_dl = None

        if not lazy_load:
            # to load the objects, just need to reference them
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
    def train_data(self):
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
    def train_dl(self):
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
    def val_data(self):
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
    def val_dl(self):
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
    def train_acc_data(self):
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
    def train_acc_dl(self):
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
        self, dataset: Union[VisionDataset, Subset], show=True
    ) -> Dict[int, int]:
        if isinstance(dataset, Subset):
            result = dict(Counter(dataset.dataset.targets))
        else:
            result = dict(Counter(dataset.targets))
        if show:
            print(json.dumps(result, indent=4))
        return result


def datasetPartition(
    dataset: str,
    batch_size=128,
    workers=8,
    data_subset_percent: float = None,
    seed: int = 42,
    resize: int = None,
) -> List[Dataset]:
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
