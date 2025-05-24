"""Dataset download and preparation utilities.

This scipt provides functionality for downloading and preparing common image datasets
including MNIST, CIFAR10, CIFAR100, ImageNet, and TinyImageNet-200. It supports:
- Automatic download of datasets to a local directory
- Progress tracking during downloads
- Dataset-specific formatting and organization
- Support for both torchvision and custom datasets

Dependencies:
    - requests: For downloading datasets
    - torchvision: For dataset implementations
    - tqdm: For progress bars
    - zipfile: For handling compressed datasets

Usage: python download_dataset.py -name <dataset_name>
"""

import io
import shutil
import zipfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Type, Union

import requests
from torchvision import datasets
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# Dictionary mapping dataset names to their torchvision dataset classes
torchvision_datasets: Dict[str, Type[ImageFolder]] = {
    "MNIST": datasets.MNIST,
    "CIFAR10": datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100,
    "ImageNet": datasets.ImageNet,
}

# List of all supported datasets
supported_datasets: List[str] = ["tiny-imagenet-200"] + list(torchvision_datasets.keys())

# Default download path
download_path: Path = Path.cwd() / "datasets"

# Create download directory if it doesn't exist
if not download_path.exists():
    download_path.mkdir(parents=True, exist_ok=True)


def download(url: str, fname: Union[str, Path], chunk_size: int = 1024) -> requests.Response:
    """Download a file from a URL with progress tracking.
    
    Args:
        url: URL to download from
        fname: Path to save the file to
        chunk_size: Size of chunks to download at a time
        
    Returns:
        Response object from the download request
        
    Raises:
        AssertionError: If download fails (status code != 200)
    """
    if not isinstance(fname, str):
        fname = str(fname)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
    assert (
        resp.status_code == 200
    ), f"Download failed with status code {resp.status_code}."
    return resp


def downloadDataset(dataset_name: str) -> None:
    """Download and prepare a dataset.
    
    This function handles both torchvision datasets and custom datasets like TinyImageNet-200.
    For torchvision datasets, it uses the built-in download functionality.
    For TinyImageNet-200, it downloads the zip file and formats the directory structure.
    
    Args:
        dataset_name: Name of the dataset to download
        
    Raises:
        ValueError: If dataset_name is not in supported_datasets
    """
    if dataset_name not in supported_datasets:
        raise ValueError(
            f"{dataset_name} not supported.  Supported datasets are {supported_datasets}."
        )
    dataset_path = download_path / dataset_name
    print(f"Downloading {dataset_name} ...")
    
    # Handle torchvision datasets
    if dataset_name in torchvision_datasets:
        torchvision_datasets[dataset_name](root=dataset_path, download=True)
    
    # Handle TinyImageNet-200 dataset
    if dataset_name == "tiny-imagenet-200":
        file = download_path / "tiny-imagenet-200.zip"
        if not file.exists():
            download(
                "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
                fname=download_path / "tiny-imagenet-200.zip",
            )
        z = zipfile.ZipFile(file)
        z.extractall(dataset_path.parent)
        Path(file).unlink()

        # Format training data directory structure
        print("Formatting training data ...")
        for class_folder in tqdm((dataset_path / "train").glob("*")):
            if not class_folder.is_dir():
                class_folder.unlink()
            img_folder = class_folder / "images"
            if img_folder.exists():
                for img in img_folder.glob("*JPEG"):
                    img.rename(dataset_path / "train" / class_folder.name / img.name)
                shutil.rmtree(img_folder)

        # Format validation data directory structure
        val_folder = dataset_path / "val"
        annotation_file = val_folder / "val_annotations.txt"
        file_to_label = {}
        with open(annotation_file, "r") as f:
            annotations = f.readlines()
        for line in annotations:
            if line.find("JPEG") >= 0:
                words = line.split("\t")
                file_to_label[words[0]] = words[1]

        # Create subfolders for validation images based on label
        print("Formatting validation data ...")
        for img, folder in tqdm(file_to_label.items()):
            img_file = val_folder / "images" / img
            newpath = val_folder / folder
            newpath.mkdir(exist_ok=True)
            if img_file.exists():
                img_file.rename(newpath / img)
        shutil.rmtree(val_folder / "images")

    print(f"Completed.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Download and prepare a dataset")
    parser.add_argument(
        "-name",
        required=True,
        type=str,
        help=f"The name of the dataset, currently supports {supported_datasets}",
    )
    args = parser.parse_args()

    downloadDataset(args.name)
