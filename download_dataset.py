"""Downloads a dataset to <current working dir>/datasets/<name>"""

import io
import shutil
import zipfile
from argparse import ArgumentParser
from pathlib import Path

import requests
from torchvision import datasets
from torchvision.datasets import ImageFolder
from tqdm import tqdm

torchvision_datasets = {
    "MNIST": datasets.MNIST,
    "CIFAR10": datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100,
    "ImageNet": datasets.ImageNet,
}

supported_datasets = ["tiny-imagenet-200"] + list(torchvision_datasets.keys())

download_path = Path.cwd() / "datasets"

if not download_path.exists():
    download_path.mkdir(parents=True, exist_ok=True)


def download(url: str, fname: str, chunk_size=1024):
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


def downloadDataset(dataset_name):
    if dataset_name not in supported_datasets:
        raise ValueError(
            f"{dataset_name} not supported.  Supported datasets are {supported_datasets}."
        )
    dataset_path = download_path / dataset_name
    print(f"Downloading {dataset_name} ...")
    if dataset_name in torchvision_datasets:
        torchvision_datasets[dataset_name](root=dataset_path, download=True)
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

        # format the image folder from /class/images/n01443537_0.JPEG
        # to get rid of images folder, resulting in /class/n01443537_0.JPEG
        print("Formatting training data ...")
        for class_folder in tqdm((dataset_path / "train").glob("*")):
            if not class_folder.is_dir():
                class_folder.unlink()
            # text_files = [x for x in class_folder.glob("*.txt")]
            # for text_file in text_files:
            #     text_file.unlink()
            img_folder = class_folder / "images"
            if img_folder.exists():
                for img in img_folder.glob("*JPEG"):
                    img.rename(dataset_path / "train" / class_folder.name / img.name)
                shutil.rmtree(img_folder)

        # format the validation folder
        val_folder = dataset_path / "val"
        annotation_file = val_folder / "val_annotations.txt"
        file_to_label = {}
        with open(annotation_file, "r") as f:
            annotations = f.readlines()
        for line in annotations:
            if line.find("JPEG") >= 0:
                words = line.split("\t")
                file_to_label[words[0]] = words[1]

        # Create subfolders (if not present) for validation images based on label,
        # and move images into the respective folders
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
    a = ArgumentParser()
    a.add_argument(
        "-name",
        required=True,
        type=str,
        help=f"The name of the dataset, currently supports {supported_datasets}",
    )
    args = a.parse_args()

    downloadDataset(args.name)
