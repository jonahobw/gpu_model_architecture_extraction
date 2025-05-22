import sys
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm

# setting path
sys.path.append("../edge_profile")

from datasets import Dataset
from get_model import get_model, getModelParams
from model_metrics import accuracy, correct
from online import OnlineStats

# parameters
arch = "squeezenet1_0"
deterministic_train = False
gpu = 1
optim_sgd = False
data_subset_percent = 0.5
batch_size = 128
pretrained = False
nesterov = True
epochs = 300
lr = 0.0001
plateau_lr = True
patience = 10
gamma = 0.75
resize = None
normalize = None
# normalize = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
# normalize = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# build necessary objects
def getDevice(gpu):
    device = torch.device("cpu")
    if gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    return device


def getDataset(data_subset_percent, arch, batch_size, normalize, resize):
    if resize is None:
        resize = getModelParams(arch).get("input_size", None)
    return Dataset(
        "cifar10",
        data_subset_percent=data_subset_percent,
        idx=0,
        batch_size=batch_size,
        resize=resize,
        normalize=normalize,
    )


def getModel(arch, pretrained, dataset):
    return get_model(
        arch, pretrained=pretrained, kwargs={"num_classes": dataset.num_classes}
    )


def runEpoch(
    train: bool,
    deterministic_train: bool,
    epoch: int,
    epochs: int,
    optim: torch.optim.Optimizer,
    loss_fn: Callable,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    debug: int = None,
) -> tuple[int]:
    """Run a single epoch."""

    model.eval()
    prefix = "val"
    dl = dataset.val_dl
    if train:
        model.train()
        prefix = "train"
        dl = dataset.train_dl
        if deterministic_train:
            dl = dataset.train_acc_dl

    total_loss = OnlineStats()
    acc1 = OnlineStats()
    acc5 = OnlineStats()
    step_size = OnlineStats()
    step_size.add(optim.param_groups[0]["lr"])

    epoch_iter = tqdm(dl)
    epoch_iter.set_description(
        f"{prefix.capitalize()} Epoch {epoch if train else '1'}/{epochs if train else '1'}"
    )

    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(epoch_iter, start=1):
            if debug and i > debug:
                break
            x, y = x.to(device), y.to(device)
            yhat = model(x)
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
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(loss)
        else:
            lr_scheduler.step()
        # get actual train accuracy/loss after weights update
        # top1, top5, loss = accuracy(
        #    model=model,
        #    dataloader=dataset.train_acc_dl,
        #    loss_func=loss_fn,
        #    topk=(1, 5),
        # )

    return loss, top1, top5


if __name__ == "__main__":

    device = getDevice(gpu)
    dataset = getDataset(data_subset_percent, arch, batch_size, normalize, resize)
    model = getModel(arch, pretrained, dataset)
    model.to(device)
    model.train()

    # training setup
    optim = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, nesterov=nesterov, weight_decay=1e-4
    )
    if not optim_sgd:
        optim = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8
        )

    loss_func = torch.nn.CrossEntropyLoss()

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=patience, factor=gamma
    )
    if not plateau_lr:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=patience, gamma=gamma
        )
    # training
    for epoch in range(1, epochs + 1):
        runEpoch(
            train=True,
            deterministic_train=deterministic_train,
            epoch=epoch,
            epochs=epochs,
            optim=optim,
            loss_fn=loss_func,
            lr_scheduler=lr_scheduler,
        )

#        runEpoch(
#           train=False,
#          epoch=epoch,
#         epochs=epochs,
#        optim=optim,
#       loss_fn=loss_func,
#      lr_scheduler=lr_scheduler,
# )
