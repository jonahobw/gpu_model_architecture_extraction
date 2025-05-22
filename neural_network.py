"""
Implements a fully connected N-layer neural network.
Capability for training with decreasing learning rate.
Used for predicting the model architecture from gpu profiles
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch import nn

from model_metrics import correct


class Net(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer_factor=None, layers=None):
        super().__init__()
        if hidden_layer_factor is None:
            hidden_layer_factor = 0.5
        if layers is None:
            layers = 3
        self.construct_architecture(
            input_size, hidden_layer_factor, num_classes, layers
        )
        self.layer_count = layers
        self.x_tr = None
        self.x_test = None
        self.y_tr = None
        self.y_test = None
        self.accuracy = None
        self.scaler = None
        self.device = device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def construct_architecture(
        self, input_size, hidden_layer_factor, num_classes, layers
    ):
        layer_count = 0
        hidden_layer_size = int(input_size * hidden_layer_factor)
        for i in range(layers):
            if i == 0:
                layer = nn.Linear(input_size, hidden_layer_size)
            elif i == layers - 1:
                layer = nn.Linear(hidden_layer_size, num_classes)
            else:
                layer = nn.Linear(hidden_layer_size, hidden_layer_size)
            setattr(self, f"layer_{layer_count}", layer)
            layer_count += 1

    def get_layer(self, number):
        return getattr(self, f"layer_{number}")

    def forward(self, x):
        for i in range(self.layer_count - 1):
            x = F.relu(self.get_layer(i)(x))
        # return torch.sigmoid(self.get_layer(self.layer_count-1)(x))
        return self.get_layer(self.layer_count - 1)(x)

    def get_preds(self, x, grad=False, normalize=True):
        if normalize:
            x = self.normalize(x)
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        with torch.set_grad_enabled(grad):
            output = self(x)
        output = torch.nn.functional.softmax(output, dim=1)
        return torch.squeeze(output)

    def normalize(self, x, fit=False):
        """
        Uses standard scaling (x-u)/s to scale the data.  If fit is true, sets the scaler object as
        an instance variable so that future data can be scaled with the same mean and std.

        :param x: the data to normalize
        :param fit: whether or not to fit the scaler with this data's mean and std.  Should be true for
                the training data and false for the test data.
        :return: normalized data
        """
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        assert isinstance(x, np.ndarray)
        x = torch.from_numpy(x)

        if fit:
            # set the scaler
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(x)

        # fit is false
        if not self.scaler:
            raise ValueError(
                "calling normalize with fit = False when there is no scaler set."
            )

        return self.scaler.transform(x)

    def train_(self, x_tr, x_test, y_tr, y_test, epochs=100, lr=0.1, verbose=True):
        # format data
        # X_train = torch.from_numpy(x_tr.to_numpy()).float()
        # y_train = torch.squeeze(torch.from_numpy(y_tr.to_numpy()).float())
        # X_test = torch.from_numpy(x_test.to_numpy()).float()
        # y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

        x_tr = torch.tensor(x_tr, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_tr = torch.tensor(y_tr, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        since = time.time()
        test_loss_history = []
        test_acc_history = []
        training_loss_history = []
        training_acc_history = []

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, cooldown=3
        )
        # optimizer = optim.Adam(model.parameters(), lr=lr)

        x_tr = x_tr.to(self.device)
        y_tr = y_tr.to(self.device)
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        self.to(self.device)
        criterion = criterion.to(self.device)

        for epoch in range(epochs):
            # if (epoch % 25 == 0 and epoch != 0):
            #     lr = lr / 10
            with torch.set_grad_enabled(True):
                y_pred = self(x_tr)
                y_pred = torch.squeeze(y_pred)
                train_loss = criterion(y_pred, y_tr)
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            training_loss_history.append(train_loss)
            train_acc = correct(y_pred, y_tr, (1, 3))
            train1_acc = train_acc[0] / len(y_tr)
            train3_acc = train_acc[1] / len(y_tr)
            training_acc_history.append(train_acc)

            with torch.no_grad():
                y_test_pred = self(x_test)
                y_test_pred = torch.squeeze(y_test_pred)
                test_loss = criterion(y_test_pred, y_test)
            lr_scheduler.step(train_loss)
            actual_lr = optimizer.param_groups[0]["lr"]
            test_loss_history.append(test_loss)
            test_acc = correct(y_test_pred, y_test, (1, 3))
            test1_acc = test_acc[0] / len(y_test)
            test3_acc = test_acc[1] / len(y_test)
            test_acc_history.append(test_acc)
            if verbose:
                print(
                    "epoch {}\nTrain set - loss: {}, accuracy1: {}, accuracy3: {}\n"
                    "Test  set - loss: {}, accuracy1: {}, accuracy3: {}\n"
                    "learning rate: {}".format(
                        str(epoch),
                        str(train_loss),
                        str(train1_acc),
                        str(train3_acc),
                        str(test_loss),
                        str(test1_acc),
                        str(test3_acc),
                        str(actual_lr),
                    )
                )
        if verbose:
            time_elapsed = time.time() - since
            print(
                "Training complete in {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )

        # self.x_tr = x_tr
        # self.x_test = x_test
        # self.y_tr = y_tr
        # self.y_test = y_test

        return (
            training_acc_history,
            test_acc_history,
            training_loss_history,
            test_loss_history,
        )

    def train_test_accuracy(self):
        y_tr_pred = self.get_preds(self.x_tr)
        train_acc = correct(y_tr_pred, self.y_tr)
        y_test_pred = self.get_preds(self.y_test)
        test_acc = correct(y_test_pred, self.y_test)
        return train_acc, test_acc
