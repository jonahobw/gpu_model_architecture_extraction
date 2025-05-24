"""
Implements a fully connected N-layer neural network for architecture prediction.

This module provides a neural network implementation specifically designed for predicting
model architectures from GPU profiles. It supports configurable network depth and width,
with automatic learning rate adjustment during training.

Example Usage:
    ```python
    # Create a 3-layer network with input size 100 and 10 output classes
    net = Net(input_size=100, num_classes=10, hidden_layer_factor=0.5, layers=3)
    
    # Train the network
    train_acc, test_acc, train_loss, test_loss = net.train_(
        x_tr=train_data,
        x_test=test_data,
        y_tr=train_labels,
        y_test=test_labels,
        epochs=100,
        lr=0.1
    )
    
    # Make predictions
    predictions = net.get_preds(new_data)
    ```
"""

import time
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch import nn

from model_metrics import correct


class Net(nn.Module):
    """
    A fully connected neural network for architecture prediction.

    This network implements a configurable N-layer neural network with ReLU activations
    and softmax output. It includes built-in data normalization and training capabilities
    with learning rate scheduling.

    Attributes:
        layer_count (int): Number of layers in the network
        device (torch.device): Device to run the network on (CPU/GPU)
        scaler (Optional[StandardScaler]): Scaler for input normalization
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_layer_factor: Optional[float] = None,
        layers: Optional[int] = None,
    ) -> None:
        """
        Initialize the neural network.

        Args:
            input_size (int): Size of input features
            num_classes (int): Number of output classes
            hidden_layer_factor (Optional[float]): Factor to determine hidden layer size.
                Hidden layer size = input_size * hidden_layer_factor. Defaults to 0.5.
            layers (Optional[int]): Number of layers in the network. Defaults to 3.

        Note:
            The network architecture is:
            - Input layer: input_size neurons
            - Hidden layers: hidden_layer_size neurons (input_size * hidden_layer_factor)
            - Output layer: num_classes neurons
            All layers except the last use ReLU activation.
        """
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def construct_architecture(
        self, input_size: int, hidden_layer_factor: float, num_classes: int, layers: int
    ) -> None:
        """
        Construct the network architecture.

        Args:
            input_size (int): Size of input features
            hidden_layer_factor (float): Factor to determine hidden layer size
            num_classes (int): Number of output classes
            layers (int): Number of layers in the network
        """
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
        return self.get_layer(self.layer_count - 1)(x)

    def get_preds(
        self,
        x: Union[np.ndarray, pd.DataFrame],
        grad: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Get predictions for input data.

        Args:
            x (Union[np.ndarray, pd.DataFrame]): Input data to predict
            grad (bool): Whether to enable gradient computation. Defaults to False.
            normalize (bool): Whether to normalize input data. Defaults to True.

        Returns:
            torch.Tensor: Softmax probabilities of shape (batch_size, num_classes)

        Raises:
            ValueError: If normalize=True but no scaler is set
        """
        if normalize:
            x = self.normalize(x)
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        with torch.set_grad_enabled(grad):
            output = self(x)
        output = torch.nn.functional.softmax(output, dim=1)
        return torch.squeeze(output)

    def normalize(
        self, x: Union[np.ndarray, pd.DataFrame], fit: bool = False
    ) -> np.ndarray:
        """
        Normalize input data using standard scaling (x-u)/s.

        Args:
            x (Union[np.ndarray, pd.DataFrame]): Data to normalize
            fit (bool): Whether to fit the scaler with this data's statistics.
                Should be True for training data and False for test data.
                Defaults to False.

        Returns:
            np.ndarray: Normalized data

        Raises:
            ValueError: If fit=False but no scaler is set
        """
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        assert isinstance(x, np.ndarray)
        x = torch.from_numpy(x)

        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(x)

        if not self.scaler:
            raise ValueError(
                "Calling normalize with fit=False when there is no scaler set."
            )

        return self.scaler.transform(x)

    def train_(
        self,
        x_tr: Union[np.ndarray, pd.DataFrame],
        x_test: Union[np.ndarray, pd.DataFrame],
        y_tr: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.DataFrame],
        epochs: int = 100,
        lr: float = 0.1,
        verbose: bool = True,
    ) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]], List[float], List[float]
    ]:
        """
        Train the network.

        Args:
            x_tr (Union[np.ndarray, pd.DataFrame]): Training features
            x_test (Union[np.ndarray, pd.DataFrame]): Test features
            y_tr (Union[np.ndarray, pd.DataFrame]): Training labels
            y_test (Union[np.ndarray, pd.DataFrame]): Test labels
            epochs (int): Number of training epochs. Defaults to 100.
            lr (float): Initial learning rate. Defaults to 0.1.
            verbose (bool): Whether to print training progress. Defaults to True.

        Returns:
            Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[float], List[float]]:
                Tuple containing:
                - Training accuracy history (top-1, top-3)
                - Test accuracy history (top-1, top-3)
                - Training loss history
                - Test loss history

        Note:
            Uses SGD optimizer with momentum and learning rate scheduling.
            Learning rate is reduced when training loss plateaus.
        """
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

        x_tr = x_tr.to(self.device)
        y_tr = y_tr.to(self.device)
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        self.to(self.device)
        criterion = criterion.to(self.device)

        for epoch in range(epochs):
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

        return (
            training_acc_history,
            test_acc_history,
            training_loss_history,
            test_loss_history,
        )

    def train_test_accuracy(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Compute accuracy on training and test sets.

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]: Tuple containing:
                - Training accuracy (top-1, top-3)
                - Test accuracy (top-1, top-3)

        Raises:
            ValueError: If training or test data is not set
        """
        if self.x_tr is None or self.y_tr is None:
            raise ValueError("Training data not set")
        if self.x_test is None or self.y_test is None:
            raise ValueError("Test data not set")

        y_tr_pred = self.get_preds(self.x_tr)
        train_acc = correct(y_tr_pred, self.y_tr)
        y_test_pred = self.get_preds(self.y_test)
        test_acc = correct(y_test_pred, self.y_test)
        return train_acc, test_acc
