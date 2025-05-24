"""Takes cleaned profile data and runs classifiers on it to predict model architecture.

This module provides a framework for predicting GPU model architectures based on profile data.
It implements various machine learning classifiers including neural networks, logistic regression,
random forests, and more.

Dependencies:
    - numpy: For numerical operations
    - pandas: For data manipulation
    - scikit-learn: For machine learning models
    - torch: For neural network implementation (optional)

Example Usage:
    ```python
    from architecture_prediction import get_arch_pred_model
    
    # Get a logistic regression model
    model = get_arch_pred_model('lr', df=your_dataframe)
    
    # Make predictions
    prediction, confidence = model.predict(input_data)
    
    # Get top 3 predictions
    top_3 = model.topK(input_data, k=3)
    ```
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, StandardScaler)

from config import SYSTEM_SIGNALS
from data_engineering import (add_indicator_cols_to_input, all_data,
                              get_data_and_labels, remove_cols, shared_data,
                              softmax)
from neural_network import Net


class ArchPredBase(ABC):
    """Base class for architecture prediction models.
    
    This abstract base class defines the interface and common functionality
    for all architecture prediction models.
    
    Args:
        df: Input DataFrame containing profile data
        name: Name of the model
        label: Column name for the target variable (default: "model")
        verbose: Whether to print progress information
        deterministic: Whether the model should be deterministic
        train_size: Size of training set (optional)
        test_size: Size of test set (optional)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        name: str,
        label: Optional[str] = None,
        verbose: bool = True,
        deterministic: bool = True,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ) -> None:
        if label is None:
            label = "model"
        self.verbose = verbose
        self.data = df
        self.name = name
        self.label = label
        self.label_encoder = LabelEncoder()
        all_x, all_y = get_data_and_labels(self.data, shuffle=False, label=label)
        self.orig_cols = list(all_x.columns)
        all_y_labeled = self.label_encoder.fit_transform(all_y)
        x_tr, x_test, y_train, y_test = train_test_split(
            all_x,
            all_y_labeled,
            random_state=42,
            stratify=all_y_labeled,
            train_size=train_size,
            test_size=test_size,
        )
        self.x_tr = x_tr
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_classes = len(all_y.unique())
        self.input_size = len(all_x.columns)

        # overwritten by subclasses
        self.model = None
        self.deterministic = deterministic

    def preprocessInput(self, x: pd.Series, expand: bool = True) -> np.ndarray:
        """Preprocess input data for prediction.
        
        Args:
            x: Input data series
            expand: Whether to expand dimensions for batch processing
            
        Returns:
            Preprocessed numpy array
        """
        x = add_indicator_cols_to_input(
            self.data, x, exclude=["file", "model", "model_family"]
        )
        x = x.to_numpy(dtype=np.float32)
        if expand:
            x = np.expand_dims(x, axis=0)
        return x

    @abstractmethod
    def getConfidenceScores(self, x: pd.Series, preprocess: bool = True) -> np.ndarray:
        """Get confidence scores for each class.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Array of confidence scores for each class
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: pd.Series, preprocess: bool = True) -> Tuple[str, float]:
        """Make a prediction for the input data.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Tuple of (predicted class, confidence score)
        """
        raise NotImplementedError

    def topK(self, x: pd.Series, k: int = 3, preprocess: bool = True) -> List[str]:
        """Get top K predictions for the input data.
        
        Args:
            x: Input data series
            k: Number of top predictions to return
            preprocess: Whether to preprocess the input
            
        Returns:
            List of top K predicted classes, ordered by confidence
        """
        conf_scores = self.getConfidenceScores(x, preprocess=preprocess)
        indices = np.argpartition(conf_scores, -k)[-k:]
        indices = indices[np.argsort(np.array(conf_scores)[indices])][::-1]
        return self.label_encoder.inverse_transform(indices)

    def topKConf(
        self, x: pd.Series, k: int = 3, preprocess: bool = True
    ) -> List[Tuple[str, float]]:
        """Get top K predictions with confidence scores.
        
        Args:
            x: Input data series
            k: Number of top predictions to return
            preprocess: Whether to preprocess the input
            
        Returns:
            List of (class, confidence) tuples, ordered by confidence
        """
        conf_scores = self.getConfidenceScores(x, preprocess=preprocess)
        indices = np.argpartition(conf_scores, -k)[-k:]
        result = []
        for idx in indices:
            result.append(
                (
                    self.label_encoder.inverse_transform(np.array([idx]))[0],
                    float(conf_scores[idx]),
                )
            )
        result = sorted(result, key=lambda x: x[1], reverse=True)
        return result

    def printFeatures(self) -> None:
        """Print the features used by the model."""
        for i, col in enumerate(self.orig_cols):
            print(f"Feature {i}:\t{col[:80]}")

    def evaluateTrain(self) -> float:
        """Evaluate model performance on training data.
        
        Returns:
            Training accuracy score
        """
        acc = self.model.score(self.x_tr, self.y_train)
        print(f"{self.name} train acc: {acc}")
        return acc

    def evaluateTest(self) -> float:
        """Evaluate model performance on test data.
        
        Returns:
            Test accuracy score
        """
        acc = self.model.score(self.x_test, self.y_test)
        print(f"{self.name} test acc: {acc}")
        return acc

    def evaluateAcc(
        self, data: pd.DataFrame, y_label: str = "model", preprocess: bool = True
    ) -> float:
        """Evaluate model performance on custom data.
        
        Args:
            data: Input DataFrame
            y_label: Column name for target variable
            preprocess: Whether to preprocess the input
            
        Returns:
            Accuracy score
        """
        y = self.label_encoder.transform(data[y_label])
        if not preprocess:
            x = data.drop(columns=["file", "model_family", "model"], axis=1)
            return self.model.score(x, y)
        table = pd.DataFrame(columns=self.x_tr.columns)
        for index, row in data.iterrows():
            table.loc[index] = self.preprocessInput(row, expand=False)
        return self.model.score(table, y)


class RFEArchPred(ArchPredBase):
    """Base class for models using Recursive Feature Elimination (RFE).
    
    This class extends ArchPredBase to add RFE-specific functionality for
    feature selection and ranking.
    """

    def printFeatures(self) -> None:
        """Print the features selected by RFE."""
        if not hasattr(self.rfe, "support_"):
            return
        print("Remaining Features:")
        support = self.rfe.support_
        for i, col_name in enumerate(self.orig_cols):
            if support[i]:
                print(f"Feature {i}:\t{col_name[:80]}")


    def featureRank(
        self, save_path: Optional[Path] = None, suppress_output: bool = False
    ) -> List[str]:
        """Get ranking of features based on RFE.
        
        Args:
            save_path: Optional path to save feature ranking
            suppress_output: Whether to suppress console output
            
        Returns:
            List of feature names ordered by their ranking
        """
        if not hasattr(self.rfe, "support_"):
            return
        if not suppress_output:
            print(
                "Feature Ranking (Note only ranks features which aren't part of the model):"
            )
        if save_path is not None:
            save_file = Path(save_path) / "feature_ranking.txt"
            f = open(save_file, "w+")
        support = self.rfe.support_
        for i, col_name in enumerate(self.orig_cols):
            if support[i]:
                s = f"Rank 0:\t{col_name}"
                if save_path is not None:
                    f.write(s + "\n")
                if not suppress_output:
                    print(s[:80])
        ranking = {}
        ranks = self.rfe.ranking_
        for i, rank in enumerate(ranks):
            ranking[rank] = self.orig_cols[i]
        for i in range(len(ranking)):
            if i in ranking:
                s = f"Rank {i}:\t{ranking[i]}"
                if save_path is not None:
                    f.write(s + "\n")
                if not suppress_output:
                    print(s[:80])
        if save_path is not None:
            f.close()
        result = []
        for rank in sorted(ranking.keys()):
            result.append(ranking[rank])
        return result


class SKLearnClassifier(ArchPredBase):
    """Base class for scikit-learn based classifiers."""

    def getConfidenceScores(self, x: pd.Series, preprocess: bool = True) -> np.ndarray:
        """Get confidence scores using scikit-learn's decision function.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Array of confidence scores for each class
        """
        if preprocess:
            x = self.preprocessInput(x)
        preds = self.model.decision_function(x)[0]
        return softmax(preds)


    def predict(self, x: pd.Series, preprocess: bool = True) -> Tuple[str, float]:
        """Make a prediction using the scikit-learn model.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Tuple of (predicted class, confidence score)
        """
        preds = self.getConfidenceScores(x, preprocess)
        pred = preds.argmax()
        conf = preds[pred]
        label = self.label_encoder.inverse_transform(np.array([pred]))
        return label[0], conf.item()


class NNArchPred(ArchPredBase):
    """Neural network based architecture prediction model.
    
    This class implements a PyTorch-based neural network for architecture prediction.
    """
    
    NAME = "nn_old"
    FULL_NAME = "Neural Network (PyTorch)"

    def __init__(
        self,
        df: pd.DataFrame,
        label: Optional[str] = None,
        verbose: bool = True,
        hidden_layer_factor: Optional[float] = None,
        num_layers: Optional[int] = None,
        name: Optional[str] = None,
        epochs: int = 100,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ):
        """Initialize neural network model.
        
        Args:
            df: Input DataFrame
            label: Target variable column name
            verbose: Whether to print progress
            hidden_layer_factor: Factor to determine hidden layer sizes
            num_layers: Number of hidden layers
            name: Model name
            epochs: Number of training epochs
            train_size: Size of training set
            test_size: Size of test set
        """
        if name is None:
            name = self.NAME
        super().__init__(
            df=df,
            name=name,
            label=label,
            verbose=verbose,
            deterministic=False,
            train_size=train_size,
            test_size=test_size,
        )
        print(
            f"Instantiating neural net with {self.num_classes} classes and input size of {self.input_size}"
        )
        self.model = Net(
            input_size=self.input_size,
            num_classes=self.num_classes,
            hidden_layer_factor=hidden_layer_factor,
            layers=num_layers,
        )
        self.x_tr = self.model.normalize(self.x_tr, fit=True)
        self.x_test = self.model.normalize(self.x_test)
        self.model.train_(
            self.x_tr,
            self.x_test,
            self.y_train,
            self.y_test,
            verbose=verbose,
            epochs=epochs,
        )
        self.model.eval()


    def getConfidenceScores(self, x: pd.Series, preprocess: bool = True) -> np.ndarray:
        """Get confidence scores from neural network.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Array of confidence scores for each class
        """
        if preprocess:
            x = self.preprocessInput(x)
        preds = self.model.get_preds(x)
        return preds.cpu().numpy()


    def predict(self, x: pd.Series, preprocess: bool = True) -> Tuple[str, float]:
        """Make a prediction using the neural network.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Tuple of (predicted class, confidence score)
        """
        preds = self.getConfidenceScores(x, preprocess)
        pred = preds.argmax()
        conf = preds[pred]
        label = self.label_encoder.inverse_transform(np.array([pred]))
        return label[0], conf.item()


    def evaluateTrain(self) -> float:
        """Evaluate neural network on training data.
        
        Returns:
            Training accuracy score
        """
        train_preds = self.model.get_preds(self.x_tr, normalize=False)
        pred = train_preds.argmax(dim=1).cpu()
        train_pred_labels = self.label_encoder.inverse_transform(np.array(pred))
        y_train_labels = self.label_encoder.inverse_transform(np.array(self.y_train))
        correct1 = sum(train_pred_labels == y_train_labels)
        print(f"X_train acc1: {correct1 / len(self.y_train)}")
        return correct1 / len(self.y_train)


    def evaluateTest(self) -> float:
        """Evaluate neural network on test data.
        
        Returns:
            Test accuracy score
        """
        test_preds = self.model.get_preds(self.x_test, normalize=False)
        pred = test_preds.argmax(dim=1).cpu()
        test_pred_labels = self.label_encoder.inverse_transform(np.array(pred))
        y_test_labels = self.label_encoder.inverse_transform(np.array(self.y_test))
        correct1 = sum(test_pred_labels == y_test_labels)
        print(f"X_test acc1: {correct1 / len(self.y_test)}")
        return correct1 / len(self.y_test)


class NN2LRArchPred(SKLearnClassifier):
    """Neural network implemented using scikit-learn's MLPClassifier.
    
    This class provides a neural network implementation using scikit-learn's
    MLPClassifier instead of PyTorch.
    """
    
    NAME = "nn"
    FULL_NAME = "Neural Network"

    def __init__(
        self,
        df: pd.DataFrame,
        label: Optional[str] = None,
        verbose: bool = True,
        name: Optional[str] = None,
        rfe_num: int = 800,
        solver: str = "lbfgs",
        num_layers: int = 3,
        hidden_layer_factor: float = 1,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ):
        """Initialize scikit-learn neural network model.
        
        Args:
            df: Input DataFrame
            label: Target variable column name
            verbose: Whether to print progress
            name: Model name
            rfe_num: Number of features to select
            solver: Optimization algorithm
            num_layers: Number of hidden layers
            hidden_layer_factor: Factor to determine hidden layer sizes
            train_size: Size of training set
            test_size: Size of test set
        """
        if name is None:
            name = self.NAME
        super().__init__(
            df=df,
            name=name,
            label=label,
            verbose=verbose,
            deterministic=False,
            train_size=train_size,
            test_size=test_size,
        )
        layer_sizes = [len(self.orig_cols)]
        for i in range(num_layers - 1):
            layer_sizes.append(layer_sizes[i] * hidden_layer_factor)
        self.num_layers = num_layers
        self.hidden_layer_factor = hidden_layer_factor
        self.solver = solver
        self.rfe_num = rfe_num
        self.estimator = MLPClassifier(
            hidden_layer_sizes=layer_sizes,
            solver=solver,
        )
        self.model = make_pipeline(StandardScaler(), self.estimator)
        self.model.fit(self.x_tr, self.y_train)
        if self.verbose:
            self.evaluateTest()


    def getConfidenceScores(self, x: pd.Series, preprocess: bool = True) -> np.ndarray:
        """Get confidence scores from scikit-learn neural network.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Array of confidence scores for each class
        """
        if preprocess:
            x = self.preprocessInput(x)
        return softmax(self.model.predict_proba(x)[0])


class LRArchPred(RFEArchPred, SKLearnClassifier):
    """Logistic Regression based architecture prediction model."""
    
    NAME = "lr"
    FULL_NAME = "Logistic Regression"

    def __init__(
        self,
        df: pd.DataFrame,
        label: Optional[str] = None,
        verbose: bool = True,
        rfe_num: Optional[int] = None,
        name: Optional[str] = None,
        multi_class: str = "auto",
        penalty: str = "l2",
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ) -> None:
        """Initialize logistic regression model.
        
        Args:
            df: Input DataFrame
            label: Target variable column name
            verbose: Whether to print progress
            rfe_num: Number of features to select
            name: Model name
            multi_class: Multi-class strategy
            penalty: Regularization penalty
            train_size: Size of training set
            test_size: Size of test set
        """
        if name is None:
            name = self.NAME
        super().__init__(
            df=df,
            name=name,
            label=label,
            verbose=verbose,
            train_size=train_size,
            test_size=test_size,
        )
        self.estimator = LogisticRegression(
            multi_class=multi_class, penalty=penalty, max_iter=1000
        )
        self.rfe_num = rfe_num if rfe_num is not None else len(self.orig_cols)
        self.rfe = RFE(
            estimator=self.estimator,
            n_features_to_select=self.rfe_num,
            verbose=10 if self.verbose else 0,
        )
        if len(self.orig_cols) == 1:
            self.model = make_pipeline(StandardScaler(), MinMaxScaler(), self.estimator)
        else:
            self.model = make_pipeline(
                StandardScaler(), MinMaxScaler(), self.rfe, self.estimator
            )
        self.model.fit(self.x_tr, self.y_train)
        if self.verbose:
            self.printFeatures()
            self.evaluateTest()


class RFArchPred(RFEArchPred, SKLearnClassifier):
    """Random Forest based architecture prediction model."""
    
    NAME = "rf"
    FULL_NAME = "Random Forest"

    def __init__(
        self,
        df: pd.DataFrame,
        label: Optional[str] = None,
        verbose: bool = True,
        rfe_num: Optional[int] = None,
        name: Optional[str] = None,
        num_estimators: int = 100,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ) -> None:
        """Initialize random forest model.
        
        Args:
            df: Input DataFrame
            label: Target variable column name
            verbose: Whether to print progress
            rfe_num: Number of features to select
            name: Model name
            num_estimators: Number of trees in forest
            train_size: Size of training set
            test_size: Size of test set
        """
        if name is None:
            name = self.NAME
        super().__init__(
            df=df,
            name=name,
            label=label,
            verbose=verbose,
            deterministic=False,
            train_size=train_size,
            test_size=test_size,
        )
        self.estimator = RandomForestClassifier(n_estimators=num_estimators)
        self.num_estimators = num_estimators
        self.rfe_num = rfe_num if rfe_num is not None else len(list(self.x_tr))
        self.rfe = RFE(
            estimator=self.estimator,
            n_features_to_select=self.rfe_num,
            verbose=10 if self.verbose else 0,
        )
        if len(self.orig_cols) == 1:
            self.model = make_pipeline(StandardScaler(), MinMaxScaler(), self.estimator)
        else:
            self.model = make_pipeline(
                StandardScaler(), MinMaxScaler(), self.rfe, self.estimator
            )
        self.model.fit(self.x_tr, self.y_train)
        if self.verbose:
            self.printFeatures()
            self.evaluateTest()


    def getConfidenceScores(self, x: pd.Series, preprocess: bool = True) -> np.ndarray:
        """Get confidence scores from random forest.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Array of confidence scores for each class
        """
        if preprocess:
            x = self.preprocessInput(x)
        preds = self.model.predict_proba(x)[0]
        return softmax(preds)


class KNNArchPred(SKLearnClassifier):
    """K-Nearest Neighbors based architecture prediction model."""
    
    NAME = "knn"
    FULL_NAME = "K Nearest Neighbors"

    def __init__(
        self,
        df: pd.DataFrame,
        label: Optional[str] = None,
        verbose: bool = True,
        name: Optional[str] = None,
        k: int = 5,
        weights: str = "distance",
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ) -> None:
        """Initialize KNN model.
        
        Args:
            df: Input DataFrame
            label: Target variable column name
            verbose: Whether to print progress
            name: Model name
            k: Number of neighbors
            weights: Weight function for prediction
            train_size: Size of training set
            test_size: Size of test set
        """
        if name is None:
            name = self.NAME
        super().__init__(
            df=df,
            name=name,
            label=label,
            verbose=verbose,
            train_size=train_size,
            test_size=test_size,
        )
        self.estimator = KNeighborsClassifier(n_neighbors=k, weights=weights)
        self.k = k
        self.weights = weights
        self.model = make_pipeline(StandardScaler(), MinMaxScaler(), self.estimator)
        self.model.fit(self.x_tr, self.y_train)
        if self.verbose:
            self.printFeatures()
            self.evaluateTest()


    def getConfidenceScores(self, x: pd.Series, preprocess: bool = True) -> np.ndarray:
        """Get confidence scores from KNN.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Array of confidence scores for each class
        """
        if preprocess:
            x = self.preprocessInput(x)
        return softmax(self.model.predict_proba(x)[0])


class CentroidArchPred(SKLearnClassifier):
    """Nearest Centroid based architecture prediction model."""
    
    NAME = "centroid"
    FULL_NAME = "Nearest Centroid"

    def __init__(
        self,
        df: pd.DataFrame,
        label: Optional[str] = None,
        verbose: bool = True,
        name: Optional[str] = None,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ) -> None:
        """Initialize nearest centroid model.
        
        Args:
            df: Input DataFrame
            label: Target variable column name
            verbose: Whether to print progress
            name: Model name
            train_size: Size of training set
            test_size: Size of test set
        """
        if name is None:
            name = self.NAME
        super().__init__(
            df=df,
            name=name,
            label=label,
            verbose=verbose,
            train_size=train_size,
            test_size=test_size,
        )
        self.estimator = NearestCentroid()
        self.model = make_pipeline(StandardScaler(), MinMaxScaler(), self.estimator)
        self.model.fit(self.x_tr, self.y_train)
        if self.verbose:
            self.printFeatures()
            self.evaluateTest()


    def getConfidenceScores(self, x: pd.Series, preprocess: bool = True) -> np.ndarray:
        """Get confidence scores from nearest centroid.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Array of confidence scores for each class
        """
        if preprocess:
            x = self.preprocessInput(x)
        pred_class = self.model.predict(x)[0]
        preds = [0] * self.num_classes
        preds[pred_class] = 1
        return preds


class NBArchPred(SKLearnClassifier):
    """Naive Bayes based architecture prediction model."""
    
    NAME = "nb"
    FULL_NAME = "Naive Bayes"

    def __init__(
        self,
        df: pd.DataFrame,
        label: Optional[str] = None,
        verbose: bool = True,
        name: Optional[str] = None,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ) -> None:
        """Initialize naive bayes model.
        
        Args:
            df: Input DataFrame
            label: Target variable column name
            verbose: Whether to print progress
            name: Model name
            train_size: Size of training set
            test_size: Size of test set
        """
        if name is None:
            name = self.NAME
        super().__init__(
            df=df,
            name=name,
            label=label,
            verbose=verbose,
            train_size=train_size,
            test_size=test_size,
        )
        self.estimator = GaussianNB()
        self.model = make_pipeline(StandardScaler(), MinMaxScaler(), self.estimator)
        self.model.fit(self.x_tr, self.y_train)
        if self.verbose:
            self.printFeatures()
            self.evaluateTest()


    def getConfidenceScores(self, x: pd.Series, preprocess: bool = True) -> np.ndarray:
        """Get confidence scores from naive bayes.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Array of confidence scores for each class
        """
        if preprocess:
            x = self.preprocessInput(x)
        return softmax(self.model.predict_proba(x)[0])


class ABArchPred(RFEArchPred, SKLearnClassifier):
    """AdaBoost based architecture prediction model."""
    
    NAME = "ab"
    FULL_NAME = "AdaBoost"

    def __init__(
        self,
        df: pd.DataFrame,
        label: Optional[str] = None,
        verbose: bool = True,
        rfe_num: Optional[int] = None,
        name: Optional[str] = None,
        num_estimators: int = 100,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ) -> None:
        """Initialize AdaBoost model.
        
        Args:
            df: Input DataFrame
            label: Target variable column name
            verbose: Whether to print progress
            rfe_num: Number of features to select
            name: Model name
            num_estimators: Number of estimators
            train_size: Size of training set
            test_size: Size of test set
        """
        if name is None:
            name = self.NAME
        super().__init__(
            df=df,
            name=name,
            label=label,
            verbose=verbose,
            deterministic=False,
            train_size=train_size,
            test_size=test_size,
        )
        self.estimator = AdaBoostClassifier(n_estimators=num_estimators)
        self.num_estimators = num_estimators
        self.rfe_num = rfe_num if rfe_num is not None else len(list(self.x_tr))
        self.rfe = RFE(
            estimator=self.estimator,
            n_features_to_select=self.rfe_num,
            verbose=10 if self.verbose else 0,
        )
        if len(self.orig_cols) == 1:
            self.model = make_pipeline(StandardScaler(), MinMaxScaler(), self.estimator)
        else:
            self.model = make_pipeline(
                StandardScaler(), MinMaxScaler(), self.rfe, self.estimator
            )
        self.model.fit(self.x_tr, self.y_train)
        if self.verbose:
            self.printFeatures()
            self.evaluateTest()


    def getConfidenceScores(self, x: pd.Series, preprocess: bool = True) -> np.ndarray:
        """Get confidence scores from AdaBoost.
        
        Args:
            x: Input data series
            preprocess: Whether to preprocess the input
            
        Returns:
            Array of confidence scores for each class
        """
        if preprocess:
            x = self.preprocessInput(x)
        preds = self.model.predict_proba(x)[0]
        return softmax(preds)


def get_arch_pred_model(
    model_type: str, df: Optional[pd.DataFrame] = None, label: Optional[str] = None, kwargs: Dict[str, Any] = {}
) -> ArchPredBase:
    """Get an architecture prediction model instance.
    
    Args:
        model_type: Type of model to create
        df: Input DataFrame
        label: Target variable column name
        kwargs: Additional keyword arguments for model initialization
        
    Returns:
        Instance of requested model type
    """
    if df is None:
        path = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe"
        df = all_data(path)
    arch_model = {
        NNArchPred.NAME: NNArchPred,
        LRArchPred.NAME: LRArchPred,
        NN2LRArchPred.NAME: NN2LRArchPred,
        KNNArchPred.NAME: KNNArchPred,
        CentroidArchPred.NAME: CentroidArchPred,
        NBArchPred.NAME: NBArchPred,
        RFArchPred.NAME: RFArchPred,
        ABArchPred.NAME: ABArchPred,
    }
    return arch_model[model_type](df=df, label=label, **kwargs)


def arch_model_names() -> List[str]:
    """Get list of available model names.
    
    Returns:
        List of model name strings
    """
    return [
        LRArchPred.NAME,
        NN2LRArchPred.NAME,
        KNNArchPred.NAME,
        CentroidArchPred.NAME,
        NBArchPred.NAME,
        RFArchPred.NAME,
        ABArchPred.NAME,
    ]


def arch_model_full_name() -> Dict[str, str]:
    """Get mapping of model names to their full names.
    
    Returns:
        Dictionary mapping model names to their full names
    """
    return {
        LRArchPred.NAME: LRArchPred.FULL_NAME,
        NN2LRArchPred.NAME: NN2LRArchPred.FULL_NAME,
        KNNArchPred.NAME: KNNArchPred.FULL_NAME,
        CentroidArchPred.NAME: CentroidArchPred.FULL_NAME,
        NBArchPred.NAME: NBArchPred.FULL_NAME,
        RFArchPred.NAME: RFArchPred.FULL_NAME,
        ABArchPred.NAME: ABArchPred.FULL_NAME,
    }
