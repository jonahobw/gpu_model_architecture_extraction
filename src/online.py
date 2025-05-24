"""
Online statistics computation using Welford's algorithm.

Adapted from https://github.com/jonahobw/shrinkbench/blob/master/util/online.py

This module provides classes for computing running statistics (mean, variance, standard deviation)
incrementally from a stream of data. It implements Welford's online algorithm for numerical
stability and efficiency.

Example Usage:
    ```python
    # Compute running statistics for a stream of numbers
    stats = OnlineStats()
    stats.add(1.0)
    stats.add(2.0)
    stats.add(3.0)
    print(f"Mean: {stats.mean}, Std: {stats.std}")

    # Compute statistics for multiple metrics
    stats_map = OnlineStatsMap("accuracy", "loss")
    stats_map.add("accuracy", 0.95)
    stats_map.add("loss", 0.1)
    ```
"""

from typing import Dict, Iterable, Optional, Union

import numpy as np


class OnlineStats:
    """
    Welford's algorithm to compute sample mean and sample variance incrementally.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
    """

    def __init__(self, iterable: Optional[Iterable[float]] = None) -> None:
        """
        Initialize the online statistics calculator.

        Args:
            iterable (Optional[Iterable[float]]): Initial values to process.
                Can be any iterable of numbers. Defaults to None.

        Note:
            If iterable is provided, all values will be processed immediately.
        """
        self.n = 0
        self.mean = 0.0
        self.S = 0.0
        if iterable is not None:
            self.addN(iterable)

    def add(self, datum: float) -> None:
        """
        Add a single datum to the running statistics.

        Updates the internal statistics using Welford's method:
        - Updates the count of samples
        - Updates the running mean
        - Updates the running sum of squared differences

        Args:
            datum (float): New value to add to the statistics
        """
        self.n += 1
        delta = datum - self.mean
        # Mk = Mk-1 + (xk – Mk-1)/k
        self.mean += delta / self.n
        # Sk = Sk-1 + (xk – Mk-1)*(xk – Mk)
        self.S += delta * (datum - self.mean)

    def addN(self, iterable: Iterable[float], batch: bool = False) -> None:
        """
        Add multiple data points to the running statistics.

        Args:
            iterable (Iterable[float]): Collection of values to add
            batch (bool): If True, computes statistics over the new array using numpy
                and then updates current stats. If False, processes each value
                individually. Defaults to False.

        Note:
            When batch=True, the statistics are computed over the new array using numpy
            and then combined with the current statistics. This is more efficient for
            large arrays but may be less numerically stable.
        """
        if batch:
            add = self + OnlineStats.from_values(
                len(iterable), np.mean(iterable), np.std(iterable), 0
            )
            self.n, self.mean, self.S = add.n, add.mean, add.S
        else:
            for datum in iterable:
                self.add(datum)

    def pop(self, datum: float) -> None:
        """
        Remove a single datum from the running statistics.

        Args:
            datum (float): Value to remove from the statistics

        Raises:
            ValueError: If there are no samples to remove (n == 0)
        """
        if self.n == 0:
            raise ValueError("Stats must be non empty")

        self.n -= 1
        delta = datum - self.mean
        # Mk-1 = Mk - (xk - Mk) / (k - 1)
        self.mean -= delta / self.n
        # Sk-1 = Sk - (xk – Mk-1) * (xk – Mk)
        self.S -= (datum - self.mean) * delta

    def popN(self, iterable, batch=False):
        raise NotImplementedError

    @property
    def variance(self) -> float:
        """
        Get the current sample variance.

        Returns:
            float: Sample variance (S/n)
        """
        return self.S / self.n

    @property
    def std(self) -> float:
        """
        Get the current sample standard deviation.

        Returns:
            float: Sample standard deviation (sqrt(variance))
        """
        return np.sqrt(self.variance)

    @property
    def flatmean(self):
        # for datapoints which are arrays
        return np.mean(self.mean)

    @property
    def flatvariance(self):
        # for datapoints which are arrays
        return np.mean(self.variance + self.mean**2) - self.flatmean**2

    @property
    def flatstd(self):
        return np.sqrt(self.flatvariance)

    @staticmethod
    def from_values(n: int, mean: float, std: float) -> "OnlineStats":
        """
        Create an OnlineStats instance from pre-computed values.

        Args:
            n (int): Number of samples
            mean (float): Mean of the samples
            std (float): Standard deviation of the samples

        Returns:
            OnlineStats: New instance initialized with the given values
        """
        stats = OnlineStats()
        stats.n = n
        stats.mean = mean
        stats.S = std**2 * n
        return stats

    @staticmethod
    def from_raw_values(n: int, mean: float, S: float) -> "OnlineStats":
        """
        Create an OnlineStats instance from raw statistics.

        Args:
            n (int): Number of samples
            mean (float): Mean of the samples
            S (float): Running sum of squared differences

        Returns:
            OnlineStats: New instance initialized with the given values
        """
        stats = OnlineStats()
        stats.n = n
        stats.mean = mean
        stats.S = S
        return stats

    def __str__(self):
        return f"n={self.n}  mean={self.mean}  std={self.std}"

    def __repr__(self):
        return (
            f"OnlineStats.from_values("
            + f"n={self.n}, mean={self.mean}, "
            + f"std={self.std})"
        )

    def __add__(self, other: Union["OnlineStats", float, int]) -> "OnlineStats":
        """
        Add statistics or a constant to the current statistics.

        Args:
            other (Union[OnlineStats, float, int]): Statistics to add or constant value

        Returns:
            OnlineStats: New instance with combined statistics

        Raises:
            TypeError: If other is not an OnlineStats instance or a number
        """
        if isinstance(other, OnlineStats):
            # Add the means, variances and n_samples of two objects
            n1, n2 = self.n, other.n
            mu1, mu2 = self.mean, other.mean
            S1, S2 = self.S, other.S
            # New stats
            n = n1 + n2
            mu = n1 / n * mu1 + n2 / n * mu2
            S = (S1 + n1 * mu1 * mu1) + (S2 + n2 * mu2 * mu2) - n * mu * mu
            return OnlineStats.from_raw_values(n, mu, S)
        if isinstance(other, (int, float)):
            # Add a fixed amount to all values. Only changes the mean
            return OnlineStats.from_raw_values(self.n, self.mean + other, self.S)
        else:
            raise TypeError("Can only add other groups or numbers")

    def __sub__(self, other):
        raise NotImplementedError

    def __mul__(self, k):
        # Multiply all values seen by some constant
        return OnlineStats.from_raw_values(self.n, self.mean * k, self.S * k**2)


class OnlineStatsMap:
    """
    A map of online statistics for multiple metrics.

    This class maintains separate OnlineStats instances for each metric,
    allowing tracking of multiple statistics simultaneously.

    Attributes:
        stats (Dict[str, OnlineStats]): Dictionary mapping metric names to their statistics
    """

    def __init__(self, *keys: str) -> None:
        """
        Initialize the statistics map.

        Args:
            *keys (str): Initial metric names to track
        """
        self.stats: Dict[str, OnlineStats] = {}
        if keys:
            self.register(*keys)

    def register(self, *keys: str) -> None:
        """
        Register new metrics to track.

        Args:
            *keys (str): Names of metrics to add
        """
        for k in keys:
            if k not in self.stats:
                self.stats[k] = OnlineStats()

    def __str__(self) -> str:
        """
        Get a string representation of all statistics.

        Returns:
            str: String showing statistics for each metric
        """
        s = "Stats"
        for k in self.stats:
            s += f"  {k}:  {str(self.stats[k])}"
        return s
