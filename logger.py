"""
A CSV logging utility for tracking and storing experiment results.

This module provides a CSVLogger class that enables structured logging of experiment
metrics and results to CSV files. It supports both immediate and buffered writing,
making it suitable for various logging scenarios.

Features:
    - Column-based structured logging
    - Support for PyTorch tensors
    - Buffered writing for batch operations
    - Append or overwrite modes
    - Automatic file creation and management

Example Usage:
    ```python
    # Create a logger for tracking training metrics
    logger = CSVLogger(
        folder="experiments/results",
        columns=["epoch", "loss", "accuracy"],
        name="training_log.csv"
    )
    
    # Log a single row
    logger.set(epoch=1, loss=0.5, accuracy=0.95)
    logger.update()
    
    # Buffer multiple rows for batch writing
    logger.futureWrite({"epoch": 2, "loss": 0.4, "accuracy": 0.96})
    logger.futureWrite({"epoch": 3, "loss": 0.3, "accuracy": 0.97})
    logger.flush()  # Write all buffered rows
    
    # Clean up
    logger.close()
    ```

Note:
    - Adapted from https://github.com/jonahobw/shrinkbench/blob/master/util/csvlogger.py
    - All numeric values are automatically converted to Python scalars
    - File is automatically created if it doesn't exist
    - Supports both append and overwrite modes
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch


class CSVLogger:
    """
    A CSV logging utility for structured experiment logging.

    This class provides methods for logging structured data to CSV files,
    with support for both immediate and buffered writing operations.
    """

    def __init__(
        self,
        folder: Union[str, Path],
        columns: List[str],
        append: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize the CSV logger.

        Args:
            folder: Path to folder where the CSV file will be created
            columns: List of column names that will be logged
            append: If True, append to existing file; if False, raise error if file exists
            name: Name of the CSV file (default: "logs.csv")

        Raises:
            FileExistsError: If append is False and the file already exists
            AssertionError: If the filename doesn't end with .csv
        """
        if name is None:
            name = "logs.csv"
        assert name.endswith(".csv")
        file = Path(folder) / name
        print(f"Logging results to {file}")
        file_existed = file.exists()
        if file_existed and not append:
            raise FileExistsError
        self.file = open(file, "a+")
        self.columns = columns
        self.values: Dict[str, Any] = {}
        self.writer = csv.writer(self.file)

        if not file_existed:
            self.writer.writerow(self.columns)
        self.file.flush()

        self.to_write: List[Dict[str, Any]] = []  # buffer used for future writes

    def futureWrite(self, kwargs: Dict[str, Any]) -> None:
        """
        Buffer a row for future writing.

        Args:
            kwargs: Dictionary of column names and values to be written
        """
        self.to_write.append(kwargs)

    def flush(self) -> None:
        """
        Write all buffered rows to the CSV file.

        This method processes all rows that were previously buffered using
        futureWrite() and writes them to the file.
        """
        for kwargs in self.to_write:
            self.set(**kwargs)
            self.update()
        self.to_write = []

    def set(self, **kwargs: Any) -> None:
        """
        Set values for the current row.

        Args:
            **kwargs: Column names and their corresponding values

        Raises:
            ValueError: If a column name is not in the predefined columns list
        """
        for k, v in kwargs.items():
            if k in self.columns:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.values[k] = v
            else:
                raise ValueError(f"{k} not in columns {self.columns}")

    def update(self) -> None:
        """
        Write the current row to the CSV file.

        This method takes the current values and writes them as a row in the CSV,
        maintaining the order of columns as specified during initialization.
        """
        row = [self.values.get(c, "") for c in self.columns]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """Close the file descriptor for the CSV"""
        self.file.close()
