"""
CSV Logger from https://github.com/jonahobw/shrinkbench/blob/master/util/csvlogger.py
"""
import csv
from pathlib import Path

import torch


class CSVLogger:
    def __init__(self, folder, columns, append: bool = True, name: str = None):
        """General purpose CSV Logger
        Initialized with a set of columns, it then has two operations
          - set(**kwargs) - to add entries into the current row
          - update - flush a row to file
        Arguments:
            folder {str} -- Path to folder where file will be created
            columns {List[str]} -- List of keys that CSV is going to log
        """
        if name is None:
            name = "logs.csv"
        assert name.endswith(".csv")
        file = folder / name
        print(f"Logging results to {file}")
        file_existed = file.exists()
        if file_existed and not append:
            raise FileExistsError
        self.file = open(file, "a+")
        self.columns = columns
        self.values = {}
        self.writer = csv.writer(self.file)

        if not file_existed:
            self.writer.writerow(self.columns)
        self.file.flush()

        self.to_write = []  # buffer used for future writes

    def futureWrite(self, kwargs: dict):
        """Store the call for the future"""
        self.to_write.append(kwargs)

    def flush(self):
        """Write all future writes (from calls to self.futureWrite())"""
        for kwargs in self.to_write:
            self.set(**kwargs)
            self.update()
        self.to_write = []

    def set(self, **kwargs):
        """Set value for current row
        [description]
        Arguments:
            **kwargs {[type]} -- [description]
        Raises:
            ValueError -- [description]
        """
        for k, v in kwargs.items():
            if k in self.columns:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.values[k] = v
            else:
                raise ValueError(f"{k} not in columns {self.columns}")

    def update(self):
        """Take current values and write a row in the CSV"""
        row = [self.values.get(c, "") for c in self.columns]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """Close the file descriptor for the CSV"""
        self.file.close()
