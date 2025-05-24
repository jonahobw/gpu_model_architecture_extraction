"""
This file tests an output from the function
testKnockoffTrain() in the experiments.py file.
Specifically, the testKnockoffTrain() function
tries different parameter configurations and 
records metrics for each.  This file looks at
each metric and finds the best configuration to
minimize or maximize that metric. Essentially it
is a grid search.
"""
import json
from pathlib import Path

import pandas as pd

file = Path.cwd() / "test_knockoff.csv"
df = pd.read_csv(file)

parameters = [
    "dataset",
    "transfer_size",
    "sample_avg",
    "random_policy",
    "entropy",
    "pretrained",
]
metrics = ["val_loss", "val_acc1", "val_agreement", "l1_weight_bound"]

result = {}
for metric in metrics:
    metric_results = {}
    for parameter in parameters:
        unique_vals = df[parameter].unique()
        best_choice = unique_vals[0]
        best_val = df[df[parameter] == best_choice][metric].mean()
        for choice in unique_vals:
            val = df[df[parameter] == choice][metric].mean()
            if metric in ["val_loss", "l1_weight_bound"]:
                if val < best_val:
                    best_val = val
                    best_choice = choice
            else:
                if val > best_val:
                    best_val = val
                    best_choice = choice
        metric_results[parameter] = best_choice
    result[metric] = metric_results

print(json.dumps(result, indent=4, default=lambda x: str(x)))

max_val = df[df["transfer_size"] == 10000]["val_acc1"].max()
print(df[df["transfer_size"] == 10000][df["val_acc1"] > max_val * 0.9])
