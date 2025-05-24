"""
Combine feature rankings from multiple models into a single consensus ranking.

This module takes feature rankings from different models and combines them into a single
consensus ranking. The combination is done by averaging the ranks of each feature across
all models, then sorting by the average rank to create the final ranking.

Example Usage:
    ```python
    # Combine feature rankings from multiple models
    to_combine = ["ab", "lr", "rf"]
    combined_rank = combine_feature_ranks(to_combine)
    saveFeatureRank(combined_rank, metadata={"files": combined_rank_files})
    ```
"""

import sys
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append("../edge_profile")

from arch_pred_accuracy import loadReport, saveFeatureRank


def combine_feature_ranks(model_types: List[str]) -> Tuple[List[str], List[str]]:
    """
    Combine feature rankings from multiple models into a single consensus ranking.

    Args:
        model_types (List[str]): List of model types whose feature rankings to combine.
            Each model type should have a corresponding feature rank file named
            "feature_rank_{model_type}.json".

    Returns:
        Tuple[List[str], List[str]]: A tuple containing:
            - List[str]: The combined feature ranking
            - List[str]: List of input feature rank files used

    The combination process:
    1. Loads feature rankings from each model
    2. For each feature, sums its ranks across all models
    3. Sorts features by their total rank to create the consensus ranking
    """
    total_rank: Dict[str, int] = {}
    combined_rank_files: List[str] = []

    for model_type in model_types:
        file = f"feature_rank_{model_type}.json"
        combined_rank_files.append(file)
        report = loadReport(file)
        for i, feature in enumerate(report["feature_rank"]):
            if feature in total_rank:
                total_rank[feature] += i
            else:
                total_rank[feature] = i

    combined_rank = [x[0] for x in sorted(total_rank.items(), key=lambda x: x[1])]
    return combined_rank, combined_rank_files


def main() -> None:
    """
    Main function to combine feature rankings and save the result.

    This function:
    1. Specifies the models whose feature rankings to combine
    2. Combines the rankings using combine_feature_ranks
    3. Saves the combined ranking with metadata about the input files
    """
    # Models whose feature rankings to combine
    to_combine = ["ab", "lr", "rf"]

    # Combine rankings and get list of input files
    combined_rank, combined_rank_files = combine_feature_ranks(to_combine)

    # Generate save name based on input models
    save_name = "combined_feature_rank"
    for model_type in to_combine:
        save_name += f"_{model_type}"

    # Save the combined ranking
    saveFeatureRank(
        combined_rank,
        metadata={"files": combined_rank_files},
        save_name=f"{save_name}.json",
    )


if __name__ == "__main__":
    main()
