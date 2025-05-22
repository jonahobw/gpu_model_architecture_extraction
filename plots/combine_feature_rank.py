"""Combine the feature ranks from several models"""

import sys

# plt.style.use('ggplot')

# setting path
sys.path.append("../edge_profile")

from arch_pred_accuracy import loadReport, saveFeatureRank


if __name__ == "__main__":
    to_combine = ["ab", "lr", "rf"]

    total_rank = {}
    combined_rank_files = []
    save_name = "combined_feature_rank"

    for model_type in to_combine:
        file = f"feature_rank_{model_type}.json"
        combined_rank_files.append(file)
        save_name += f"_{model_type}"
        report = loadReport(file)
        for i, feature in enumerate(report["feature_rank"]):
            if feature in total_rank:
                total_rank[feature] += i
            else:
                total_rank[feature] = i

    combined_rank = [x[0] for x in sorted(total_rank.items(), key=lambda x: x[1])]
    saveFeatureRank(
        combined_rank,
        metadata={"files": combined_rank_files},
        save_name=f"{save_name}.json",
    )
