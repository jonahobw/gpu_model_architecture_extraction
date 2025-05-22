from pathlib import Path
import sys
import json
from typing import List
from tqdm import tqdm

sys.path.append("../edge_profile")

from architecture_prediction import (
    get_arch_pred_model,
    arch_model_names,
)
from data_engineering import filter_cols, all_data
from format_profiles import parse_one_profile, findProfiles
from get_model import name_to_family


def predictOneProfile(
    df,
    profile_path,
    arch_pred_names: List[str],
    gpu: int = 0,
    verbose: bool = False,
    k: int = 5,
):
    profile_features = parse_one_profile(profile_path, gpu=gpu)
    for arch_name in arch_pred_names:
        arch_pred_model = get_arch_pred_model(
            model_type=arch_name, df=df, kwargs={"verbose": verbose}
        )
        preds = arch_pred_model.topKConf(profile_features, k=k)
        print(f"{arch_name} prediction for {PROFILE_PATH[-50:]}\n")
        for pred in preds:
            print(f"\t{pred}")


def predictFolder(
    df,
    folder_path,
    arch_pred_names: List[str],
    gpu: int = 0,
    verbose: bool = False,
    topk: int = 5,
):
    profiles = findProfiles(folder_path)
    result = {}
    for arch_pred_name in tqdm(arch_pred_names):
        arch_pred_model = get_arch_pred_model(
            model_type=arch_pred_name, df=df, kwargs={"verbose": verbose}
        )

        # this result will keep track of top k accuracy per architecture prediction model
        arch_pred_result = {
            "topk": {x: 0 for x in range(1, topk + 1)},
            "family_top1": 0,
        }
        profiles_tested = 0
        for dnn_arch in profiles:
            for profile in profiles[dnn_arch]:
                profile_features = parse_one_profile(profile, gpu=gpu)
                preds = arch_pred_model.topK(profile_features, k=topk)
                if name_to_family[preds[0]] == name_to_family[dnn_arch]:
                    arch_pred_result["family_top1"] += 1
                for k in range(1, topk + 1):
                    top_k_preds = preds[:k]
                    correct = dnn_arch in top_k_preds
                    if correct:
                        arch_pred_result["topk"][k] += 1
                profiles_tested += 1
        for k in range(1, topk + 1):
            arch_pred_result["topk"][k] = arch_pred_result["topk"][k] / profiles_tested
        arch_pred_result["family_top1"] = (
            arch_pred_result["family_top1"] / profiles_tested
        )
        result[arch_pred_name] = arch_pred_result
    result["profiles tested"] = profiles_tested
    print(json.dumps(result, indent=4))
    return result


if __name__ == "__main__":
    PROFILE_PATH = "/Users/jgow98/Library/CloudStorage/OneDrive-UniversityofMassachusetts/2Grad/research/model_extraction/code/edge_profile/profiles/quadro_rtx_8000/tensorflow_profiles/resnet50/resnet50679033.csv"
    PROFILE_FOLDER = (
        Path(__file__).parent.parent
        / "profiles"
        / "quadro_rtx_8000"
        / "tensorflow_profiles"
    )
    GPU = 1
    TRAIN_DATA = (
        Path(__file__).parent.parent
        / "profiles"
        / "quadro_rtx_8000"
        / "zero_exe_pretrained"
    )

    FEATURE_RANK_PATH = (
        Path(__file__).parent.parent / "plots" / "reports" / "rf_gpu_kernels_nomem.json"
    )
    NUM_FEATURES = 10  # how many features to use

    ARCH_PRED_NAMES = ["rf", "knn"]  # arch_model_names()
    K = 1
    PREDICT_ONE = False

    # ------------------------------------------

    df = all_data(TRAIN_DATA)
    with open(FEATURE_RANK_PATH, "r") as f:
        feature_rank = json.load(f)["feature_rank"]
    selected_features = feature_rank[:NUM_FEATURES]
    df = filter_cols(df, substrs=selected_features)

    if PREDICT_ONE:
        predictOneProfile(
            df=df,
            profile_path=PROFILE_PATH,
            arch_pred_names=ARCH_PRED_NAMES,
            gpu=GPU,
            verbose=False,
            k=K,
        )
    else:
        predictFolder(
            df=df,
            folder_path=PROFILE_FOLDER,
            arch_pred_names=ARCH_PRED_NAMES,
            gpu=GPU,
            verbose=False,
            topk=K,
        )
