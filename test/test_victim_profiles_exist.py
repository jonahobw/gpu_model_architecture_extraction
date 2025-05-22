import sys

# setting path
sys.path.append("../edge_profile")

from model_manager import VictimModelManager


def gpuProfExists(gpu_type: str):
    paths = VictimModelManager.getModelPaths()
    for path in paths:
        manager = VictimModelManager.load(path)
        _, conf = manager.getProfile(filters={"gpu_type": gpu_type})
        assert manager.architecture in conf["file"]


if __name__ == "__main__":
    gpuProfExists("tesla_t4")
