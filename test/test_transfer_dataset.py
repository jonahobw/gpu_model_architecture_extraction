import sys

# setting path
sys.path.append("../edge_profile")

from model_manager import VictimModelManager

arch = "alexnet"
path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]

manager = VictimModelManager.load(path)

dataset = "tiny-imagenet-200"
transfer_size = 1000
sample_average = 5
random = False
entropy = False

manager.loadKnockoffTransferSet(
    dataset_name=dataset,
    transfer_size=transfer_size,
    sample_avg=sample_average,
    random_policy=random,
    entropy=entropy,
    force=True,
)

# file, dataset = manager.loadKnockoffTransferSet(dataset_name=dataset, transfer_size=transfer_size, sample_avg=sample_average, random_policy=random)

# print(file)
# print(dataset.config)
