# test victim profile filtering
import json
import sys

# setting path
sys.path.append("../edge_profile")

from model_manager import VictimModelManager

arch = "alexnet"
path = [x for x in VictimModelManager.getModelPaths() if str(x).find(arch) >= 0][0]

manager = VictimModelManager.load(path)

filters = {"profile_number": "2282150"}
profile = manager.getProfile(filters=filters)
print(json.dumps(profile[1]))

print("\n\n")


profiles = manager.getAllProfiles(filters=filters)
for profile_path, config in profiles:
    print(json.dumps(config))
