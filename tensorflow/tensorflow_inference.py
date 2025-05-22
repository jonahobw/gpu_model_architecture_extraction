"""
Runs N inferences on a model A (maybe pretrained) on GPU K with inputs X.
This file is turned into an executable and profiling is enabled while running the executable.
"""
import argparse

import numpy as np

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default="resnet50", required=False)
parser.add_argument(
    "-n", type=int, default=1, required=False, help="number of inferences"
)
parser.add_argument(
    "-gpu", type=int, default=-1, required=False, help="-1 for cpu, else number of gpu"
)
# parser.add_argument("-input", type=str, default="random",
#                     help="Input type to pass to model. See construct_inputs.py")
# parser.add_argument("-seed", type=int, default=42, help="Random seed for random inputs.")
# parser.add_argument("-pretrained", action='store_true', help="Use a pretrained model")
# parser.add_argument("-load_path", default=None, required=False, help="Provide a path to a model to be used.")
parser.add_argument(
    "-debug", default=False, help="Prints which device completes each operation"
)

args = parser.parse_args()

MODEL_MAP = {
    # "alexnet": tf.keras.applications.,
    # "resnet18": tf.keras.applications.,
    # "resnet34": tf.keras.applications.,
    "resnet50": tf.keras.applications.ResNet50,
    "resnet101": tf.keras.applications.ResNet101,
    "resnet152": tf.keras.applications.ResNet152,
    # "resnext50_32x4d": tf.keras.applications.,
    # "resnext101_32x8d": tf.keras.applications.,
    # "wide_resnet50_2": tf.keras.applications.,
    # "wide_resnet101_2": tf.keras.applications.,
    # "vgg11": tf.keras.applications.,
    # "vgg11_bn": tf.keras.applications.,
    # "vgg13": tf.keras.applications.,
    # "vgg13_bn": tf.keras.applications.,
    "vgg16": tf.keras.applications.VGG16,
    # "vgg16_bn": tf.keras.applications.,
    # "vgg19_bn": tf.keras.applications.,
    "vgg19": tf.keras.applications.VGG19,
    # "squeezenet1_0": tf.keras.applications.,
    # "squeezenet1_1": tf.keras.applications.,
    "densenet121": tf.keras.applications.DenseNet121,
    "densenet169": tf.keras.applications.DenseNet169,
    "densenet201": tf.keras.applications.DenseNet201,
    # "densenet161": tf.keras.applications.,
    # "googlenet": tf.keras.applications.,
    "mobilenet_v2": tf.keras.applications.MobileNetV2,
    "mobilenet_v3_large": tf.keras.applications.MobileNetV3Large,
    "mobilenet_v3_small": tf.keras.applications.MobileNetV3Small,
    # "mnasnet0_5": tf.keras.applications.,
    # "mnasnet0_75": tf.keras.applications.,
    # "mnasnet1_0": tf.keras.applications.,
    # "mnasnet1_3": tf.keras.applications.,
    # "shufflenet_v2_x0_5": tf.keras.applications.,
    # "shufflenet_v2_x1_0": tf.keras.applications.,
    # "shufflenet_v2_x1_5": tf.keras.applications.,
    # "shufflenet_v2_x2_0": tf.keras.applications.
}


def getDeviceName(gpu_num: int) -> str:
    if gpu_num < 0:
        return "/device:CPU:0"
    return f"/device:GPU:{gpu_num}"


assert args.gpu <= len(tf.config.list_physical_devices("GPU"))
if args.debug:
    tf.debugging.set_log_device_placement(True)

assert args.model in MODEL_MAP, f"Valid models are {list(MODEL_MAP.keys())}"
model = MODEL_MAP[args.model]()

print(f"Running {args.n} inferences on {args.model} on {getDeviceName(args.gpu)}...")

with tf.device(getDeviceName(args.gpu)):
    # input = tf.constant(0.0, dtype=tf.float32, shape=pretrained_model.input_shape[1:])
    input = tf.constant(0.0, dtype=tf.float32, shape=(224, 224, 3))
    input = np.expand_dims(input, axis=0)
    output = model(input)
print(np.argmax(output))

print("Completed.")
