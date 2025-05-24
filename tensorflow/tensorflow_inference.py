"""
Executes TensorFlow model inference with profiling capabilities.

This script runs multiple inferences on a specified TensorFlow model with configurable
device placement. It supports various pre-trained models from tf.keras.applications.

The script is designed to be run as an executable with profiling enabled,
making it suitable for performance analysis and benchmarking.

Example Usage:
    ```bash
    # Run 10 inferences on ResNet50 on GPU 0
    python tensorflow_inference.py -model resnet50 -n 10 -gpu 0

    # Run inference on VGG16 on CPU
    python tensorflow_inference.py -model vgg16 -gpu -1

    # Run inference with device placement debugging
    python tensorflow_inference.py -model mobilenet_v2 -debug
    ```

Note:
    - Profiling should be enabled when running this script
    - Input is currently fixed to zeros with shape (224, 224, 3)
    - GPU selection can be disabled by setting -gpu to -1
    - Device placement debugging can be enabled with -debug
"""

import argparse
from typing import Dict, Optional, Type, Union

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for TensorFlow model inference.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run TensorFlow model inference with profiling capabilities"
    )
    parser.add_argument(
        "-model",
        type=str,
        default="resnet50",
        required=False,
        help="Model architecture to use",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1,
        required=False,
        help="Number of inferences to run",
    )
    parser.add_argument(
        "-gpu",
        type=int,
        default=-1,
        required=False,
        help="-1 for CPU, else number of GPU to use",
    )
    parser.add_argument(
        "-debug",
        default=False,
        help="Prints which device completes each operation",
    )
    return parser.parse_args()


# Map of supported model names to their TensorFlow model classes
MODEL_MAP: Dict[str, Type[tf.keras.Model]] = {
    "resnet50": tf.keras.applications.ResNet50,
    "resnet101": tf.keras.applications.ResNet101,
    "resnet152": tf.keras.applications.ResNet152,
    "vgg16": tf.keras.applications.VGG16,
    "vgg19": tf.keras.applications.VGG19,
    "densenet121": tf.keras.applications.DenseNet121,
    "densenet169": tf.keras.applications.DenseNet169,
    "densenet201": tf.keras.applications.DenseNet201,
    "mobilenet_v2": tf.keras.applications.MobileNetV2,
    "mobilenet_v3_large": tf.keras.applications.MobileNetV3Large,
    "mobilenet_v3_small": tf.keras.applications.MobileNetV3Small,
}


def getDeviceName(gpu_num: int) -> str:
    if gpu_num < 0:
        return "/device:CPU:0"
    return f"/device:GPU:{gpu_num}"


def main() -> None:
    """Main execution function for TensorFlow model inference."""
    args = parse_args()

    # Validate GPU selection
    if args.gpu >= 0:
        gpus = tf.config.list_physical_devices("GPU")
        assert args.gpu < len(gpus), f"GPU {args.gpu} not available. Found {len(gpus)} GPUs."

    # Enable device placement debugging if requested
    if args.debug:
        tf.debugging.set_log_device_placement(True)

    # Validate model selection
    assert args.model in MODEL_MAP, f"Valid models are {list(MODEL_MAP.keys())}"
    model = MODEL_MAP[args.model]()

    print(f"Running {args.n} inferences on {args.model} on {getDeviceName(args.gpu)}...")

    # Run inference
    with tf.device(getDeviceName(args.gpu)):
        # Create zero input tensor with shape (224, 224, 3)
        input_tensor = tf.constant(0.0, dtype=tf.float32, shape=(224, 224, 3))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Run inference
        output = model(input_tensor)
        print(f"Output class: {np.argmax(output)}")

    print("Completed.")


if __name__ == "__main__":
    main()
