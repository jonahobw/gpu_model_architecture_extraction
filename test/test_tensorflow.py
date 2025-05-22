"""
This file validates that tensorflow can run on GPU.
"""

import tensorflow as tf
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# tf.debugging.set_log_device_placement(True)

# pretrained_model = tf.keras.applications.MobileNetV3Large()
pretrained_model = tf.keras.applications.ResNet50()

# input = tf.constant(0.0, dtype=tf.float32, shape=pretrained_model.input_shape[1:])
input = tf.constant(0.0, dtype=tf.float32, shape=(224, 224, 3))
input = np.expand_dims(input, axis=0)
output = pretrained_model(input)
print(np.argmax(output))
