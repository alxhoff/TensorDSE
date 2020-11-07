import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('../splitting_tflite_model/saved_test_model')

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0


def representative_dataset_gen():
    mnist_train, _ = tf.keras.datasets.mnist.load_data()
    images = tf.cast(mnist_train[0], tf.float32) / 255.0
    mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
    for input_value in mnist_ds.take(100):
    # Model has only one input so each data point has one element.
        yield [input_value]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8



tflite_quant_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

import pathlib

tflite_model_dir = pathlib.Path("../splitting_tflite_model/mnist_tflite_model/")
tflite_model_dir.mkdir(exist_ok=True, parents=True)

# Save the quantized model:
tflite_quant_model_file = tflite_model_dir/"test_model_quant.tflite"
tflite_quant_model_file.write_bytes(tflite_quant_model)
