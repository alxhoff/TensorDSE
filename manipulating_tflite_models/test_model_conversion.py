import numpy as np
import tensorflow as tf
from tensorflow import keras
import pathlib

#Load the Keras model from appropriate directory
model = tf.keras.models.load_model('../splitting_tflite_model/saved_test_model')

#Reload MNIST data set needed for the representative_dataset_gen function
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#Normalize data to have values between 0 and 1
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

#Creates a data sample needed by the TFLite Converter
def representative_dataset_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]

#Full-integer quantization followed by conversion to tflite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8

tflite_quant_model = converter.convert()

#Creates directory to save the converted model
tflite_model_dir = pathlib.Path("../splitting_tflite_model/mnist_tflite_model/")
tflite_model_dir.mkdir(exist_ok=True, parents=True)

# Save the model:
tflite_quant_model_file = tflite_model_dir/"test_model_quant.tflite"
tflite_quant_model_file.write_bytes(tflite_quant_model)
