import numpy as np
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

#model = keras.models.load_model("test_model.h5")

#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#def representative_dataset_gen():
#  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
#    yield [input]
#converter.representative_dataset = representative_dataset_gen
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8  # or tf.uint8
#converter.inference_output_type = tf.int8  # or tf.uint8
#tflite_quant_model = converter.convert()

#with open('test_model.tflite', 'wb') as f:
#  f.write(tflite_test_model)
