import tensorflow as tf
import numpy as np

def import_data():
    (data_train, labels_train), (data_test, labels_test) = tf.keras.datasets.mnist.load_data() #Data Retrieval

    #Reshaping Data
    data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], data_train.shape[2], 1)
    data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], data_test.shape[2], 1)

    #Defining Input Tensor Shape
    input_tensor_shape = (data_test.shape[1], data_test.shape[2], 1)

    #Defining tensor type
    data_train = data_train.astype('float32')
    data_test = data_test.astype('float32')

    #Normalizing/Standardization
    data_train /= 255
    data_test /= 255

    return data_test

#@tf.numpy_function
@tf.function
def convert_tflite(sess, input_place, output_place):
    tf.compat.v1.lite.TFLiteConverter.from_session(sess, input_place, output_place)
    return

