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

@tf.function
def convert_tflite(sess, input_place, output_place):
    tf.compat.v1.lite.TFLiteConverter.from_session(sess, input_place, output_place)
    return

def save_graph(graph=None, dir=None):
  import os

  if graph != None:
    graph = tf.compat.v1.get_default_graph()

  name = "graph_saved"
  graph_file_name = name + '.pb'

  tf.io.write_graph(graph, dir, graph_file_name, as_text=False)

def reset():

  #Clear prev tensorboard logs
  if tf.io.gfile.exists(logdir):
    tf.compat.v1.gfile.DeleteRecursively(logdir) 

  #Set and clear default graph
  tf.compat.v1.get_default_graph()
  tf.compat.v1.reset_default_graph()

