import platform
from utils import *

edge_folder="results/edge/"
cpu_folder="results/cpu/"
gpu_folder="results/gpu/"

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def deduce_operation(compiled_file):
    f = compiled_file
    op = f.split("edge_")[1]
    op = op.split("_edgetpu.tflite")[0]

    return op

def make_interpreter(model_file):

    import tflite_runtime.interpreter as tflite

    model_file, *device = model_file.split('@')

    device = {'device': device[0]} if device else {}
    shared_library = EDGETPU_SHARED_LIB
    experimental_delegates = [ tflite.load_delegate(shared_library, device) ]   #Returns loaded Delegate object

    return tflite.Interpreter(model_path=model_file, 
                              model_content=None, 
                              experimental_delegates=experimental_delegates)

def edge_group_tflite_deployment(models_folder):

    import os
    from os import listdir
    from os.path import isfile, join

    tflite_model=None
    op_path = models_folder

    for edge_file in listdir(models_folder):
        tflite_model = edge_file

        if(tflite_model):
            op = deduce_operation(tflite_model)
            tflite_model = op_path + "/" + tflite_model
            edge_tflite_deployment(tflite_model, op, 1000)


def edge_tflite_deployment(model_file, model_name, count):

    import time
    import numpy as np

    interpreter = make_interpreter(model_file)                                  #Creates Interpreter Object
    interpreter.allocate_tensors()                                              #Allocates its tensors

    """
    This is where model specific inputs/labels are fed to the model in
    order to be run correctly.
    """

    input_details = interpreter.get_input_details()                             #Get input and output tensors
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']                                     #Test the model on randon input data
    input_dtype = input_details[0]['dtype']                                     #Test the model on randon input data

    input_data = np.array(np.random.random_sample(input_shape),dtype=input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    for i in range(count):

        print('----INFERENCE TIME----')
        start = time.perf_counter()
        interpreter.invoke()                                                    #Runs the interpreter/inference, be sure
                                                                                #to have set the input sizes and allocate 
                                                                                #tensors.
        """ This is where we should 
            retrieve the ouput
        """
        inference_time = time.perf_counter() - start
        output_data = interpreter.get_tensor(output_details[0]['index'])

        EDGE_RESULTS.append([i, inference_time])

    create_csv_file(edge_folder, model_name, EDGE_RESULTS)

def cpu_group_tflite_deployment(models_folder, operations):
    for op in operations:
        op_path = models_folder + op + "/"
        tflite_model = fetch_file(op_path, ".tflite")
        tflite_model = op_path + tflite_model

        cpu_tflite_deployment(tflite_model, op, 1000)


def cpu_tflite_deployment(model_file, model_name, count):
    import time
    import tensorflow as tf
    import numpy as np

    CPU_RESULTS = []

    interpreter = tf.lite.Interpreter(model_path=model_file)                    #Creates Interpreter Object
    interpreter.allocate_tensors()                                              #Allocates its tensors

    input_details = interpreter.get_input_details()                             #Get input and output tensors
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']                                     #Test the model on randon input data
    input_dtype = input_details[0]['dtype']                                     #Test the model on randon input data

    #input_data = np.array(np.random.random_sample(input_shape),dtype=np.float32)
    input_data = np.array(np.random.random_sample(input_shape),dtype=input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    for i in range(count):

        """ INFERENCE TIME
        """
        start = time.perf_counter()
        interpreter.invoke()                                                    #Runs the interpreter/inference, be sure
                                                                                #to have set the input sizes and allocate 

        inference_time = time.perf_counter() - start
        output_data = interpreter.get_tensor(output_details[0]['index'])

        """ END
        """
        CPU_RESULTS.append([i, inference_time])

    create_csv_file(cpu_folder, model_name, CPU_RESULTS)

def gpu_group_tflite_deployment(models_folder, operations):
    pass

def gpu_tflite_deployment(model_name, model_file, count):
    pass

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model', required=True, help='File path to the .tflite file.')
    parser.add_argument('-d', '--delegate', required=True, help='cpu, gpu or edge_tpu.')
    parser.add_argument('-n', '--name', required=True, help='Name of Model/Operation, needed to create corresponding folder name.')
    parser.add_argument('-c', '--count', type=int, default=5,help='Number of times to run inference.')

    args = parser.parse_args()

    if ("cpu" in args.delegate):
        cpu_tflite_deployment(args.model, args.name, args.count)
    elif ("gpu" in args.delegate):
        gpu_tflite_deployment(args.model, args.name, args.count)
    elif ("edge_tpu" in args.delegate):
        edge_tflite_deployment(args.model, args.name, args.count)
    else:
        print("INVALID delegate input.")

