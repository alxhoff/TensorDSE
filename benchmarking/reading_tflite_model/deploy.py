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


def deduce_operation_from_file(tflite_file, beginning=None, ending=None):
    f = tflite_file
    op=""

    if(beginning and not ending):
        if beginning in tflite_file:
            op = f.split(beginning)[1]

    elif(beginning and ending):
        if beginning in tflite_file and ending in tflite_file:
            op = f.split(beginning)[1]
            op = op.split(ending)[0]

    elif(ending and not beginning):
        if ending in tflite_file:
            op = f.split(ending)[0]

    return op

def deduce_operations_from_folder(models_folder, beginning=None, ending=None):
    import os
    from os import listdir
    from os.path import isfile,isdir,join

    tflite_models_info = [] 

    for F_1 in listdir(models_folder):
        F_1_PATH = models_folder + F_1
        if isdir(F_1_PATH):
            for F_2 in listdir(F_1_PATH):
                F_2_PATH = F_1_PATH + "/" +  F_2
                if (isfile(F_2_PATH) and F_2.endswith(".tflite")):
                    OP = deduce_operation_from_file(F_2, beginning=beginning, ending=ending)
                    tflite_models_info.append([F_2_PATH, OP])

        elif isfile(F_1_PATH):
            if (isfile(F_1_PATH) and F_1.endswith(".tflite")):
                OP = deduce_operation_from_file(F_1, beginning=beginning, ending=ending)
                tflite_models_info.append([F_1_PATH, OP])

    return tflite_models_info

def make_interpreter(model_file):

    import tflite_runtime.interpreter as tflite

    model_file, *device = model_file.split('@')

    device = {'device': device[0]} if device else {}
    shared_library = EDGETPU_SHARED_LIB
    experimental_delegates = [ tflite.load_delegate(shared_library, device) ]   #Returns loaded Delegate object

    return tflite.Interpreter(model_path=model_file, 
                              model_content=None, 
                              experimental_delegates=experimental_delegates)

def edge_group_tflite_deployment(models_folder, count=5):

    for model_info in deduce_operations_from_folder(models_folder, beginning="quant_", ending="_edgetpu.tflite"):
        edge_tflite_deployment(model_info[0], model_info[1], count)

def edge_tflite_deployment(model_file, model_name, count):

    import time
    import numpy as np

    EDGE_RESULTS = []

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

        """ INFERENCE TIME
        """
        start = time.perf_counter()
        interpreter.invoke()                                                    #Runs the interpreter/inference, be sure
                                                                                #to have set the input sizes and allocate 
                                                                                #tensors.
        """ END
        """
        inference_time = time.perf_counter() - start
        output_data = interpreter.get_tensor(output_details[0]['index'])

        EDGE_RESULTS.append([i, inference_time])

    create_csv_file(edge_folder, model_name, EDGE_RESULTS)

def cpu_group_tflite_deployment(models_folder, count=5):

    for model_info in deduce_operations_from_folder(models_folder, beginning=None, ending=".tflite"):
        cpu_tflite_deployment(model_info[0], model_info[1], count)


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


def full_tflite_deployment():
    import os
    from utils import *
    from compile import *

    path_to_TensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
    docker_copy(path_to_TensorDSE, TO_DOCKER)
    docker_exec("python")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--delegate', required=True, help='cpu, gpu or edge_tpu.')
    parser.add_argument('-m', '--model', help='File path to the .tflite file.')
    parser.add_argument('-n', '--name',  help='Name of Model/Operation, needed to create corresponding folder name.')
    parser.add_argument('-c', '--count', type=int, default=5,help='Number of times to run inference.')
    parser.add_argument('-g', '--group', type=bool, default=False,help='Flag to determine if its a group deployment or single model deplyment.')
    parser.add_argument('-f', '--group_folder', default="",help='Path to folder where the group of models is located. Only accepted in group mode.')

    args = parser.parse_args()

    if ("cpu" in args.delegate):
        if (args.group):
            cpu_group_tflite_deployment(args.group_folder, count=args.count)
        else:
            cpu_tflite_deployment(args.model, args.name, args.count)

    elif ("edge_tpu" in args.delegate):
        if (args.group):
            edge_group_tflite_deployment(args.group_folder, count=args.count)
        else:
            edge_tflite_deployment(args.model, args.name, args.count)
    else:
        print("INVALID delegate input.")

