import platform
from utils import *

EDGE_FOLDER = "results/edge/"
CPU_FOLDER = "results/cpu/"

EDGETPU_SHARED_LIB = {
    'Linux':   'libedgetpu.so.1',
    'Darwin':   'libedgetpu.1.dylib',
    'Windows':   'edgetpu.dll'
}[platform.system()]


def deduce_operation_from_file(tflite_file, beginning=None, ending=None):
    f = tflite_file
    op = ""

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
    from os.path import isfile, isdir, join

    tflite_models_info = []

    for f_1 in listdir(models_folder):
        f_1_path = models_folder + f_1
        if isdir(f_1_path):
            for f_2 in listdir(f_1_path):
                f_2_path = f_1_path + "/" + f_2
                if (isfile(f_2_path) and f_2.endswith(".tflite")):
                    op = deduce_operation_from_file(
                        f_2, beginning=beginning, ending=ending)
                    tflite_models_info.append([f_2_path, op])

        elif isfile(f_1_path):
            if (isfile(f_1_path) and f_1.endswith(".tflite")):
                op = deduce_operation_from_file(
                    f_1, beginning=beginning, ending=ending)
                tflite_models_info.append([f_1_path, op])

    return tflite_models_info


def make_interpreter(model_file):

    import tflite_runtime.interpreter as tflite

    model_file, *device = model_file.split('@')

    device = {'device': device[0]} if device else {}
    shared_library = EDGETPU_SHARED_LIB
    experimental_delegates = [tflite.load_delegate(
        shared_library, device)]  # Returns loaded Delegate object

    return tflite.Interpreter(model_path=model_file,
                              model_content=None,
                              experimental_delegates=experimental_delegates)


def edge_group_tflite_deployment(models_folder, count=5, log_performance=True):

    for model_info in deduce_operations_from_folder(models_folder, beginning="quant_", ending="_edgetpu.tflite"):
        edge_tflite_deployment(
            model_info[0], model_info[1], count, log_performance)


def edge_tflite_deployment(model_file, model_name, count, log_performance=True):

    import time
    import numpy as np

    edge_results = []

    # Creates Interpreter Object.
    interpreter = make_interpreter(model_file)
    interpreter.allocate_tensors()

    # This is where model specific inputs/labels are fed to the model in
    # order to be run correctly.

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    input_data = np.array(np.random.random_sample(
        input_shape), dtype=input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    for i in range(count):

        # INFERENCE TIME

        start = time.perf_counter()
        interpreter.invoke()  # Runs the interpreter/inference.

        # END

        inference_time = time.perf_counter() - start
        output_data = interpreter.get_tensor(output_details[0]['index'])

        edge_results.append([i, inference_time])

    if (log_performance == True):
        create_csv_file(EDGE_FOLDER, model_name, edge_results)


def cpu_group_tflite_deployment(models_folder, count=5, log_performance=True):

    for model_info in deduce_operations_from_folder(models_folder, beginning=None, ending=".tflite"):
        cpu_tflite_deployment(model_info[0], model_info[1], count)


def cpu_tflite_deployment(model_file, model_name, count, log_performance=True):
    import time
    import tensorflow as tf
    import numpy as np

    cpu_results = []

    # Creates Interpreter Object.
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    input_data = np.array(np.random.random_sample(
        input_shape), dtype=input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    for i in range(count):

        # INFERENCE TIME

        start = time.perf_counter()
        interpreter.invoke()  # Runs the interpreter/inference.

        inference_time = time.perf_counter() - start
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # END

        cpu_results.append([i, inference_time])

    if (log_performance == True):
        create_csv_file(CPU_FOLDER, model_name, cpu_results)


def full_tflite_deployment(count=1000):
    import os
    import utils
    import docker
    from docker import TO_DOCKER, FROM_DOCKER, home

    path_to_tensorDSE = utils.retrieve_folder_path(os.getcwd(), "TensorDSE")
    path_to_docker_results = home + \
        "TensorDSE/benchmarking/reading_tflite_model/results/"
    path_to_results = "results/"

    docker.set_globals(count)

    docker.docker_copy(path_to_tensorDSE, TO_DOCKER)

    docker.docker_exec("edge_python_deploy")
    docker.docker_copy(path_to_docker_results + "edge/",
                       FROM_DOCKER, path_to_results)

    docker.docker_exec("cpu_python_deploy")
    docker.docker_copy(path_to_docker_results + "cpu/",
                       FROM_DOCKER, path_to_results)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--delegate',
                        required=True, help='cpu or edge_tpu.')

    parser.add_argument('-m', '--model',
                        help='File path to the .tflite file.')

    parser.add_argument('-n', '--name',
                        help='Name of Model/Operation, needed to create corresponding folder name.')

    parser.add_argument('-c', '--count', type=int, default=5,
                        help='Number of times to run inference.')

    parser.add_argument('-g', '--group', type=bool, default=False,
                        help='Flag to determine if its a group deployment or single model deplyment.')

    parser.add_argument('-l', '--log', type=bool, default=False,
                        help='Flag to know if the user wishes to log the performance or not.')

    parser.add_argument('-f', '--group_folder', default="",
                        help='Path to folder where the group of models is located. Only accepted in group mode.')

    args = parser.parse_args()

    if ("cpu" in args.delegate):
        if (args.group):
            cpu_group_tflite_deployment(
                args.group_folder, count=args.count, log_performance=(not args.log))
        else:
            cpu_tflite_deployment(args.model, args.name, args.count, args.log)

    elif ("edge_tpu" in args.delegate):
        if (args.group):
            edge_group_tflite_deployment(
                args.group_folder, count=args.count, log_performance=(not args.log))
        else:
            edge_tflite_deployment(args.model, args.name,
                                   args.count, log_performance=(not args.log))
    else:
        print("INVALID delegate input.")
