import platform

EDGE_FOLDER = "results/edge/"
CPU_FOLDER = "results/cpu/"

EDGETPU_SHARED_LIB = {
    'Linux':   'libedgetpu.so.1',
    'Darwin':   'libedgetpu.1.dylib',
    'Windows':   'edgetpu.dll'
}[platform.system()]


def make_interpreter(model_file):
    """Creates the interpreter object needed to deploy a model onto the tpu.

    Parameters
    ----------
    model_file : String
    Path to the tflite model that will be deployed to the edge tpu.

    Returns
    -------
    tflite.Interpreter Object
    """
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
    """Deploys a group/series of tflite models onto the tpu.

    Parameters
    ----------
    models_folder : String
    Indicates the path to the folder where the compiled tflite models are
    located.

    count : Integer
    Number of times each model will be deployed.

    log_performance : Bool
    Tells if the deployed edge models are to be timed or not.
    """
    from utils import deduce_operations_from_folder

    for model_info in deduce_operations_from_folder(models_folder, beginning="quant_", ending="_edgetpu.tflite"):
        edge_tflite_deployment(
            model_info[0], model_info[1], count, log_performance)


def edge_tflite_deployment(model_file, model_name, count, log_performance=True):
    """The actual deployment on the edge tpu of a single model.

    Parameters
    ----------
    model_file : String
    Indicates the path to the model which will be deployed located.

    model_name : String
    Indicates the name of the operation regarding this model, needed to know
    which folder the results must be saved onto.

    count : Integer
    Number of times this model will be deployed.

    log_performance : Bool
    Tells if the deployed edge models are to be timed or not.
    """
    import time
    import numpy as np
    from utils import create_csv_file

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

    if count > 1:
        print("EDGE DEPLOYMENT")

    for i in range(count):
        print(f"{model_name}: {i+1}/{count}")

        # INFERENCE TIME

        start = time.perf_counter()
        interpreter.invoke()  # Runs the interpreter/inference.

        # END
        inference_time = time.perf_counter() - start

        output_data = interpreter.get_tensor(output_details[0]['index'])

        edge_results.append([i, inference_time])
        print('Inference Time %.1fms' % (inference_time * 1000))

    if (log_performance == True):
        create_csv_file(EDGE_FOLDER, model_name, edge_results)


def cpu_group_tflite_deployment(models_folder, count=5, log_performance=True):
    """Deploys a group/series of tflite models onto the cpu.

    Parameters
    ----------
    models_folder : String
    Indicates the path to the folder where the compiled tflite models are
    located.

    count : Integer
    Number of times each model will be deployed.

    log_performance : Bool
    Tells if the deployed edge models are to be timed or not.
    """
    from utils import deduce_operations_from_folder

    for model_info in deduce_operations_from_folder(models_folder, beginning=None, ending=".tflite"):
        cpu_tflite_deployment(model_info[0], model_info[1], count)


def cpu_tflite_deployment(model_file, model_name, count, log_performance=True):
    """The actual deployment on the cpu of a single model.

    Parameters
    ----------
    model_file : String
    Indicates the path to the model which will be deployed located.

    model_name : String
    Indicates the name of the operation regarding this model, needed to know
    which folder the results must be saved onto.

    count : Integer
    Number of times each model will be deployed.

    log_performance : Bool
    Tells if the deployed edge models are to be timed or not.
    """
    import time
    import tensorflow as tf
    import numpy as np
    from utils import create_csv_file

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

    if count > 1:
        print("CPU DEPLOYMENT")

    for i in range(count):
        print(f"{model_name}: {i+1}/{count}")

        # INFERENCE TIME
        start = time.perf_counter()
        interpreter.invoke()  # Runs the interpreter/inference.

        # END
        inference_time = time.perf_counter() - start

        output_data = interpreter.get_tensor(output_details[0]['index'])

        cpu_results.append([i, inference_time])
        print('Inference Time %.1fms' % (inference_time * 1000))

    if (log_performance == True):
        create_csv_file(CPU_FOLDER, model_name, cpu_results)


def tflite_deployment(count=1000):
    """Manager function responsible for preping and executing the deployment
    of the compiled tflite models.

    Starts the docker, sets the number of times (count) that they will be
    deployed, copies the necessary folders on the docker, deploys the models, 
    copies the results back.

    Parameters
    ----------
    count : Integer
    Indicates the number of times each model will be deplyed.
    """
    import os
    import utils
    from docker import set_docker_globals, docker_copy, docker_exec
    from docker import TO_DOCKER, FROM_DOCKER, HOME

    path_to_tensorDSE = utils.retrieve_folder_path(os.getcwd(), "TensorDSE")
    path_to_docker_results = HOME + \
        "TensorDSE/benchmarking/reading_tflite_model/results/"

    path_to_results = "results/"

    set_docker_globals(count)

    docker_copy(path_to_tensorDSE, TO_DOCKER)

    docker_exec("edge_python_deploy")
    docker_copy(path_to_docker_results + "edge/",
                FROM_DOCKER, path_to_results)

    docker_exec("cpu_python_deploy")
    docker_copy(path_to_docker_results + "cpu/",
                       FROM_DOCKER, path_to_results)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--delegate', default="",
                        required=False, help='cpu or edge_tpu.')

    parser.add_argument('-m', '--model',
                        help='File path to the .tflite file.')

    parser.add_argument('-n', '--name',
                        help='Name of Model/Operation, needed to create corresponding folder name.')

    parser.add_argument('-c', '--count', type=int, default=1,
                        help='Number of times to run inference.')

    parser.add_argument('-g', '--group', type=bool, default=False,
                        help='Flag to determine if its a group deployment or single model deplyment.')

    parser.add_argument('-l', '--log', type=bool, default=False,
                        help='Flag to know if the user wishes to log the performance or not.')

    parser.add_argument('-f', '--group_folder', default="models/compiled/",
                        help='Path to folder where the group of models is located. Only accepted in group mode.')

    parser.add_argument('-M', '--mode', default="Deploy",
                        required=False, help='Debug or Deploy, should only be used with source==Host.')

    parser.add_argument('-s', '--source', default="Docker",
                        required=False, help='Host or Docker.')

    args = parser.parse_args()

    if args.source == "Docker":
        if ("cpu" in args.delegate):
            if (args.group):
                cpu_group_tflite_deployment(
                    args.group_folder, count=args.count, log_performance=(not args.log))
            else:
                cpu_tflite_deployment(args.model, args.name, args.count, log_performance=(not args.log))

        elif ("edge_tpu" in args.delegate):
            if (args.group):
                edge_group_tflite_deployment(
                    args.group_folder, count=args.count, log_performance=(not args.log))
            else:
                if args.name == None:
                    from utils import deduce_filename
                    from plot import deduce_plot_filename
                    args.name = (deduce_plot_filename(deduce_filename(args.model))).split(
                                "quant_")[1]

                edge_tflite_deployment(args.model, args.name,
                                       args.count, log_performance=(not args.log))
        else:
            print("INVALID delegate input.")

    else:
        if args.mode == "All":
            tflite_deployment(args.count)

        elif args.mode == "Debug":
            import os
            from utils import deduce_operations_from_folder, retrieve_folder_path
            from docker import docker_start, docker_exec, docker_copy, TO_DOCKER

            docker_start()
            models_info = deduce_operations_from_folder(args.group_folder,
                                                        beginning="quant_",
                                                        ending="_edgetpu.tflite")

            path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
            docker_copy(path_to_tensorDSE, TO_DOCKER)

            for m_i in models_info:
                inp = input(f"Operation {m_i[1]}, Continue to Next? ")
                if inp == "":
                    continue
                elif inp == "c":
                    print("End.")
                    break
                else:
                    inp = "NONE"
                    while(inp != "" and inp != 'c'):
                        if args.delegate == "cpu":
                            docker_exec("cpu_single_deploy", m_i[0])
                        else:
                            docker_exec("shark_single_edge_deploy", m_i[0])
                        inp = input("Continue: ")

        else:
            print("INVALID delegate input.")
