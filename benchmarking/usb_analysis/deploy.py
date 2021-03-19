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


def edge_group_tflite_deployment(models_folder, count=5):
    """Deploys a group/series of tflite models onto the tpu.

    Parameters
    ----------
    models_folder : String
    Indicates the path to the folder where the compiled tflite models are
    located.

    count : Integer
    Number of times each model will be deployed.
    """
    from utils import deduce_operations_from_folder

    for model_info in deduce_operations_from_folder(models_folder, beginning="quant_", ending="_edgetpu.tflite"):
        edge_tflite_deployment(
                count, model_info[0], model_info[1])


def edge_tflite_deployment(count, model_file, model_name=None):
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
    """
    import time
    import logging
    import numpy as np
    from utils import create_csv_file, deduce_filename, deduce_operation_from_file

    edge_results = []
    if model_name == None:
        model_name = model_file.split("/")[model_file.count("/")]
        model_name = deduce_operation_from_file(
                model_name, beginning="quant_", ending="_edgetpu")

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

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    if count > 1: 
        log.info("EDGE DEPLOYMENT...")

    for i in range(count):
        log.info(f"{model_name}: {i+1}/{count}")

        # INFERENCE TIME

        start = time.perf_counter()
        interpreter.invoke()  # Runs the interpreter/inference.

        # END
        inference_time = time.perf_counter() - start

        output_data = interpreter.get_tensor(output_details[0]['index'])

        edge_results.append([i, inference_time])
        # log.info('Inference Time %.1f us' % (inference_time * 10**6))

    create_csv_file(EDGE_FOLDER, model_name, edge_results)


def cpu_group_tflite_deployment(models_folder, count=5):
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
        cpu_tflite_deployment(count, model_info[0], model_info[1])


def cpu_tflite_deployment(count, model_file, model_name=None):
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

    """
    import time
    import logging
    import tensorflow as tf
    import numpy as np
    from utils import create_csv_file, deduce_filename, deduce_operation_from_file

    cpu_results = []
    if model_name == None:
        model_name = model_file.split("/")[model_file.count("/")]
        model_name = deduce_operation_from_file(model_name, ending=".tflite")

    # Creates Interpreter Object.
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    input_data = np.array(
            np.random.random_sample(input_shape), dtype=input_dtype)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    if count > 1:
        log.info("CPU DEPLOYMENT...")

    for i in range(count):
        log.info(f"{model_name}: {i+1}/{count}")

        # INFERENCE TIME
        start = time.perf_counter()
        interpreter.invoke()  # Runs the interpreter/inference.

        # END
        inference_time = time.perf_counter() - start

        output_data = interpreter.get_tensor(output_details[0]['index'])

        cpu_results.append([i, inference_time])
        # log.info('Inference Time %.1f us\n' % (inference_time * 10**6))

    create_csv_file(CPU_FOLDER, model_name, cpu_results)


def tflite_deployment(quant_folder, tflite_folder, count=1000):
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
    from docker import docker_copy, docker_exec
    from docker import TO_DOCKER, FROM_DOCKER, HOME

    path_to_tensorDSE = utils.retrieve_folder_path(os.getcwd(), "TensorDSE")
    path_to_docker_results = HOME + \
        "TensorDSE/benchmarking/usb_analysis/results/"

    path_to_results = "results/"

    docker_copy(path_to_tensorDSE, TO_DOCKER)

    docker_exec("edge_deploy", quant_folder, count)
    docker_copy(path_to_docker_results + "edge/",
                FROM_DOCKER, path_to_results)

    docker_exec("cpu_deploy", tflite_folder, count)
    docker_copy(path_to_docker_results + "cpu/",
                       FROM_DOCKER, path_to_results)


if __name__ == '__main__':
    """Entry point to execute this script.

    Flags
    ---------
    -m or --mode
    Mode in which the script should run. Group, Single or Debug.
        Group is supposed to be used to deploy a group of models in sequence(lazy).
        Single deploys a single model.
        Debug simply deploys model on the edge tpu specified by user input within
        the debug mode.

    -d or --delegate
        Specifies which hardware should be used to delegate a model.
            cpu => cpu
            edge => edgetpu

    -t or --target
        Should be used in conjunction with the 'Single' mode, where -t must be followed
        by the path to the model (target) that will be deployed.

    -f or --folder
        Should be used in conjunction with the 'Group' mode, where -f must be followed
        by the path to a folder containing all models to be deployed.

    -c or --count
        Should be followed by the number of times one wishes to deploy the group of models
        or the single target (Depends on mode).
    """
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--mode', default="Deploy",
                        required=False, help='Group, Single, Debug.')

    parser.add_argument('-d', '--delegate', default="",
                        required=False, help='cpu or edge.')

    parser.add_argument('-t', '--target',
                        help='File path to the .tflite file.')

    parser.add_argument('-f', '--folder', default="models/compiled/",
                        help='Path to folder where the group of models is located.')

    parser.add_argument('-c', '--count', type=int, default=1,
                        help='Number of times to run inference.')

    args = parser.parse_args()

    if (args.mode == "Group"):
        if (args.delegate == "cpu"):
            cpu_group_tflite_deployment(args.folder, count=args.count)
        else:
            edge_group_tflite_deployment(args.folder, count=args.count)

    elif (args.mode == "Single"):
        if (args.delegate == "cpu"):
            cpu_tflite_deployment(args.count, args.target)
        else:
            edge_tflite_deployment(args.count, args.target)

    elif args.mode == "Deploy-Group":
        import os
        for model in os.listdir(args.folder):
            if args.delegate == "cpu": target = f"{args.folder}{model}/{model}.tflite"
            else: target = f"{args.folder}{model}"
            os.system(
            f"python deploy.py -m Deploy-Single -d {args.delegate} -t {target} -c {str(args.count)}")


    elif args.mode == "Deploy-Single":
        import os
        from utils import retrieve_folder_path
        from utils import deduce_operation_from_file, extend_directory
        from docker import docker_exec, docker_copy, HOME, TO_DOCKER, FROM_DOCKER

        path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
        path_to_docker_results = HOME + \
        "TensorDSE/benchmarking/usb_analysis/results/"

        docker_copy(path_to_tensorDSE, TO_DOCKER)
        op = args.target.split("/")[args.target.count("/")]
        if args.delegate == "cpu":
            op = deduce_operation_from_file(op, ending=".tflite")
            extend_directory("results/cpu/", f"{op}")
            path_to_results = f"results/cpu/{op}/"
            docker_exec("cpu_single_deploy", args.target, args.count)
            docker_copy(f"{path_to_docker_results}cpu/{op}/Results.csv",
                            FROM_DOCKER, path_to_results)
            docker_exec("remove", "TensorDSE")
        else:
            op = deduce_operation_from_file(op, beginning="quant_", ending="_edgetpu")
            extend_directory("results/edge/", f"{op}")
            path_to_results = f"results/edge/{op}/"
            docker_exec("edge_single_deploy", args.target, args.count)
            docker_copy(f"{path_to_docker_results}edge/{op}/Results.csv",
                            FROM_DOCKER, path_to_results)
            docker_exec("remove", "TensorDSE")

    elif args.mode == "Debug":
        import os
        from utils import deduce_operations_from_folder, retrieve_folder_path
        from docker import docker_start, docker_exec, docker_copy, TO_DOCKER

        docker_start()
        models_info = deduce_operations_from_folder(
                args.folder, beginning="quant_",ending="_edgetpu.tflite")

        path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
        docker_copy(path_to_tensorDSE, TO_DOCKER)

        for m_i in models_info:
            inp = input(f"Operation {m_i[1]}, Continue to Next? ")
            if inp == "": continue
            elif inp == "c": break
            else:
                inp = "NONE"
                while(inp != "" and inp != 'c'):
                    if args.delegate == "cpu":
                        docker_exec("cpu_single_deploy", m_i[0], args.count)
                    else:
                        docker_exec("edge_single_deploy", m_i[0], args.count)
                    inp = input("Continue: ")

        docker_exec("remove", "TensorDSE")

    else:
        print("INVALID delegate input.")
