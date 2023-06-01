import argparse
import utils

from typing import Dict, Tuple

from utils.model import Model


def isCPUavailable() -> bool:
    return True


def isTPUavailable() -> bool:
    # https://github.com/ultralytics/yolov5/issues/5709
    # from pycoral.utils import edgetpu
    # list = edgetpu.list_edge_tpus()
    out = utils.run("lsusb").split("\n")
    for device in out:
        if ("Global" in device) or ("Google" in device):
            return True
    return False


def isGPUavailable() -> Tuple[bool, str]:
    out = utils.run("lshw -numeric -C display").split("\n")
    for line in out:
        if "vendor" in line:
            gpu = line.split()[1].lower()
            if "intel" in line.lower():
                return False, gpu
            return True, gpu
    return False, ""


def MakeInterpreterTPU(model_file: str, library: str):
    """Creates the interpreter object needed to deploy a model onto the tpu.

    Parameters
    ----------
    model_file : String
    Path to the tflite model that will be deployed to the edge tpu.

    system : String

    Returns
    -------
    tflite.Interpreter Object
    """
    # https://github.com/ultralytics/yolov5/issues/5709
    # https://github.com/google-coral/pycoral/issues/57
    # import tensorflow as tf
    import tflite_runtime.interpreter as tflite

    model_file, *device = model_file.split("@")

    device = {"device": device[0]} if device else {}
    shared_library = library
    experimental_delegates = [tflite.load_delegate(shared_library, device)]

    return tflite.Interpreter(
        model_path=model_file,
        model_content=None,
        experimental_delegates=experimental_delegates,
    )


def MakeInterpreterGPU(model_file: str, library: str):
    """Creates the interpreter object needed to deploy a model onto the tpu.

    Parameters
    ----------
    model_file : String
    Path to the tflite model that will be deployed to the edge tpu.

    system : String

    Returns
    -------
    tflite.Interpreter Object
    """
    import tensorflow as tf

    experimental_delegates = [tf.lite.experimental.load_delegate(library=library)]

    return tf.lite.Interpreter(
        model_path=model_file,
        model_content=None,
        experimental_delegates=experimental_delegates,
    )


def TPUDeploy(m: Model, count: int, timeout: int = 10) -> Model:
    from multiprocessing import Process, Queue
    import queue
    from main import log
    from utils.log import Log
    from utils.usb import END_DEPLOYMENT
    from utils.usb.usb import capture_stream
    import sys
    import time
    import numpy as np
    import platform

    DEPLOY_WAIT_TIME = 10
    TPU_LIBRARY = {
        "Linux": "libedgetpu.so.1",
        "Darwin": "libedgetpu.1.dylib",
        "Windows": "edgetpu.dll",
    }[platform.system()]

    results = []
    timers = []

    interpreter = MakeInterpreterTPU(m.model_path, TPU_LIBRARY)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(
        np.random.random_sample(input_details[0]["shape"]),  # input shape
        dtype=input_details[0]["dtype"],
    )  # input dtype

    interpreter.set_tensor(input_details[0]["index"], input_data)
    m.set_input(input_details[0]["shape"], input_details[0]["dtype"])

    for i in range(count):
        signalsQ = Queue()
        dataQ = Queue()

        p = Process(
            target=capture_stream,
            args=(signalsQ, dataQ, timeout, Log(f"results/{m.model_name}_USB.log")),
        )
        p.start()

        sig = signalsQ.get()
        if sig == END_DEPLOYMENT:
            p.join()
            break

        time.sleep(DEPLOY_WAIT_TIME)
        start = time.perf_counter()  # START
        interpreter.invoke()  # RUNS
        inference_time = time.perf_counter() - start  # END

        _ = interpreter.get_tensor(output_details[0]["index"])  # output data
        results.append(inference_time)

        data = dataQ.get()
        if not data == {}:
            timers.append(data)
        p.join()

        sys.stdout.write(f"\r {i+1}/{count} for TPU ran -> {m.model_name}")
        sys.stdout.flush()

    sys.stdout.write("\n")

    m.results = results
    m.timers = timers
    return m


def GPUDeploy(m: Model, count: int) -> Model:
    import time
    import sys
    import tensorflow as tf
    import numpy as np

    GPU_LIBRARY = "/home/lib/tf2.9/libtensorflowlite_gpu_delegate.so"
    results = []

    interpreter = MakeInterpreterGPU(m.model_path, GPU_LIBRARY)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(
        np.random.random_sample(input_details[0]["shape"]),  # input shape
        dtype=input_details[0]["dtype"],
    )  # input dtype

    interpreter.set_tensor(input_details[0]["index"], input_data)
    m.set_input(input_details[0]["shape"], input_details[0]["dtype"])

    for i in range(count):
        start = time.perf_counter()  # START
        interpreter.invoke()  # RUNS
        inference_time = time.perf_counter() - start  # END

        _ = interpreter.get_tensor(output_details[0]["index"])  # output data
        results.append(inference_time)

        sys.stdout.write(f"\r {i+1}/{count} for GPU ran -> {m.model_name}")
        sys.stdout.flush()
    sys.stdout.write("\n")

    m.results = results
    return m


def CPUDeploy(m: Model, count: int) -> Model:
    import time
    import sys
    import numpy as np
    import tensorflow as tf

    results = []

    # Interpreter Object.
    interpreter = tf.lite.Interpreter(model_path=m.model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(
        np.random.random_sample(input_details[0]["shape"]),  # input shape
        dtype=input_details[0]["dtype"],
    )  # input dtype

    interpreter.set_tensor(input_details[0]["index"], input_data)
    m.set_input(input_details[0]["shape"], input_details[0]["dtype"])

    for i in range(count):
        start = time.perf_counter()  # START
        interpreter.invoke()  # RUNS
        inference_time = time.perf_counter() - start  # END

        _ = interpreter.get_tensor(output_details[0]["index"])  # output data
        results.append(inference_time)

        sys.stdout.write(f"\r {i+1}/{count} for CPU ran -> {m.model_name}")
        sys.stdout.flush()

    sys.stdout.write("\n")
    m.results = results
    return m


def DeployModels(
    hardware_list:list, model_summary:str, count:int
) -> Dict:
    """Manager function responsible for preping and executing the deployment
    of the compiled tflite models.

    Starts the docker, sets the number of times (count) that they will be
    deployed, copies the necessary folders on the docker, deploys the models,
    copies the results back.

    Parameters
    ----------
    count : Integer
    hardware_summary : Path to the hardware summary file that is to be used to
    check which devices need to be benchmarked, if None then all are benchmarked
    Indicates the number of times each model will be deplyed.
    """
    import os, sys
    from os import listdir
    from os.path import join, isdir, isfile
    #from main import log
    from main import LAYERS_FOLDER, COMPILED_MODELS_FOLDER
    #from .usb import init_usbmon
    import numpy as np
    

    from utils.model_lab.split import LAYERS_DIR, COMPILED_DIR
    from utils.model_lab.utils import LayerDetails
    from utils.model_lab.logger import log
    from backend.distributed_inference import distributed_inference

    models = {}
    models["cpu"] = []
    models["gpu"] = []
    models["tpu"] = []
    models["count"] = count

    details = LayerDetails(model_summary)

    # regular quantized tflite files for cpu
    if not isCPUavailable():
        log.warning(f"CPU is NOT available on this machine!")
    elif "cpu" not in hardware_list:
        log.info("No CPU cores in hardware summary, skipping benchmarking")
    else:
        log.info(f"CPU is available on this machine!")
        for layer_details in details.layers:
            # TODO: CPU Deploy
            details.ReadLayerDetails(layer_details)
            layer_file = "submodel_{0}_{1}_bm.tflite".format(details.index, details.name)
            input_data_vector = np.zeros(details.GetTensorSize("Input")).astype(np.int8)
            output_data_vector = np.zeros(details.GetTensorSize("Output")).astype(np.int8)
            print("##########################################################")
            print("Benchmarking: {}\n Target: CPU".format(layer_file))
            mean_inf = distributed_inference(
                os.path.join(LAYERS_DIR, layer_file),
                input_data_vector,
                output_data_vector, 
                len(input_data_vector), 
                len(output_data_vector), 
                "CPU", 
                1
            )
            print(mean_inf)
            print("########################################################## \n")
            #models["cpu"].append(m)

    # regular quantized tflite files for gpu
    available, gpu = isGPUavailable()
    if not available:
        log.warning(f"GPU is NOT available on this machine! Type: {gpu}")
    elif "gpu" not in hardware_list:
        log.info("No GPUs in hardware summary, skipping benchmarking")
    else:
        log.info(f"GPU is available on this machine! Type: {gpu}")
        for layer_details in details.layers:
            # TODO: GPU Deploy
            details.ReadLayerDetails(layer_details)
            layer_file = "submodel_{0}_{1}_bm.tflite".format(details.index, details.name)
            input_data_vector = np.zeros(details.GetTensorSize("Input")).astype(np.int8)
            output_data_vector = np.zeros(details.GetTensorSize("Output")).astype(np.int8)
            print("##########################################################")
            print("Benchmarking: {}\n Target: GPU".format(layer_file))
            mean_inf = distributed_inference(
                os.path.join(LAYERS_DIR, layer_file),
                input_data_vector,
                output_data_vector, 
                len(input_data_vector), 
                len(output_data_vector), 
                "GPU", 
                1
            )
            print(mean_inf)
            print("########################################################## \n")
            #models["gpu"].append(m)

    # edge compiled quantized tflite files tpu
    if not isTPUavailable():
        log.warning(f"TPU is NOT available on this machine!")
    elif "tpu" not in hardware_list:
        log.info("No TPUs in hardware summary, skipping benchmarking")
    else:
        log.info(f"TPU is available on this machine!")
        #if init_usbmon():
        #    log.info("Needed to introduce usbmon module")
        #else:
        #    log.info("usbmon module already present")

        for layer_details in details.layers:
            # TODO: CPU Deploy
            details.ReadLayerDetails(layer_details)
            layer_file = "submodel_{0}_{1}_bm_edgetpu.tflite".format(details.index, details.name)
            input_data_vector = np.zeros(details.GetTensorSize("Input")).astype(np.int8)
            output_data_vector = np.zeros(details.GetTensorSize("Output")).astype(np.int8)
            print("##########################################################")
            print("Benchmarking: {}\n Target: TPU".format(layer_file))
            mean_inf = distributed_inference(
                os.path.join(COMPILED_DIR, layer_file),
                input_data_vector,
                output_data_vector, 
                len(input_data_vector), 
                len(output_data_vector), 
                "TPU", 
                1
            )
            print(mean_inf)
            print("########################################################## \n")
            #models["tpu"].append(m)
            

    return models


def GetArgs() -> argparse.Namespace:
    """Argument parser, returns the Namespace containing all of the arguments.
    :raises: None

    :rtype: argparse.Namespace
    """
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-f", "--form", default="Deploy", required=True, help="Single or Group."
    )

    parser.add_argument(
        "-d", "--delegate", default="", required=True, help="cpu or edge."
    )

    parser.add_argument(
        "-m", "--model", required=True, help="File path to the .tflite file."
    )

    parser.add_argument(
        "-p", "--parent", default="", required=False, help="parent model"
    )

    parser.add_argument(
        "-c", "--count", type=int, default=1, help="Number of times to run inference."
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    """Entry point to execute this script.

    Flags
    ---------
    -f or --form
    Form in which the script should run. Group or Single
        Single deploys a single model.
        Group is supposed to be used to deploy a group of models in sequence.

    -d or --delegate
        Specifies which hardware should be used to delegate a model.
            cpu => cpu
            gpu => gpu
            tpu => edgetpu

    -m or --model
        Should be used in conjunction with the 'Single' form.

    -c or --count
        Should be followed by the number of times one wishes to deploy the group of models
        or a single model (Depends on form).
    """

    from main import LAYERS_FOLDER, COMPILED_MODELS_FOLDER
    from analysis import AnalyzeLayerResults

    args = GetArgs()

    delegators = {
        "cpu": CPUDeploy,
        "gpu": GPUDeploy,
        "tpu": TPUDeploy,
    }

    if args.form == "single":
        delegator = delegators.get(args.delegate, None)
        if not (delegator == None):
            m = delegator(Model(args.model, args.delegate), args.count)
            AnalyzeLayerResults(m, args.delegate)

    elif args.form == "group":
        import os
        from os import listdir
        from os.path import join, isdir, isfile

        delegator = delegators.get(args.delegate, None)
        if args.delegate == "tpu":
            if delegator:
                for f in listdir(COMPILED_MODELS_FOLDER):
                    if isfile(f) and f.endswith(".tflite"):
                        model_name = (f.split("quant_")[1]).split("edgetpu.tflite")[0]
                        model_path = join(os.getcwd(), COMPILED_MODELS_FOLDER, f)
                        m = delegator(Model(model_path, args.delegate), args.count)
                        AnalyzeLayerResults(m, args.delegate)
        else:
            if delegator:
                for d in listdir(LAYERS_FOLDER):
                    if isdir(d):
                        model_name = d
                        model_path = join(
                            os.getcwd(), d, "quant", f"quant_{model_name}.tflite"
                        )
                        m = delegator(Model(model_path, args.delegate), args.count)
                        AnalyzeLayerResults(m, args.delegate)

    else:
        raise Exception(f"Invalid mode: {args.mode}")
