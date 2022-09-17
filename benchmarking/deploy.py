import argparse
import utils

from typing import Dict

from utils.model import Model

def isCPUavailable() -> bool:
    return True

def isTPUavailable() -> bool:
    # https://github.com/ultralytics/yolov5/issues/5709
    # from pycoral.utils import edgetpu
    # list = edgetpu.list_edge_tpus()
    out = utils.run("lsusb").split("\n")
    line = ""
    for device in out:
        if ("Global" in device) or ("Google" in device):
            line = device
            break

    if not line: # pattern not found in any of the listed devices in lsusb
        return False
    return True

def isGPUavailable() -> bool:
    return False

def MakeInterpreter(model_file:str, library:str):
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
    experimental_delegates = [
        tflite.load_delegate(shared_library, device)
    ]

    return tflite.Interpreter(
        model_path=model_file,
        model_content=None,
        experimental_delegates=experimental_delegates,
    )


def TPUDeploy(m:Model, count:int) -> Model:
    from main import log
    from usb import capture_stream, START_DEPLOYMENT, END_DEPLOYMENT
    from multiprocessing import Process, Queue
    import sys
    import time
    import numpy as np
    import platform

    TPU_LIBRARY = {
        "Linux": "libedgetpu.so.1",
        "Darwin": "libedgetpu.1.dylib",
        "Windows": "edgetpu.dll",
    }[platform.system()]

    results = []
    timers  = []

    interpreter = MakeInterpreter(m.model_path, TPU_LIBRARY)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(
            np.random.random_sample(
                input_details[0]["shape"]),         # input shape
                dtype=input_details[0]["dtype"])    # input dtype

    interpreter.set_tensor(input_details[0]["index"], input_data)
    m.set_input(input_details[0]["shape"], input_details[0]["dtype"])

    signalsQ = Queue()
    for i in range(count):
        p = Process(target=capture_stream, args=(m, signalsQ))
        p.start()

        sig = signalsQ.get()
        if sig == END_DEPLOYMENT:
            log.error("TPU coral usb device wasnt found or id/address variables weren't able to be extracted!")
            p.join()
            break

        start = time.perf_counter()                     # START
        interpreter.invoke()                            # RUNS
        inference_time = time.perf_counter() - start    # END

        _ = interpreter.get_tensor(output_details[0]["index"])  # output data
        results.append([i, inference_time])

        t = signalsQ.get()
        if t:
            timers.append(t)
        p.join()

        sys.stdout.write(f"\r {i+1}/{count} for TPU ran -> {m.model_name}")
        sys.stdout.flush()
    sys.stdout.write("\n")

    m.results = results
    m.timers = timers
    return m


def GPUDeploy(m:Model, count:int) -> Model:
    import time
    import sys
    import tensorflow as tf
    import numpy as np

    GPU_LIBRARY = ""
    results = []

    interpreter = MakeInterpreter(m.model_path, GPU_LIBRARY)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(
            np.random.random_sample(
                input_details[0]["shape"]),         # input shape
                dtype=input_details[0]["dtype"])    # input dtype

    interpreter.set_tensor(input_details[0]["index"], input_data)
    m.set_input(input_details[0]["shape"], input_details[0]["dtype"])

    for i in range(count):
        start = time.perf_counter()                     # START
        interpreter.invoke()                            # RUNS
        inference_time = time.perf_counter() - start    # END

        _ = interpreter.get_tensor(output_details[0]["index"])  # output data
        results.append([i, inference_time])

        sys.stdout.write(f"\r {i+1}/{count} for GPU ran -> {m.model_name}")
        sys.stdout.flush()
    sys.stdout.write("\n")

    m.results = results
    return m

def CPUDeploy(m:Model, count:int) -> Model:
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
            np.random.random_sample(
                input_details[0]["shape"]),         # input shape
                dtype=input_details[0]["dtype"])    # input dtype

    interpreter.set_tensor(input_details[0]["index"], input_data)
    m.set_input(input_details[0]["shape"], input_details[0]["dtype"])

    for i in range(count):
        start = time.perf_counter()                     # START
        interpreter.invoke()                            # RUNS
        inference_time = time.perf_counter() - start    # END

        _ = interpreter.get_tensor(output_details[0]["index"])  # output data
        results.append([i, inference_time])

        sys.stdout.write(f"\r {i+1}/{count} for CPU ran -> {m.model_name}")
        sys.stdout.flush()

    sys.stdout.write("\n")
    m.results = results
    return m

def DeployModels(parent_model:str, layers:list, count=1000)  -> Dict:
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
    from os import listdir
    from os.path import join, isdir, isfile
    from main import log
    from main import LAYERS_FOLDER, COMPILED_MODELS_FOLDER

    models = {}
    models["cpu"]   = []
    models["gpu"]   = []
    models["tpu"]   = []
    models["count"] = count

    # regular quantized tflite files for cpu
    if not isCPUavailable():
        log.info(f"CPU is NOT available on this machine!")
    else:
        log.warning(f"CPU is available on this machine!")
        for d in listdir(LAYERS_FOLDER):
            if isdir(join(LAYERS_FOLDER, d)) and (d in layers):
                model_name = d
                model_path = join(LAYERS_FOLDER, d, "quant", f"quant_{model_name}.tflite")
                log.info(f"Deploying layer/operation {model_name} onto the cpu")

                m = CPUDeploy(Model(model_path, "cpu", parent_model), count)
                models["cpu"].append(m)


    # regular quantized tflite files for gpu
    if not isGPUavailable():
        log.info(f"GPU is NOT available on this machine!")
    else:
        log.warning(f"GPU is available on this machine!")
        for d in listdir(LAYERS_FOLDER):
            if isdir(join(LAYERS_FOLDER, d)) and (d in layers):
                model_name = d
                model_path = join(LAYERS_FOLDER, d, "quant", f"quant_{model_name}.tflite")
                log.info(f"Deploying layer/operation {model_name} onto the gpu")

                m = GPUDeploy(Model(model_path, "gpu", parent_model), count)
                models["gpu"].append(m)

    # edge compiled quantized tflite files tpu
    if not isTPUavailable():
        log.info(f"TPU is NOT available on this machine!")
    else:
        log.warning(f"TPU is available on this machine!")
        for f in listdir(COMPILED_MODELS_FOLDER):
            if isfile(join(COMPILED_MODELS_FOLDER, f)) and f.endswith(".tflite"):
                model_name = (f.split("quant_")[1]).split("_edgetpu.tflite")[0]
                if model_name not in layers:
                    continue # skip this iteration
                model_path = join(COMPILED_MODELS_FOLDER, f)
                log.info(f"Deploying layer/operation {model_name} onto the tpu")

                m = TPUDeploy(Model(model_path, "tpu", parent_model), count)
                models["tpu"].append(m)

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
        "-f", "--form", default="Deploy", required=False, help="Single or Group."
    )

    parser.add_argument(
        "-d", "--delegate", default="", required=False, help="cpu or edge."
    )

    parser.add_argument(
        " -m", "--model", help="File path to the .tflite file.")

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

    from main import  LAYERS_FOLDER, COMPILED_MODELS_FOLDER

    args = GetArgs()

    delegators = {
        "cpu"   : CPUDeploy,
        "gpu"   : GPUDeploy,
        "tpu"   : TPUDeploy,
    }

    if args.form == "Single":
        delegator = delegators.get(args.delegate, None)
        if not (delegator == None):
            delegator(Model(args.model, args.delegate), args.count)

    if args.form == "Group":
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
                        delegator(Model(model_path, "tpu"), args.count)
        else:
            if delegator:
                for d in listdir(LAYERS_FOLDER):
                    if isdir(d):
                        model_name = d
                        model_path = join(os.getcwd(),
                                    d,
                                    "quant",
                                    f"quant_{model_name}.tflite")
                        delegator(Model(model_path, args.delegate), args.count)

    else:
        raise Exception(f"Invalid mode: {args.mode}")
