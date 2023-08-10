import argparse
import utils

from typing import Dict, Tuple

from utils.model import Model

def GetArraySizeFromShape(shape: list) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size

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


def TPUDeploy(m: Model, count: int, timeout: int = 10) -> Model:
    from multiprocessing import Process, Queue
    from utils.log import Log
    from utils.usb import END_DEPLOYMENT
    from utils.usb.usb import capture_stream
    import sys
    import time
    import numpy as np

    from backend.distributed_inference import distributed_inference

    DEPLOY_WAIT_TIME = 10

    results = []
    timers = []

    input_size  = GetArraySizeFromShape(m.input_shape)
    output_size = GetArraySizeFromShape(m.output_shape)

    for i in range(count):
        signalsQ = Queue()
        dataQ = Queue()

        p = Process(
            target=capture_stream,
            args=(signalsQ, dataQ, timeout, Log(f"resources/results/layer_{m.index}_{m.model_name}_USB.log")),
        )
        p.start()

        sig = signalsQ.get()
        if sig == END_DEPLOYMENT:
            p.join()
            break

        input_data_vector = np.array(
            np.random.random_sample(input_size),
            dtype=m.get_np_dtype(m.input_datatype),
        )
        output_data_vector = np.zeros(output_size).astype(m.get_np_dtype(m.output_datatype))
        inference_times_vector = np.zeros(count).astype(np.uint32)

        time.sleep(DEPLOY_WAIT_TIME)

        mean_inference_time = distributed_inference(
            m.model_path,
            input_data_vector,
            output_data_vector, 
            inference_times_vector,
            len(input_data_vector), 
            len(output_data_vector), 
            "TPU", 
            1
        )

        results.append(mean_inference_time)

        try:
            data = dataQ.get(block=False, timeout=0.1)
            if dataQ.empty:
                raise Exception
        except Exception as e:
            print("DATA QUEUE IS EMPTY!")
            data = None

        if not data == {}:
            timers.append(data)
            
        if p.is_alive():
            p.join(timeout=0.1)

        sys.stdout.write(f"\r {i+1}/{count} for TPU ran -> {m.model_name}")
        sys.stdout.flush()

    sys.stdout.write("\n")

    m.results = results
    m.timers = timers
    return m


def BenchmarkLayer(m: Model, count: int, hardware_target: str) -> Model:
    import os
    import numpy as np

    from utils.model_lab.split import LAYERS_DIR, COMPILED_DIR
    from backend.distributed_inference import distributed_inference


    if (hardware_target == "TPU"):
        m.model_path = os.path.join(COMPILED_DIR, "submodel_{0}_{1}_bm_edgetpu.tflite".format(m.details["index"], m.details["type"]))
        m = TPUDeploy(m=m, count=count)
    else:
        
        m.model_path = os.path.join(LAYERS_DIR, "submodel_{0}_{1}_bm.tflite".format(m.details["index"], m.details["type"]))

        input_size  = GetArraySizeFromShape(m.input_shape)
        output_size = GetArraySizeFromShape(m.output_shape)

        input_data_vector = np.zeros(input_size).astype(m.get_np_dtype(m.input_datatype))
        output_data_vector = np.zeros(output_size).astype(m.get_np_dtype(m.output_datatype))
        inference_times_vector = np.zeros(count).astype(np.uint32)

        print(m.model_path)
        mean_inference_time = distributed_inference(
            m.model_path,
            input_data_vector,
            output_data_vector, 
            inference_times_vector,
            len(input_data_vector), 
            len(output_data_vector), 
            hardware_target, 
            count
        )
        print(mean_inference_time)
        print(inference_times_vector)

        m.results = inference_times_vector.tolist()
    
    return m


def BenchmarkModelLayers(
    parent_model: str, hardware_list:list, model_summary:dict, count:int
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

    from .usb import init_usbmon
    from .analysis import AnalyzeLayerResults
    from utils.model_lab.logger import log
    

    models = {}
    models["cpu"] = []
    models["gpu"] = []
    models["tpu"] = []
    models["count"] = count

    # regular quantized tflite files for cpu
    if not isCPUavailable():
        log.warning(f"CPU is NOT available on this machine!")
    elif "cpu" not in hardware_list:
        log.info("No CPU cores in hardware summary, skipping benchmarking")
    else:
        log.info(f"CPU is available on this machine!")
        for layer in model_summary["models"][0]["layers"]:
            m = BenchmarkLayer(Model(layer, "cpu", parent_model), count, "CPU")
            models["cpu"].append(m)

    # regular quantized tflite files for gpu
    available, gpu = isGPUavailable()
    if not available:
        log.warning(f"GPU is NOT available on this machine! Type: {gpu}")
    elif "gpu" not in hardware_list:
        log.info("No GPUs in hardware summary, skipping benchmarking")
    else:
        log.info(f"GPU is available on this machine! Type: {gpu}")
        for layer in model_summary["layers"]:
            m = BenchmarkLayer(Model(layer, "gpu", parent_model), count, "GPU")
            models["gpu"].append(m)

    # edge compiled quantized tflite files tpu
    if not isTPUavailable():
        log.warning(f"TPU is NOT available on this machine!")
    elif "tpu" not in hardware_list:
        log.info("No TPUs in hardware summary, skipping benchmarking")
    else:
        log.info(f"TPU is available on this machine!")
        if init_usbmon():
            log.info("Needed to introduce usbmon module")
        else:
            log.info("usbmon module already present")

        for layer in model_summary["layers"]:
            m = BenchmarkLayer(Model(layer, "tpu", parent_model), count, "TPU")
            #AnalyzeLayerResults(m, "tpu")

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
        "cpu": BenchmarkLayer,
        "gpu": BenchmarkLayer,
        "tpu": BenchmarkLayer,
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
