import utils
import argparse

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


def TPUDeploy(m: Model, count: int, usbmon:int, timeout: int = 10, core_index: int = 0) -> Model:
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

    input_size = GetArraySizeFromShape(m.input_shape)
    output_size = GetArraySizeFromShape(m.output_shape)

    time.sleep(DEPLOY_WAIT_TIME)

    for i in range(count):
        signalsQ = Queue()
        dataQ = Queue()

        p = Process(
                target=capture_stream,
                args=(
                    signalsQ,
                    dataQ,
                    timeout,
                    Log(f"resources/logs/layer_{m.index}_{m.model_name}_USB.log"),
                    usbmon,
                    ),
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
        output_data_vector = np.zeros(output_size).astype(
                m.get_np_dtype(m.output_datatype)
                )
        inference_times_vector = np.zeros(count).astype(np.uint32)

        mean_inference_time = distributed_inference(
                m.model_path,
                input_data_vector,
                output_data_vector,
                inference_times_vector,
                len(input_data_vector),
                len(output_data_vector),
                "tpu",
                1,
                core_index
                )

        results.append(mean_inference_time)

        try:
            data = dataQ.get()
        except Exception as e:
            data = None

        if not data == {}:
            timers.append(data)

        if p.is_alive():
            p.join()

        sys.stdout.write(f"\r {i+1}/{count} for TPU ran -> {m.model_name}")
        sys.stdout.flush()

    sys.stdout.write("\n")

    m.results = results
    m.timers = timers
    return m


def StandardDeploy(m: Model, count: int, hardware_target: str) -> Model:
    import numpy as np
    from backend.distributed_inference import distributed_inference

    input_size = GetArraySizeFromShape(m.input_shape)
    output_size = GetArraySizeFromShape(m.output_shape)

    input_data_vector = np.zeros(input_size).astype(
            m.get_np_dtype(m.input_datatype)
            )
    output_data_vector = np.zeros(output_size).astype(
            m.get_np_dtype(m.output_datatype)
            )
    inference_times_vector = np.zeros(count).astype(np.uint32)

    try:
        mean_inference_time = distributed_inference(
                m.model_path,
                input_data_vector,
                output_data_vector,
                inference_times_vector,
                len(input_data_vector),
                len(output_data_vector),
                hardware_target,
                count,
                )
    except Exception as e:
        print(e)

    m.results = inference_times_vector.tolist()

    return m


def ProfileLayer(m: Model, count: int, hardware_target: str, platform: str) -> Model:
    import os
    from utils.splitter.split import SUB_DIR, COMPILED_DIR

    if (hardware_target == "TPU"):
        m.model_path = os.path.join(
                COMPILED_DIR,
                "submodel_{0}_{1}_bm_edgetpu.tflite".format(
                    m.details["index"], m.details["type"]
                    ),
                )
    else:
        m.model_path = os.path.join(
                SUB_DIR,
                "tflite",
                "submodel_{0}_{1}_bm".format(m.details["index"], m.details["type"]),
                "submodel_{0}_{1}_bm.tflite".format(
                    m.details["index"], m.details["type"]
                    ),
                )

    if ((platform == "desktop") or (platform == "rpi")):
        if os.path.isfile(m.model_path):
            if (hardware_target == "TPU"):
                m = TPUDeploy(m=m, count=count)
            else:
                m = StandardDeploy(m=m, count=count, hardware_target=hardware_target)

        else:
            m.results = [-1] * count
            m.timers = {
                    "error": {
                        "name": "file_not_found",
                        "reason": "Cannot find model {}".format(m.model_path),
                        },
                    }
    else:
        m = StandardDeploy(m=m, count=count, hardware_target=hardware_target)
        
    return m


def ProfileModelLayers(
        parent_model: str,
        hardware_list: list,
        model_summary: dict,
        count: int,
        platform: str,
        usbmon_bus: int
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
    # from .analysis import AnalyzeLayerResults
    from utils.splitter.logger import log

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
        log.info(f"[PROFILE MODEL LAYERS] CPU is available on this machine!")
        for layer in model_summary["models"][0]["layers"]:
            m = ProfileLayer(Model(layer, "cpu", parent_model), count, "cpu", platform, usbmon=usbmon_bus)
            models["cpu"].append(m)
        # AnalyzeLayerResults(m, "cpu")

    # regular quantized tflite files for gpu
    if (platform == "desktop"):
        available, gpu = isGPUavailable()
        if not available:
            log.warning(f"GPU is NOT available on this machine! Type: {gpu}")
        elif "gpu" not in hardware_list:
            log.info("No GPUs in hardware summary, skipping benchmarking")
        else:
            log.info(
                    f"[PROFILE MODEL LAYERS] GPU is available on this machine! Type: {gpu}"
                    )
            for layer in model_summary["models"][0]["layers"]:
                m = ProfileLayer(
                        Model(layer, "gpu", parent_model), count, "gpu", platform, usbmon=usbmon_bus
                        )
                models["gpu"].append(m)
                # AnalyzeLayerResults(m, "gpu")
            print("[PROFILE MODEL LAYERS] GPUs profiled")

    elif (platform == "coral") or (platform == "rpi"):
        for layer in model_summary["models"][0]["layers"]:
            m = ProfileLayer(
                    Model(layer, "gpu", parent_model), count, "GPU", platform
                    )
            models["gpu"].append(m)
            # AnalyzeLayerResults(m, "gpu")

    # edge compiled quantized tflite files tpu
    if (platform == "desktop") or (platform == "rpi"):
        if not isTPUavailable():
            log.warning(f"TPU is NOT available on this machine!")
        elif "tpu" not in hardware_list:
            log.info("No TPUs in hardware summary, skipping benchmarking")
        else:
            log.info(f"[PROFILE MODEL LAYERS] TPU is available on this machine!")
            if init_usbmon(usb_bus=usbmon_bus) and (not (platform == "coral")):
                log.info("Needed to introduce usbmon module")
            else:
                log.info("usbmon module already present")
                for layer in model_summary["models"][0]["layers"]:
                    m = ProfileLayer(
                            Model(layer, "tpu", parent_model), count, "tpu", platform, usbmon=usbmon_bus
                            )
                    models["tpu"].append(m)
                    # AnalyzeLayerResults(m, "tpu")
            print("[PROFILE MODEL LAYERS] TPUs profiled")

    elif (platform == "coral"):
        for layer in model_summary["models"][0]["layers"]:
            m = ProfileLayer(
                    Model(layer, "tpu", parent_model), count, "TPU", platform
                    )
            models["tpu"].append(m)
            # AnalyzeLayerResults(m, "tpu")

    return models
