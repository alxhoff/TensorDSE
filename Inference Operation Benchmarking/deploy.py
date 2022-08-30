import argparse

class Model:
    def __init__(self, file:str, delegate:str):
        self.file = file
        self.delegate = delegate
        self._get_model_name(file)
        self.results = []

    def  _get_model_name (self, file:str):
        file = file.split("/")[file.count("/")]
        if (not file.startswith("quant_") or
            not file.endswith(".tflite")):
               raise Exception("")

        self.op_name = (
         file.split("quant_")[1]
        ).split("_edgetpu.tflite"
           if self.delegate == "tpu"
           else ".tflite" )[0]

def make_interpreter(model_file:str, library:str):
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


def TPUDeploy(m:Model, count:int) -> None:
    import time
    import logging
    import numpy as np
    import platform

    results = []

    TPU_LIBRARY = {
        "Linux": "libedgetpu.so.1",
        "Darwin": "libedgetpu.1.dylib",
        "Windows": "edgetpu.dll",
    }[platform.system()]

    interpreter = make_interpreter(m.file, TPU_LIBRARY)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(
            np.random.random_sample(
                input_details[0]["shape"]),         # input shape
                dtype=input_details[0]["dtype"])    # input dtype

    interpreter.set_tensor(input_details[0]["index"], input_data)

    for i in range(count):
        start = time.perf_counter()                     # START
        interpreter.invoke()                            # RUNS
        inference_time = time.perf_counter() - start    # END

        _ = interpreter.get_tensor(output_details[0]["index"])  # output data
        results.append([i, inference_time])

    m.results = results


def GPUDeploy(m:Model, count:int) -> None:
    pass

def CPUDeploy(m:Model, count:int) -> None:
    import time
    import logging
    import tensorflow as tf
    import numpy as np

    results = []

    # Interpreter Object.
    interpreter = tf.lite.Interpreter(model_path=m.file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(
            np.random.random_sample(
                input_details[0]["shape"]),         # input shape
                dtype=input_details[0]["dtype"])    # input dtype

    interpreter.set_tensor(input_details[0]["index"], input_data)

    for i in range(count):
        start = time.perf_counter()                     # START
        interpreter.invoke()                            # RUNS
        inference_time = time.perf_counter() - start    # END

        _ = interpreter.get_tensor(output_details[0]["index"])  # output data
        results.append([i, inference_time])

    m.results = results

def DeployModels(count=1000)  -> None:
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
    from main import MODELS_FOLDER, COMPILED_MODELS_FOLDER


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
        "-m", "--mode", default="Deploy", required=False, help="Single, Group."
    )

    parser.add_argument(
        "-d", "--delegate", default="", required=False, help="cpu or edge."
    )

    parser.add_argument(
        " -t", "--target", help="File path to the .tflite file.")

    parser.add_argument(
        "-c", "--count", type=int, default=1, help="Number of times to run inference."
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    """Entry point to execute this script.

    Flags
    ---------
    -m or --mode
    Mode in which the script should run. Group, Single or Debug.
        Single deploys a single model.
        Group is supposed to be used to deploy a group of models in sequence.

    -d or --delegate
        Specifies which hardware should be used to delegate a model.
            cpu => cpu
            edge => edgetpu

    -t or --target
        Should be used in conjunction with the 'Single' mode, where -t must be followed
        by the path to the model (target) that will be deployed.

    -c or --count
        Should be followed by the number of times one wishes to deploy the group of models
        or the single target (Depends on mode).
    """

    args = GetArgs()

    delegators = {
        "cpu"   : CPUDeploy,
        "gpu"   : GPUDeploy,
        "tpu"   : TPUDeploy,
    }

    if args.mode == "Single":
        delegator = delegators.get(args.delegate, None)
        if not (delegator == None):
            delegator(Model(args.target, args.delegate), args.count)

    if args.mode == "Group":
        from os import listdir
        from os.path import join

    else:
        raise Exception(f"Invalid mode: {args.mode}")
