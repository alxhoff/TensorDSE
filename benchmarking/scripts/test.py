# import sys
# sys.path.insert(0,'utils')

from utils.model import Model

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
    # import tflite_runtime.interpreter as tflite
    import tensorflow as tf

    shared_library = library
    experimental_delegates = [
        tf.lite.experimental.load_delegate(shared_library)
    ]

    return tf.lite.Interpreter(
        model_path=model_file,
        model_content=None,
        experimental_delegates=experimental_delegates,
    )

def TPUDeploy(m:Model, count:int) -> Model:
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

    for i in range(count):
        start = time.perf_counter()                     # START
        interpreter.invoke()                            # RUNS
        inference_time = time.perf_counter() - start    # END

        _ = interpreter.get_tensor(output_details[0]["index"])  # output data
        results.append([i, inference_time])

        sys.stdout.write(f"\r {i+1}/{count} for TPU ran -> {m.model_name}")
        sys.stdout.flush()
    sys.stdout.write("\n")

    m.results = results
    m.timers = timers
    return m

file="models/compiled/quant_CONV_2D_edgetpu.tflite"
count=10

TPUDeploy(Model(file, "tpu"), count)
