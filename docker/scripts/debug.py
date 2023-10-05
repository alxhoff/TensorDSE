import sys
import os
sys.path[0]="/home/tensorDSE" # need to overwrite working directory, so imports can work

def debug_stream(model_path, timeout, usbmon):
    from multiprocessing import Process,Queue
    from utils.logging.log import Log
    from usb import START_DEPLOYMENT, END_DEPLOYMENT, SUCCESSFULL_DEPLOYMENT
    from usb.debug import capture_stream
    from deploy import MakeInterpreter
    import numpy as np
    import time
    import platform

    TPU_LIBRARY = {
        "Linux": "libedgetpu.so.1",
        "Darwin": "libedgetpu.1.dylib",
        "Windows": "edgetpu.dll",
    }[platform.system()]

    interpreter = MakeInterpreter(model_path, TPU_LIBRARY)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(
            np.random.random_sample(
                input_details[0]["shape"]),         # input shape
                dtype=input_details[0][Constants.dtype])    # input dtype

    interpreter.set_tensor(input_details[0]["index"], input_data)

    signalsQ = Queue()
    dataQ = Queue()

    p = Process(target=capture_stream, args=(signalsQ, dataQ, timeout, Log(f"results/DEBUG_USB.log", usbmon)))
    p.start()

    sig = signalsQ.get()
    if sig == END_DEPLOYMENT:
        p.join()
        return

    time.sleep(10)
    start = time.perf_counter()                     # START
    interpreter.invoke()                            # RUNS
    _ = time.perf_counter() - start    # END
    _ = interpreter.get_tensor(output_details[0]["index"])  # output data

    _ = dataQ.get()
    p.join()

debug_stream("models/compiled/quant_CONV_2D_edgetpu.tflite", 15)
