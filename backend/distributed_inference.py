import numpy as np
from cffi import FFI


def distributed_inference(
    tflite_model_path:str,
    input_data:np.array,
    output_data:np.array,
    inference_times:np.array,
    input_data_size:int,
    output_data_size:int,
    hardware_target:str,
    benchmarking_count: int
):
    try:
        from backend import cpp_interface

    except (ModuleNotFoundError, ImportError) as e:
        print(f"Importing of Backend failed:\n{e}\n")
        raise

    _ffi = FFI()

    tflite_model_path_ptr = _ffi.new("char[]", tflite_model_path.encode("utf-8"))
    hardware_target_ptr = _ffi.new("char[]", hardware_target.encode("utf-8"))

    input_data_ptr = _ffi.cast("int8_t*", input_data.ctypes.data)
    output_data_ptr = _ffi.cast("int8_t*", output_data.ctypes.data)
    inference_times_ptr = _ffi.cast("uint32_t*", inference_times.ctypes.data)

    result = cpp_interface.lib.distributed_inference_interface(tflite_model_path_ptr, input_data_ptr,
                                                                                      output_data_ptr,
                                                                                      inference_times_ptr,
                                                                                      input_data_size,
                                                                                      output_data_size,
                                                                                      hardware_target_ptr,
                                                                                      benchmarking_count)

    print("[DISTRIBUTED INFERENCE] Result:{}".format(result))

    if result == -1:
        raise Exception("C++ distributed inference failed")

    return result