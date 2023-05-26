import numpy as np
from cffi import FFI


def distributed_inference(
    tflite_model_path:str,
    input_data:np.array,
    output_data:np.array, 
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

    encoded_string = tflite_model_path.encode("utf-8")
    c_str = _ffi.new("char[]", len(encoded_string) + 1)
    _ffi.memmove(c_str, encoded_string, len(encoded_string))
    tflite_model_path_ptr = _ffi.cast("char*", c_str)

    encoded_string = hardware_target.encode("utf-8")
    c_str = _ffi.new("char[]", len(encoded_string) + 1)
    _ffi.memmove(c_str, encoded_string, len(encoded_string))
    hardware_target_ptr = _ffi.cast("char*", c_str)

    input_data_ptr = _ffi.cast("uint8_t*", input_data.ctypes.data)
    output_data_ptr = _ffi.cast("uint8_t*", output_data.ctypes.data)
    

    result = cpp_interface.lib.distributed_inference_interface(tflite_model_path_ptr, input_data_ptr,
                                                                                      output_data_ptr, 
                                                                                      input_data_size, 
                                                                                      output_data_size,
                                                                                      hardware_target_ptr,
                                                                                      benchmarking_count)
    return result
