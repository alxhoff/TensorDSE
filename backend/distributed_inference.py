"""
Missing  Docstring: TODO
"""

import numpy as np
import time
from cffi import FFI

try:
    from backend import cpp_interface

except (ModuleNotFoundError, ImportError) as e:
    print(f"Importing of Backend failed:\n{e}\n")
    raise

def distributed_inference(
    tflite_model_path: str,
    input_data: np.array,
    output_data: np.array,
    inference_times: np.array,
    data_times: np.array,
    input_data_size: int,
    output_data_size: int,
    hardware_target: str,
    platform: str,
    benchmarking_count: int,
    core_index: int,
    data_type: str
):
    """
    Missing  Docstring: TODO
    """

    check_point_1 = time.time_ns()

    _ffi = FFI()

    tflite_model_path_ptr = _ffi.new("char[]", tflite_model_path.encode("utf-8"))
    hardware_target_ptr = _ffi.new("char[]", hardware_target.encode("utf-8"))
    platform_ptr = _ffi.new("char[]", platform.encode("utf-8"))

    if data_type == 'int8':
        input_data_ptr = _ffi.cast("int8_t*", input_data.ctypes.data)
        output_data_ptr = _ffi.cast("int8_t*", output_data.ctypes.data)
        data_type_flag = 0  # Corresponding to INT8 in the C++ enum
    elif data_type == 'uint8':
        input_data_ptr = _ffi.cast("uint8_t*", input_data.ctypes.data)
        output_data_ptr = _ffi.cast("uint8_t*", output_data.ctypes.data)
        data_type_flag = 1  # Corresponding to UINT8 in the C++ enum
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    inference_times_ptr = _ffi.cast("uint32_t*", inference_times.ctypes.data)
    data_times_ptr = _ffi.cast("uint32_t*", data_times.ctypes.data)

    result = cpp_interface.lib.distributed_inference_interface(
        tflite_model_path_ptr,
        input_data_ptr,
        output_data_ptr,
        inference_times_ptr,
        data_times_ptr,
        input_data_size,
        output_data_size,
        hardware_target_ptr,
        platform_ptr,
        benchmarking_count,
        core_index,
        data_type_flag
        )

    if result == -1:
        raise IOError("C++ distributed inference failed")

    result /= 1000000000.0

    check_point_2 = time.time_ns()

    backend_inference_time = check_point_2 - check_point_1

    return result, backend_inference_time
