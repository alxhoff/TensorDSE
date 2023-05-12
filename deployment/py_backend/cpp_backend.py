import sys
import numpy as np
from cffi import FFI

try:
  import cpp_interface

except (ModuleNotFoundError, ImportError) as e:
  print(f"Importing of Backend failed:\n{e}\n")
  raise

def distributed_inference(input_data:np.array, benchmarking_count:int):
    
    _ffi = FFI()

    input_data_ptr = _ffi.cast("uint8_t*", input_data.ctypes.data)

    inference_times = cpp_interface.lib.distributed_inference_interface(input_data_ptr, benchmarking_count)

    return inference_times