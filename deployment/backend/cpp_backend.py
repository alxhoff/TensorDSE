import sys
import numpy as np
from cffi import FFI

try:
  import cpp_interface

except (ModuleNotFoundError, ImportError) as e:
  print(f"Importing of Backend failed:\n{e}\n")
  raise

def distributed_inference(input_data:np.array, output_data:np.array, input_data_size:int, output_data_size:int):
    
    _ffi = FFI()

    input_data_ptr = _ffi.cast("uint8_t*", input_data.ctypes.data)
    output_data_ptr = _ffi.cast("uint8_t*", output_data.ctypes.data)

    result = cpp_interface.lib.distributed_inference_interface(input_data_ptr, output_data_ptr, input_data_size, output_data_size)
    return result