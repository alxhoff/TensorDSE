from cffi import FFI

def compile_interface(verbose:bool = True) -> object:
  ffi = FFI()
  ffi.cdef("""int distributed_inference_interface(char* tflite_model_path_ptr, int8_t* input_data, int8_t* output_data, uint32_t* inference_times, const unsigned int input_data_size, const unsigned int output_data_size, char* hardware_target_ptr, char* platform_ptr, const unsigned int benchmarking_count, const unsigned int core_index);""")

  import os
  print(os.getcwd())
  ffi.set_source("cpp_interface",
                 """#include "interface.h" """,
                 include_dirs=['/home/tensorDSE/tensorflow_env/tensorflow_src', 'include'],
                 libraries=['backend', 'tensorflowlite', 'tensorflowlite_gpu_delegate'],
                 library_dirs=['/usr/lib'],
                 source_extension='.cpp')

  return ffi.compile(verbose=verbose)

if __name__ == "__main__":
  compile_interface()
