from cffi import FFI

def compile_interface(verbose:bool = True) -> object:
  ffi = FFI()
  ffi.cdef("""int distributed_inference_interface(uint8_t* input_data, uint8_t* output_data, const unsigned int input_data_size, const unsigned int output_data_size);""")

  import os
  print(os.getcwd())
  ffi.set_source("cpp_interface",
                 """#include "interface.h" """,
                 include_dirs=['../../tensorflow_env/tensorflow_src', 'include'],
                 libraries=['backend', 'tensorflowlite'],
                 library_dirs=['../../tensorflow_env/bazel-output/'],
                 source_extension='.cpp')

  return ffi.compile(verbose=verbose)

if __name__ == "__main__":
  compile_interface()
