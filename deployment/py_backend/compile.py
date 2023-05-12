from cffi import FFI

def compile_interface(verbose:bool = True) -> object:
  ffi = FFI()
  ffi.cdef("""void* distributed_inference_interface(uint8_t* input_data, int benchmarking_count);""")

  ffi.set_source("cpp_interface",
                 """ #include "interface.h" """,
                 include_dirs=['include', '/home/tensorflow_env/tensorflow_src'],
                 libraries=['py_backend', 'tensorflowlite'],
                 library_dirs=['lib', '/home/tensorflow_env/bazel-output/'],
                 source_extension='.cpp')

  return ffi.compile(verbose=verbose)

if __name__ == "__main__":
  compile_interface()
