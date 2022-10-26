cmake_minimum_required(VERSION 3.11)
project(TFLiteCheck)

set(CMAKE_CXX_STANDARD 14)

#[[]]

set(TARGET_ENV "Desktop" CACHE STRING "Target environment")
message("Target Environment: ${TARGET_ENV}")
if( TARGET_ENV STREQUAL "Desktop" )
    set(TARGET_ARCH x86_64)
endif()
if( TARGET_ENV STREQUAL "DevBoard" )
    set(TARGET_ARCH aarch64)
endif()
message("Target Architecture: ${TARGET_ARCH}")

#[[]]

set(TARGET_LINUX "Arch" CACHE STRING "Target Linux system")
if( TARGET_LINUX STREQUAL "Arch" )
    set(LIB_PATH /usr/lib)
endif()
if( TARGET_ENV STREQUAL "Debian" )
    set(LIB_PATH /usr/lib/${TARGET_ARCH}-linux-gnu)
endif()
message("Target Linux System: ${TARGET_LINUX}-based Linux")

set(EDGE_TPU_SUPPORT "OFF" CACHE STRING "Use Edge TPU for inference")
message("Edge TPU support: ${EDGE_TPU_SUPPORT}")

set(GPU_SUPPORT "OFF" CACHE STRING "Use GPU for inference")
message("GPU support: ${GPU_SUPPORT}")

set(TENSORFLOW_VERSION "2.9.1" CACHE STRING "Tensorflow Version")
message("Tensorflow Version: ${TENSORFLOW_VERSION}")

set(TENSORFLOW_ENV $ENV{HOME}/Downloads/tensorflow_env)

set(TENSORFLOW_SRC_DIR ${TENSORFLOW_ENV}/tensorflow-${TENSORFLOW_VERSION})

set(TFLITE_BUILD_DIR ${TENSORFLOW_ENV}//tflite-build)

INCLUDE_DIRECTORIES(${TENSORFLOW_SRC_DIR}/)
INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/)
INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/flatbuffers/include/)
INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/abseil-cpp)
INCLUDE_DIRECTORIES(${TENSORFLOW_SRC_DIR}/third_party)

if( GPU_SUPPORT STREQUAL "ON" )
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/gemmlowp)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/eigen)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/ruy)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/opencl_headers)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/neon2sse)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/FP16-source/include/)
endif()

if( EDGE_TPU_SUPPORT STREQUAL "ON" )
    INCLUDE_DIRECTORIES(${TENSORFLOW_ENV}/edgetpu_runtime/libedgetpu)
endif()

if( GPU_SUPPORT STREQUAL "ON" )
    set(PATH_TO_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_check/main.cpp)
elseif( EDGE_TPU_SUPPORT STREQUAL "ON" )
    set(PATH_TO_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/src/edgetpu_check/main.cpp)
else()
    set(PATH_TO_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_check/main.cpp)
endif()
add_executable(TFLiteCheck ${PATH_TO_MAIN})

if( GPU_SUPPORT STREQUAL "ON" )
    target_link_libraries(TFLiteCheck PRIVATE
        ${TTENSORFLOW_ENV}/bazel-output/libtensorflowlite_gpu_delegate.so
        ${LIB_PATH}/libEGL.so
        ${LIB_PATH}/libGL.so
        ${LIB_PATH}/libGLESv2.so
        ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite.so
    )
elseif( EDGE_TPU_SUPPORT STREQUAL "ON" )
    target_link_libraries(TFLiteCheck PRIVATE
        ${TENSORFLOW_ENV}/bazel-output/libtensorflow-lite.a
        ${LIB_PATH}/libedgetpu.so.1.0
    )
else()
    target_link_libraries(TFLiteCheck PRIVATE
        ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite.so
    )
endif()
