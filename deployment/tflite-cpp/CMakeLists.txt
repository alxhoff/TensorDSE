cmake_minimum_required(VERSION 3.7...3.23)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()
project(TFLiteApp VERSION 1.0
                    DESCRIPTION "Run Distributed Inference on CPU, GPU, and TPU"
                    LANGUAGES CXX)

#set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_BUILD_TYPE Debug)

set(TARGET_ENV "Desktop" CACHE STRING "Target environment")
message("Target Environment: ${TARGET_ENV}")
if( TARGET_ENV STREQUAL "Desktop" )
    set(TARGET_ARCH x86_64)
elseif( TARGET_ENV STREQUAL "DevBoard" )
    set(TARGET_ARCH aarch64)
else()
    message( FATAL_ERROR "Target Environment not supported. CMake will exit." )
endif()
message("Target Architecture: ${TARGET_ARCH}")

#[[]]

set(TARGET_LINUX "Arch" CACHE STRING "Target Linux System")
if( TARGET_LINUX STREQUAL "Arch" )
    set(LIB_PATH /usr/lib)
    set(LIBEDGETPU_PATH ${LIB_PATH}/libedgetpu.so)
elseif( TARGET_LINUX STREQUAL "Debian" )
    set(LIB_PATH /usr/lib/${TARGET_ARCH}-linux-gnu)
    set(LIBEDGETPU_PATH ${LIB_PATH}/libedgetpu.so.1)
else()
    message( FATAL_ERROR "Target Linux System not supported. CMake will exit." )
endif()
message("Target Linux System: ${TARGET_LINUX}-based Linux")

set(DISTRIBUTED "OFF" CACHE STRING "Distributed Inference")
set(EDGE_TPU_SUPPORT "OFF" CACHE STRING "Use Edge TPU for inference")
set(GPU_SUPPORT "OFF" CACHE STRING "Use GPU for inference")

if( DISTRIBUTED STREQUAL "ON" )
    set(EDGE_TPU_SUPPORT "ON")
    set(GPU_SUPPORT "ON")
endif()
message("Distributed Inference: ${DISTRIBUTED}")
message("Edge TPU support: ${EDGE_TPU_SUPPORT}")
message("GPU support: ${GPU_SUPPORT}")

set(INFERENCE_TYPE "CLASSIFICATION" CACHE STRING "Inference Type")
if( INFERENCE_TYPE STREQUAL "CHECK" )
    set(SRC_SUB_FOLDER check)
elseif( INFERENCE_TYPE STREQUAL "CLASSIFICATION" )
    set(SRC_SUB_FOLDER classification)
else()
    message( FATAL_ERROR "Inference Type not supported. CMake will exit." )
endif()
message("Inference Type: ${INFERENCE_TYPE}")

set(TENSORFLOW_VERSION "2.9.1" CACHE STRING "Tensorflow Version")
message("Tensorflow Version: ${TENSORFLOW_VERSION}")

set(TENSORFLOW_ENV /home/tensorflow_env)

set(TENSORFLOW_SRC_DIR ${TENSORFLOW_ENV}/tensorflow_src)

set(TFLITE_BUILD_DIR ${TENSORFLOW_ENV}//tflite_build)


if(EXISTS ${TENSORFLOW_SRC_DIR}/)
    INCLUDE_DIRECTORIES(${TENSORFLOW_SRC_DIR}/)
else()
    message( FATAL_ERROR "${TENSORFLOW_SRC_DIR}: No such file or directory found!" )    
endif()


INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/)
INCLUDE_DIRECTORIES(${TENSORFLOW_SRC_DIR}/third_party/)
INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/flatbuffers/include/)



if( GPU_SUPPORT STREQUAL "ON" )
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/eigen)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/gemmlowp)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/ruy)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/opencl_headers)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/neon2sse)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/FP16-source/include/)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/vulkan_headers/include/)
    INCLUDE_DIRECTORIES(${TFLITE_BUILD_DIR}/abseil-cpp/)
endif()

#[[INCLUDE_DIRECTORIES(${TENSORFLOW_ENV}/edgetpu_runtime/libedgetpu)]]
if( EDGE_TPU_SUPPORT STREQUAL "ON" )
    INCLUDE_DIRECTORIES(${TENSORFLOW_ENV}/edgetpu/) 
    INCLUDE_DIRECTORIES(${TENSORFLOW_ENV}/edgetpu/libedgetpu/)
    INCLUDE_DIRECTORIES(${TENSORFLOW_ENV}/edgetpu/libedgetpu/direct/k8)
endif()

link_libraries(stdc++fs)

if( DISTRIBUTED STREQUAL "ON" )
    set(PATH_TO_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/src/${SRC_SUB_FOLDER}/distributed/main.cpp)
else()
    if( GPU_SUPPORT STREQUAL "ON" )
        set(PATH_TO_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/src/${SRC_SUB_FOLDER}/gpu/main.cpp)
    elseif( EDGE_TPU_SUPPORT STREQUAL "ON" )
        set(PATH_TO_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/src/${SRC_SUB_FOLDER}/tpu/main.cpp)
    else()
        set(PATH_TO_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/src/${SRC_SUB_FOLDER}/cpu/main.cpp)
    endif()
endif()
add_executable(TFLiteApp ${PATH_TO_MAIN})
message("Path to main: ${PATH_TO_MAIN}")

if( DISTRIBUTED STREQUAL "ON" )
    target_link_libraries(TFLiteApp PRIVATE
        ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite_gpu_delegate.so
        ${LIB_PATH}/libEGL.so
        ${LIB_PATH}/libGL.so
        ${LIB_PATH}/libGLESv2.so
        ${LIBEDGETPU_PATH}
        ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite.so
    )
else()
    if( GPU_SUPPORT STREQUAL "ON" )
        target_link_libraries(TFLiteApp PRIVATE
            ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite_gpu_delegate.so
            ${LIB_PATH}/libEGL.so
            ${LIB_PATH}/libGL.so
            ${LIB_PATH}/libGLESv2.so
            ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite.so
        )   
    elseif( EDGE_TPU_SUPPORT STREQUAL "ON" )
        target_link_libraries(TFLiteApp PRIVATE
            ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite.so
            ${LIBEDGETPU_PATH}
        )
    else()
        target_link_libraries(TFLiteApp PRIVATE
            ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite.so
        )
    endif()
endif()
