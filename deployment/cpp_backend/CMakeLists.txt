cmake_minimum_required(VERSION 3.7...3.24)
project(CPP_BACKEND)

set (LIBRARY_NAME py_backend)
set (PY_BACKEND_DIR ${PROJECT_SOURCE_DIR}/../py_backend/)
add_subdirectory(lib) # builds lib project

set (CMAKE_CXX_STANDARD 14)

if( ${CMAKE_BINARY_DIR} STREQUAL ${PROJECT_SOURCE_DIR} )
    Message( FATAL_ERROR "Source and build directories are the same!")
endif()

set(SOURCES "main.cpp")

set(PROJECT_SOURCES
    ${SOURCES} 
)

set(TENSORFLOW_ENV /home/starkaf/Downloads/tensorflow_env)
set(TENSORFLOW_SRC_DIR ${TENSORFLOW_ENV}/tensorflow_src/)
set(TFLITE_BUILD_DIR ${TENSORFLOW_ENV}//tflite_build)
set(TENSORFLOWLITE_LIBRARY_PATH ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite.so)
#set(TENSORFLOWLITE_GPU_DELEGATE_LIBRARY_PATH ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite_gpu_delegate.so)
#
#set(LIB_PATH /usr/lib/x86_64-linux-gnu)
#
#set(EGL_LIBRARY_PATH ${LIB_PATH}/libEGL.so)
#set(GL_LIBRARY_PATH ${LIB_PATH}/libGL.so)
#set(GLESv2_LIBRARY_PATH ${LIB_PATH}/libGLESv2.so)
#set(EDGETPU_LIBRARY_PATH ${LIB_PATH}/libedgetpu.so.1)

#INCLUDE_DIRECTORIES(${TENSORFLOW_SRC_DIR}/)

set(PROJECT_INCLUDES
    ${PROJECT_SOURCE_DIR}/lib/include
    ${TENSORFLOW_SRC_DIR}/
    ${TFLITE_BUILD_DIR}/
    ${TENSORFLOW_SRC_DIR}/third_party/
    ${TFLITE_BUILD_DIR}/flatbuffers/include/
    ${TFLITE_BUILD_DIR}/eigen
    ${TFLITE_BUILD_DIR}/gemmlowp
    ${TFLITE_BUILD_DIR}/ruy
    ${TFLITE_BUILD_DIR}/opencl_headers
    ${TFLITE_BUILD_DIR}/neon2sse
    ${TFLITE_BUILD_DIR}/FP16-source/include/
    ${TFLITE_BUILD_DIR}/vulkan_headers/include/
    ${TFLITE_BUILD_DIR}/abseil-cpp/
    ${TENSORFLOW_ENV}/edgetpu/
    ${TENSORFLOW_ENV}/edgetpu/libedgetpu/
    ${TENSORFLOW_ENV}/edgetpu/libedgetpu/direct/k8
)

MESSAGE(STATUS message(STATUS "INCS: ${PROJECT_INCLUDES}"))

include_directories(${PROJECT_INCLUDES})

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

link_libraries(stdc++fs)

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARY_NAME} ${TENSORFLOWLITE_LIBRARY_PATH}
                                                              ${TENSORFLOWLITE_GPU_DELEGATE_LIBRARY_PATH}
                                                              ${EGL_LIBRARY_PATH}
                                                              ${GL_LIBRARY_PATH}
                                                              ${GLESv2_LIBRARY_PATH}
                                                              ${EDGETPU_LIBRARY_PATH}
)