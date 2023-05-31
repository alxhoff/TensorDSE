cmake_minimum_required(VERSION 3.7...3.24)
project(deploy)
set (CMAKE_CXX_STANDARD 14)

if( ${CMAKE_BINARY_DIR} STREQUAL ${PROJECT_SOURCE_DIR} )
    Message( FATAL_ERROR "Source and build directories are the same!")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(TF_ENV ${PROJECT_SOURCE_DIR}/../../tensorflow_env)
set(TF_SRC ${TF_ENV}/tensorflow_src)
set(SKIP_TF_LIB TRUE)
set(FB_SRC ${TF_SRC}/third_party/flatbuffers)

find_package(TFLite REQUIRED)
find_package(Flatbuffers REQUIRED)

set(PROJECT_SOURCES
    "main.cpp"
)

set(TFL_BUILD_DIR ${TF_ENV}/tflite_build)
set(TFL_LIBRARY_DIR ${TF_ENV}/bazel-output)
set(TFL_LIB_PATH ${TFL_LIBRARY_DIR}/libtensorflowlite.so)
message(STATUS "TFLite library dir: ${TFL_LIBRARY_DIR}")
set(TFL_GPU_LIB_PATH ${TFL_LIBRARY_DIR}/libtensorflowlite_gpu_delegate.so)

set(LIB_PATH /usr/lib/x86_64-linux-gnu)
set(EGL_LIBRARY_PATH ${LIB_PATH}/libEGL.so)
set(GL_LIBRARY_PATH ${LIB_PATH}/libGL.so)
set(GLESv2_LIBRARY_PATH ${LIB_PATH}/libGLESv2.so)
set(EDGETPU_LIBRARY_PATH ${LIB_PATH}/libedgetpu.so.1)


set(PROJECT_INCLUDES
    ${PROJECT_SOURCE_DIR}/lib/include
    ${PROJECT_SOURCE_DIR}/include
    ${TF_SRC}
    ${TF_SRC}/third_party/
    ${TFL_BUILD_DIR}/
    ${TFL_BUILD_DIR}/flatbuffers/include/
    ${TFL_BUILD_DIR}/eigen
    ${TFL_BUILD_DIR}/gemmlowp
    ${TFL_BUILD_DIR}/ruy
    ${TFL_BUILD_DIR}/opencl_headers
    ${TFL_BUILD_DIR}/neon2sse
    ${TFL_BUILD_DIR}/FP16-source/include/
    ${TFL_BUILD_DIR}/vulkan_headers/include/
    ${TFL_BUILD_DIR}/abseil-cpp/
    ${TF_ENV}/edgetpu/
    ${TF_ENV}/edgetpu/libedgetpu/
    ${TF_ENV}/edgetpu/libedgetpu/direct/k8
)

include_directories(${PROJECT_INCLUDES})

# Library
set(BACKEND_LIB "backend")
file(GLOB BACKEND_LIB_SRCS "${PROJECT_SOURCE_DIR}/lib/src/*.cpp")
add_library(${BACKEND_LIB} SHARED ${BACKEND_LIB_SRCS})
target_link_libraries(${BACKEND_LIB} PRIVATE ${TFL_GPU_LIB_PATH}
                                             ${EGL_LIBRARY_PATH}
                                             ${GL_LIBRARY_PATH}
                                             ${GLESv2_LIBRARY_PATH}
                                             ${EDGETPU_LIBRARY_PATH}
                                             ${TFL_LIB_PATH})

# Executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_link_libraries(${PROJECT_NAME} ${BACKEND_LIB})

message(STATUS "Installing to: ${TFL_LIBRARY_DIR}")

# Install library to lib dir
install(TARGETS ${BACKEND_LIB} LIBRARY DESTINATION /usr/lib)