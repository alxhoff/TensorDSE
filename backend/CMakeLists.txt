cmake_minimum_required(VERSION 3.7...3.24)
project(deploy)
set (CMAKE_CXX_STANDARD 14)

if( ${CMAKE_BINARY_DIR} STREQUAL ${PROJECT_SOURCE_DIR} )
    Message( FATAL_ERROR "Source and build directories are the same!")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if(NOT DEFINED TF_ENV)
    set(TF_ENV ${PROJECT_SOURCE_DIR}/../tensorflow_env)
endif()
message(STATUS "TF_ENV: ${TF_ENV}")

if(NOT DEFINED TF_SRC)
    set(TF_SRC ${TF_ENV}/tensorflow_src)
endif()
message(STATUS "TF_SRC: ${TF_SRC}")

set(SKIP_TF_LIB TRUE)
set(FB_SRC ${TF_ENV}/flatbuffers)

find_package(Flatbuffers REQUIRED)
find_package(TFLite REQUIRED)

set(LIB_SOURCES)

set(TFL_BUILD_DIR ${TF_ENV}/tflite_build)
set(TFL_LIB_PATH /usr/lib/libtensorflowlite.so)
set(TFL_GPU_LIB_PATH /usr/lib/libtensorflowlite_gpu_delegate.so)

#Docker
set(LIB_PATH /usr/lib/x86_64-linux-gnu)

#Not sure what
# set(LIB_PATH /usr/lib/aarch64-linux-gnu)

set(EGL_LIBRARY_PATH ${LIB_PATH}/libEGL.so)
set(GL_LIBRARY_PATH ${LIB_PATH}/libGL.so)
set(GLESv2_LIBRARY_PATH ${LIB_PATH}/libGLESv2.so)
set(EDGETPU_LIBRARY_PATH ${LIB_PATH}/libedgetpu.so.1)

set(PROJECT_INCLUDES
    ${PROJECT_SOURCE_DIR}/include
    ${TF_SRC}/
    ${TF_SRC}/third_party/
    ${TFL_BUILD_DIR}/
    ${TFL_BUILD_DIR}/flatbuffers/include/
    ${TFL_BUILD_DIR}/abseil-cpp/
    ${TFL_BUILD_DIR}/FP16-source/include/
    ${TFL_BUILD_DIR}/vulkan_headers/include/
    ${TFL_BUILD_DIR}/eigen
    ${TFL_BUILD_DIR}/gemmlowp
    ${TFL_BUILD_DIR}/ruy
    ${TFL_BUILD_DIR}/opencl_headers
    ${TFL_BUILD_DIR}/neon2sse
    ${TF_ENV}/edgetpu/
    ${TF_ENV}/edgetpu/libedgetpu/
    ${TF_ENV}/edgetpu/libedgetpu/direct/k8
)

include_directories(${PROJECT_INCLUDES})

# Backend Library
set(BACKEND_LIB "backend")
file(GLOB BACKEND_LIB_SRCS "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(${BACKEND_LIB} SHARED ${BACKEND_LIB_SRCS})
target_link_libraries(${BACKEND_LIB} PRIVATE ${TFL_LIB_PATH}
                                             ${TFL_GPU_LIB_PATH}
                                             ${EGL_LIBRARY_PATH}
                                             ${GL_LIBRARY_PATH}
                                             ${GLESv2_LIBRARY_PATH}
                                             ${EDGETPU_LIBRARY_PATH}
)

# Install library to lib dir
install(TARGETS ${BACKEND_LIB} PERMISSIONS WORLD_EXECUTE WORLD_READ LIBRARY DESTINATION /usr/lib)
