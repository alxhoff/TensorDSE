cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(${LIBRARY_NAME})

set (CMAKE_CXX_STANDARD 14)

file(GLOB SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

set(PROJECT_SOURCES
    ${SOURCES} 
)

set(TENSORFLOW_ENV /home/starkaf/Downloads/tensorflow_env)
set(TENSORFLOW_SRC_DIR ${TENSORFLOW_ENV}/tensorflow_src/)
set(TFLITE_BUILD_DIR ${TENSORFLOW_ENV}//tflite_build/)
set(TENSORFLOWLITE_LIBRARY_PATH ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite.so)
set(TENSORFLOWLITE_GPU_DELEGATE_LIBRARY_PATH ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite_gpu_delegate.so)
set(LIB_PATH /usr/lib/x86_64-linux-gnu)
set(EGL_LIBRARY_PATH ${LIB_PATH}/libEGL.so)
set(GL_LIBRARY_PATH ${LIB_PATH}/libGL.so)
set(GLESv2_LIBRARY_PATH ${LIB_PATH}/libGLESv2.so)
set(EDGETPU_LIBRARY_PATH ${LIB_PATH}/libedgetpu.so.1)


set(PROJECT_INCLUDES
#    ${PARENT_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/include
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

include_directories(${PROJECT_INCLUDES})

link_directories(${TENSORFLOWLITE_LIBRARY_PATH})

add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCES})

link_libraries(stdc++fs)

target_link_libraries(${PROJECT_NAME} PRIVATE ${TENSORFLOWLITE_GPU_DELEGATE_LIBRARY_PATH}
                                              ${EGL_LIBRARY_PATH}
                                              ${GL_LIBRARY_PATH}
                                              ${GLESv2_LIBRARY_PATH}
                                              ${EDGETPU_LIBRARY_PATH}
                                              ${TENSORFLOWLITE_LIBRARY_PATH})

                                               
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include 
        DESTINATION ${PY_BACKEND_DIR}
        MESSAGE_NEVER FILES_MATCHING PATTERN "*"
        )

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${PYBACKEND_DIR}/lib
        CONFIGURATIONS Release
        )