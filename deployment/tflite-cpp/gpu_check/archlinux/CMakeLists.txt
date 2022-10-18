cmake_minimum_required(VERSION 3.11)
project(TFLiteGPUCheck)

set(CMAKE_CXX_STANDARD 14)

set(TARGET_ENV "Desktop" CACHE STRING "Target environment")

message("Target Environment is set to: ${TARGET_ENV}")

if( TARGET_ENV STREQUAL "Desktop" )
    set(TARGET_ARCH x86_64)
endif()

if( TARGET_ENV STREQUAL "DevBoard" )
    set(TARGET_ARCH aarch64)
endif()

message("Target Architecture is set to: ${TARGET_ARCH}")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/tflite-lib/v2.9.1/include/)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/tflite-lib/v2.9.1/libs/${TARGET_ARCH}/archlinux/)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../resources/tensorflow-2.9.1/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../resources/tflite-build/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../resources/tflite-build/abseil-cpp)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../resources/tflite-build/gemmlowp)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../resources/tflite-build/eigen)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../resources/tflite-build/ruy)
#[[INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../resources/tflite-build/opencl_headers)]]
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../resources/tflite-build/neon2sse)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../resources/tflite-build/FP16-source/include/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../resources/tensorflow-2.9.1/third_party)

add_executable(TFLiteGPUCheck main.cpp)

target_link_libraries(TFLiteGPUCheck PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/tflite-lib/v2.9.1/libs/${TARGET_ARCH}/archlinux/libtensorflowlite_gpu_delegate.so
    /usr/lib/libEGL.so
    /usr/lib/libGL.so
    /usr/lib/libGLESv2.so
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/tflite-lib/v2.9.1/libs/${TARGET_ARCH}/archlinux/libtensorflowlite.so
)