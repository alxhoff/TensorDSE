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

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tflite-lib/v2.9.1/include/)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tflite-lib/v2.9.1/libs/${TARGET_ARCH}/)

add_executable(TFLiteGPUCheck main.cpp)

target_link_libraries(TFLiteGPUCheck PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tflite-lib/v2.9.1/libs/${TARGET_ARCH}/libtensorflowlite_gpu_delegate.so
    /usr/lib/${TARGET_ARCH}-linux-gnu/libEGL.so
    /usr/lib/${TARGET_ARCH}-linux-gnu/libGL.so
    /usr/lib/${TARGET_ARCH}-linux-gnu/libGLESv2.so
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tflite-lib/v2.9.1/libs/${TARGET_ARCH}/libtensorflowlite.so
)