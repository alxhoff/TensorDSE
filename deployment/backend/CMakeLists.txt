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
set(LIBRARY_DIR ${TF_ENV}/bazel-output)
set(TFL_LIB_PATH ${LIBRARY_DIR}/libtensorflowlite.so)
message(STATUS "TFLite library dir: ${LIBRARY_DIR}")
set(TFL_GPU_LIB_PATH ${LIBRARY_DIR}/libtensorflowlite_gpu_delegate.so)

set(PROJECT_INCLUDES
    ${PROJECT_SOURCE_DIR}/lib/include
    ${PROJECT_SOURCE_DIR}/include
    ${TF_SRC}
    ${TF_SRC}/third_party/
    ${TFL_BUILD_DIR}/
    ${TFL_BUILD_DIR}/flatbuffers/include/
)

include_directories(${PROJECT_INCLUDES})

# Library
set(BACKEND_LIB "backend")
file(GLOB BACKEND_LIB_SRCS "${PROJECT_SOURCE_DIR}/lib/src/*.cpp")
add_library(${BACKEND_LIB} SHARED ${BACKEND_LIB_SRCS})
target_link_libraries(${BACKEND_LIB} PRIVATE ${TFL_LIB_PATH})

# Executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_link_libraries(${PROJECT_NAME} ${BACKEND_LIB})

message(STATUS "Installing to: ${LIBRARY_DIR}")

# Install library to lib dir
install(TARGETS ${BACKEND_LIB}
    LIBRARY DESTINATION ${LIBRARY_DIR}
)
