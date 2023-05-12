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

set(TENSORFLOW_ENV /home/tensorflow_env)
set(TENSORFLOW_SRC_DIR ${TENSORFLOW_ENV}/tensorflow_src/)
set(TFLITE_BUILD_DIR ${TENSORFLOW_ENV}//tflite_build)
set(TENSORFLOWLITE_LIBRARY_PATH ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite.so)

INCLUDE_DIRECTORIES(${TENSORFLOW_SRC_DIR}/)

set(PROJECT_INCLUDES
    ${PROJECT_SOURCE_DIR}/lib/include
    ${TENSORFLOW_SRC_DIR}/
    ${TFLITE_BUILD_DIR}/
    ${TENSORFLOW_SRC_DIR}/third_party/
    ${TFLITE_BUILD_DIR}/flatbuffers/include/
)

include_directories(${PROJECT_INCLUDES})

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARY_NAME} ${TENSORFLOWLITE_LIBRARY_PATH})