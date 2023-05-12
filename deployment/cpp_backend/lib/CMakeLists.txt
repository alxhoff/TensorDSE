cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(${LIBRARY_NAME})

set (CMAKE_CXX_STANDARD 14)

file(GLOB SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

set(PROJECT_SOURCES
    ${SOURCES} 
)

set(TENSORFLOW_ENV /home/tensorflow_env)
set(TENSORFLOW_SRC_DIR ${TENSORFLOW_ENV}/tensorflow_src/)
set(TFLITE_BUILD_DIR ${TENSORFLOW_ENV}//tflite_build/)
set(TENSORFLOWLITE_LIBRARY_PATH ${TENSORFLOW_ENV}/bazel-output/libtensorflowlite.so)


set(PROJECT_INCLUDES
    ${PARENT_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${TENSORFLOW_SRC_DIR}/
    ${TFLITE_BUILD_DIR}/
    ${TENSORFLOW_SRC_DIR}/third_party/
    ${TFLITE_BUILD_DIR}/flatbuffers/include/
)

include_directories(${PROJECT_INCLUDES})

link_directories(${TENSORFLOWLITE_LIBRARY_PATH})

add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${TENSORFLOWLITE_LIBRARY_PATH}) 

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include 
        DESTINATION ${PY_BACKEND_DIR}
        MESSAGE_NEVER FILES_MATCHING PATTERN "*"
        )

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${PYBACKEND_DIR}/lib
        CONFIGURATIONS Release
        )