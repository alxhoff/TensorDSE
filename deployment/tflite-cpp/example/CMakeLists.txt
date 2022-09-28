cmake_minimum_required(VERSION 3.17)
project(TFLiteImageClassification)

set(CMAKE_CXX_STANDARD 14)

# OpenCV Integration
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorFlow Lite and Flatbuffers Integration
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)

add_executable(TFLiteCheck main.cpp)
ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/linux_x64/libtensorflowlite.so)
target_link_libraries(TFLiteCheck tensorflowlite ${OpenCV_LIBS})

