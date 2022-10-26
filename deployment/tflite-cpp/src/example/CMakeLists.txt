cmake_minimum_required(VERSION 3.17)
project(TFLiteClassify)

set(CMAKE_CXX_STANDARD 14)

# OpenCV Integration
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorFlow Lite and Flatbuffers Integration
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-lib/include/)

add_executable(TFLiteClassify main.cpp)
ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-lib/libs/linux_x64/libtensorflowlite.so)
target_link_libraries(TFLiteImageClassification tensorflowlite ${OpenCV_LIBS})
