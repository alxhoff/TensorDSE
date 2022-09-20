cmake_minimum_required(VERSION 3.11)
project(TFLiteCheck)

set(CMAKE_CXX_STANDARD 14)

# include has 2 subdirectories: tensorflow and flatbuffers
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-lib/include/)

# lib has 1 file: libtensorflowlite.so
#ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
#set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-lib/libs/aarch64/libtensorflowlite.so)
#
#ADD_LIBRARY(tensorflowlite-gpu SHARED IMPORTED)
#set_property(TARGET tensorflowlite-gpu PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-lib/libs/aarch64/libtensorflowlite_gpu_delegate.so)
#
#add_executable(TFLiteCheck main.cpp)
#target_link_libraries(TFLiteCheck tensorflowlite tensorflowlite-gpu)


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-lib/libs/aarch64)

# Define your executable target. You can pass in the SOURCES variable defined above.
add_executable(TFLiteCheck main.cpp)

# Link the other libraries to your executable, using their full path.
# Note, the '-L' flag is not necessary.
target_link_libraries(TFLiteCheck PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-lib/libs/aarch64/libtensorflowlite.so
    ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-lib/libs/aarch64/libtensorflowlite_gpu_delegate.so
)