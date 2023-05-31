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

set(LIB_SOURCES)

set(TFL_BUILD_DIR ${TF_ENV}/tflite_build)
set(TFL_LIB_PATH /usr/lib/libtensorflowlite.so)

set(PROJECT_INCLUDES
    ${PROJECT_SOURCE_DIR}/include
    ${TF_SRC}
    ${TF_SRC}/third_party/
    ${TFL_BUILD_DIR}/
    ${TFL_BUILD_DIR}/flatbuffers/include/
)

include_directories(${PROJECT_INCLUDES})

# Backend Library
set(BACKEND_LIB "backend")
file(GLOB BACKEND_LIB_SRCS "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(${BACKEND_LIB} SHARED ${BACKEND_LIB_SRCS})
target_link_libraries(${BACKEND_LIB} PRIVATE ${TFL_LIB_PATH})

# Install library to lib dir
install(TARGETS ${BACKEND_LIB} PERMISSIONS WORLD_EXECUTE WORLD_READ LIBRARY DESTINATION /usr/lib)