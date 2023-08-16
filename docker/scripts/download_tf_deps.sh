#!/bin/bash

set -e

if ! command -v bazel &>/dev/null; then
    echo "bazel is needed for this script!"
    exit 1
fi

if ! command -v cmake &>/dev/null; then
    echo "cmake is needed for this script!"
    exit 1
fi

[ -d /home/sources ] || mkdir /home/sources
pushd /home/sources

export TENSORFLOW_VER=r2.9
export TENSORFLOW_SRC=`pwd`/tensorflow_env/tensorflow_src

main() {
    [ -d tensorflow_env ] || mkdir tensorflow_env
    cd tensorflow_env
    # download build dependencies
    [ -d tflite_build ] || mkdir tflite_build
    cd tflite_build
    cmake /home/sources/tensorflow_env/tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_GPU=ON -DCMAKE_FIND_DEBUG_MODE=1 2>&1 | tee -a log_cmake.txt
}

main "$@"
popd

exit 0
