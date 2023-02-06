#!/bin/bash

# see the original bash script for more detailed information
# https://github.com/terryky/tflite_gles_app/blob/master/tools/scripts/tf2.9/build_libtflite_r2.9.sh

set -e

if ! command -v bazel &>/dev/null; then
    echo "bazel is needed for this script!"
    exit 1
fi

if ! command -v cmake &>/dev/null; then
    echo "cmake is needed for this script!"
    exit 1
fi

export TENSORFLOW_VER=r2.9
export TENSORFLOW_DIR=`pwd`/tensorflow_${TENSORFLOW_VER}

main() {
    git clone -b ${TENSORFLOW_VER} --depth 1 https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR}
    # download build dependencies
    cd ${TENSORFLOW_DIR}
    mkdir external
    cd external
    cmake ../tensorflow/lite -DCMAKE_FIND_DEBUG_MODE=1 2>&1 | tee -a log_cmake.txt

    # clean up bazel cache, just in case.
    cd ${TENSORFLOW_DIR}
    bazel clean

    echo "----------------------------------------------------"
    echo " (configure) press ENTER-KEY several times.         "
    echo "----------------------------------------------------"
    ./configure


    # ---------------
    #  Bazel build
    # ---------------
    # build with Bazel (libtensorflowlite.so)
    bazel build -s -c opt //tensorflow/lite:libtensorflowlite.so 2>&1 | tee -a log_build_libtflite_bazel.txt

    # build GPU Delegate library (libdelegate.so)
    bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" --copt="-DEGL_NO_X11" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so 2>&1 | tee -a log_build_delegate.txt

    echo "----------------------------------------------------"
    echo " build success."
    echo "----------------------------------------------------"

    cd ${TENSORFLOW_DIR}
    #ls -l tensorflow/lite/tools/make/gen/linux_x86_64/lib/
    ls -l bazel-bin/tensorflow/lite/
    ls -l bazel-bin/tensorflow/lite/delegates/gpu/

    mkdir -p /home/lib/tf2.9
    cp bazel-bin/tensorflow/lite/libtensorflowlite.so /home/lib/tf2.9
    cp bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so /home/lib/tf2.9
}

pushd /home/
main "$@"
popd
