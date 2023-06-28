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

[ -d /home/tensorDSE ] || mkdir /home/tensorDSE
pushd /home/tensorDSE

export TENSORFLOW_VER=r2.9
export TENSORFLOW_SRC=`pwd`/tensorflow_env/tensorflow_src

main() {

    echo "Working Dir: $(pwd)"

    # clean up bazel cache, just in case.
    cd ${TENSORFLOW_SRC}
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

    cd ${TENSORFLOW_SRC}

    [ -d /home/tensorDSE/tensorflow_env/bazel-output ] || mkdir -p /home/tensorDSE/tensorflow_env/bazel-output
    cp ${TENSORFLOW_SRC}/bazel-bin/tensorflow/lite/libtensorflowlite.so /home/tensorDSE/tensorflow_env/bazel-output/
    cp ${TENSORFLOW_SRC}/bazel-bin/tensorflow/lite/libtensorflowlite.so /usr/lib/
    cp ${TENSORFLOW_SRC}/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so /home/tensorDSE/tensorflow_env/bazel-output/
    cp ${TENSORFLOW_SRC}/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so /usr/lib/
}

main "$@"
popd
