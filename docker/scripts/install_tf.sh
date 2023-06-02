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
export TENSORFLOW_SRC=`pwd`/tensorflow_env/tensorflow_src

main() {

    echo "Working Dir: $PWD"
    # Flatbuffers
    [ -d vcpkg ] || git clone https://github.com/Microsoft/vcpkg.git
    ./vcpkg/bootstrap-vcpkg.sh
    ./vcpkg/vcpkg integrate install
    ./vcpkg/vcpkg install flatbuffers
    [ -f /usr/local/bin/flatc ] || ln -s /home/vcpkg/installed/x64-linux/tools/flatbuffers/flatc /usr/local/bin/flatc
    chmod +x /home/vcpkg/installed/x64-linux/tools/flatbuffers/flatc

    [ -d tensorflow_env ] || mkdir tensorflow_env
    cd tensorflow_env
    [ -d tensorflow_src ] || git clone -b ${TENSORFLOW_VER} --depth 1 https://github.com/tensorflow/tensorflow.git tensorflow_src
    [ -d edgetpu ] || git clone https://github.com/google-coral/edgetpu.git

    # download build dependencies
    [ -d tflite_build ] || mkdir tflite_build
    cd tflite_build
    cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_GPU=ON -DCMAKE_FIND_DEBUG_MODE=1 2>&1 | tee -a log_cmake.txt

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
    #ls -l tensorflow/lite/tools/make/gen/linux_x86_64/lib/
    ls -l bazel-bin/tensorflow/lite/
    ls -l bazel-bin/tensorflow/lite/delegates/gpu/

    [ -d /home/tensorDSE/tensorflow_env/bazel-output ] || mkdir -p /home/tensorDSE/tensorflow_env/bazel-output
    cp ${TENSORFLOW_SRC}/bazel-bin/tensorflow/lite/libtensorflowlite.so /home/tensorDSE/tensorflow_env/bazel-output/
    cp ${TENSORFLOW_SRC}/bazel-bin/tensorflow/lite/libtensorflowlite.so /usr/lib/
    cp ${TENSORFLOW_SRC}/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so /home/tensorDSE/tensorflow_env/bazel-output/
    cp ${TENSORFLOW_SRC}/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so /usr/lib/
}

pushd /home/tensorDSE
main "$@"
popd
