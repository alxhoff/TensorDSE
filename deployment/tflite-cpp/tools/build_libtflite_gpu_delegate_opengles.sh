#!/bin/sh
set -e

export TENSORFLOW_ENV_DIR=${HOME}/Downloads/tensorflow_env
export TENSORFLOW_VER=2.9
export TENSORFLOW_DIR=${TENSORFLOW_ENV_DIR}/tensorflow-${TENSORFLOW_VER}
# ----------------------------------------------------------------------
#         Build the Tensorflow Lite GPU Delegate Dynamic Library
#                        (Open GL ES support) 
# ----------------------------------------------------------------------
cd ${TENSORFLOW_DIR}
bazel clean
bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" --copt="-DEGL_NO_X11" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
cp bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so ${TENSORFLOW_ENV_DIR}/bazel-output/

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   In case of Building Tensorflow Lite GPU Delegate Dynamic Library (Open GL ES support) for aarch64
#   Run the following command instead:
#   bazel build -s -c opt --config=elinux_aarch64 --copt="-DMESA_EGL_NO_X11_HEADERS" --copt="-DEGL_NO_X11" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------