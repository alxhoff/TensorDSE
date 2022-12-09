#!/bin/sh
set -e

export TENSORFLOW_ENV_DIR=${HOME}/Downloads/tensorflow_env
export TENSORFLOW_VER=2.9
export TENSORFLOW_DIR=${TENSORFLOW_ENV_DIR}/tensorflow-${TENSORFLOW_VER}
# ------------------------------------------------------
#   Build the Tensorflow Lite Dynamic Library 
# ------------------------------------------------------
cd ${TENSORFLOW_DIR}
bazel clean
bazel build -s -c opt //tensorflow/lite:libtensorflowlite.so
echo "----------------------------------------------------"
echo " Build is successful"
echo "----------------------------------------------------"
cp bazel-bin/tensorflow/lite/libtensorflowlite.so ${TENSORFLOW_ENV_DIR}/bazel-output/

# --------------------------------------------------------------------------------------
#   In case of Building Tensorflow Lite Dynamic Library for aarch64
#   Run the following command instead:
#   bazel build -s -c opt --config=elinux_aarch64 //tensorflow/lite:libtensorflowlite.so
# --------------------------------------------------------------------------------------