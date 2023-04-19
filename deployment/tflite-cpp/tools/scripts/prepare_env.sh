#!/bin/sh
set -e

mkdir ${HOME}/Downloads/tensorflow_env
export TENSORFLOW_ENV_DIR=${HOME}/Downloads/tensorflow_env
export TENSORFLOW_VER=2.9
export TENSORFLOW_DIR=${TENSORFLOW_ENV_DIR}/tensorflow-${TENSORFLOW_VER}
# ------------------------------------------------------
#   Clone Tensorflow Source Code (shallow clone) 
# ------------------------------------------------------
git clone -b r${TENSORFLOW_VER} --depth 1 https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR}

cd ${TENSORFLOW_ENV_DIR}
# ------------------------------------------------------
#   Prepare this directory to contain dynamic libraries 
# ------------------------------------------------------
mkdir bazel-output
# ------------------------------------------------------
#   Create external directory and Download dependencies 
# ------------------------------------------------------
mkdir tflite-build
cd tflite-build
cmake ../tensorflow-${TENSORFLOW_VER}/tensorflow/lite -DTFLITE_ENABLE_GPU=ON #TFLITE_ENABLE_GPU is OFF by default

# ------------------------------------------------------
#   Configure Tensorflow
# ------------------------------------------------------
cd ${TENSORFLOW_DIR}
bazel clean
echo "----------------------------------------------------"
echo "     Initiating Tensorflow Basic Configuration      "
echo "           Press ENTER-KEY several times            "
echo "----------------------------------------------------"
./configure



