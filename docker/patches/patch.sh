#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "patch /usr/include/EGL/eglplatform.h < $SCRIPT_DIR/eglplatform.patch &> /dev/null"
patch /usr/include/EGL/eglplatform.h < $SCRIPT_DIR/eglplatform.patch &> /dev/null

echo "patch /home/tensorDSE/tensorflow_env/tensorflow_src/tensorflow/lite/kernels/internal/optimized/optimized_ops.h < $SCRIPT_DIR/optimized_ops.patch &> /dev/null"
patch /home/tensorDSE/tensorflow_env/tensorflow_src/tensorflow/lite/kernels/internal/optimized/optimized_ops.h < $SCRIPT_DIR/optimized_ops.patch &> /dev/null

exit 0
