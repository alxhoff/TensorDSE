#!/usr/bin/env bash

echo -n "Docker file/folder: "
read -r d_dir

docker cp debian-docker:/home/debian/single_layer_models/${d_dir}/edge/edge_${D_DIR}_edgetpu.tflite /home/duclos/Documents/work/FP_Files/edge/

