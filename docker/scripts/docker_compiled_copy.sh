#!/bin/sh

echo -n "Docker file/folder: "
read -r D_DIR

docker cp debian-docker:/home/debian/single_layer_models/${D_DIR}/edge/edge_${D_DIR}_edgetpu.tflite /home/duclos/Documents/work/FP_Files/edge/

