#!/bin/sh

BEGIN=$(date +%s)

sudo python3 deploy.py -l False -d edge_tpu -c 1 -m models/tpu_compiled_models/quant_CONV_2D_edgetpu.tflite

END=$(date +%s)

echo "Time Passed: $((END - BEGIN))"


