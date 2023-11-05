#!/bin/bash

MODEL="resources/models/example_models/MNIST/MNIST_extended_full_quanitization.tflite"
filename_with_extension="${MODEL##*/}"
file_basename="${filename_with_extension%.*}"

model_type="MNIST_extended"

platform="rpi"

OUTPUT_FOLDER="resources/ILP_results/${file_basename}"
if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

# GENERATIONS=20
# POPULATION=50

# for COUNT in 1 2 3 4 5; do
#     MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$COUNT.json"
#     OUTPUT_NAME="results_rpi_P$POPULATION\_G$GENERATIONS\_$COUNT.csv"
#     make dse USBMON=0 MODEL=$MODEL PROFILING_COSTS=$PROFILING_RESULTS MODEL_SUMMARY=$MODEL_SUMMARY \
#         OUTPUT_FOLDER=$OUTPUT_FOLDER OUTPUT_NAME=$OUTPUT_NAME POPULATION_SIZE=$POPULATION GENERATIONS=$GENERATIONS \
#         ILP_MAPPING="false"
#     done

GENERATIONS=20
POPULATION=100

for COUNT in 4 5 6 7; do
    MODEL_SUMMARY="resources/model_summaries/example_summaries/${model_type}/${file_basename}_multi_${COUNT}.json"
    OUTPUT_NAME="results_${platform}_multi_${COUNT}.csv"
    make dse USBMON=0 MODEL=$MODEL MODEL_SUMMARY=$MODEL_SUMMARY \
        OUTPUT_FOLDER=$OUTPUT_FOLDER OUTPUT_NAME=$OUTPUT_NAME POPULATION_SIZE=$POPULATION GENERATIONS=$GENERATIONS \
        ILP_MAPPING="false"
    done

# GENERATIONS=20
# POPULATION=200

# for COUNT in 1 2 3 4; do
#     MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$COUNT.json"
#     OUTPUT_NAME="results_rpi_P$POPULATION\_G$GENERATIONS\_$COUNT.csv"
#     make dse USBMON=0 MODEL=$MODEL PROFILING_COSTS=$PROFILING_RESULTS MODEL_SUMMARY=$MODEL_SUMMARY \
#         OUTPUT_FOLDER=$OUTPUT_FOLDER OUTPUT_NAME=$OUTPUT_NAME POPULATION_SIZE=$POPULATION GENERATIONS=$GENERATIONS \
#         ILP_MAPPING="false"
#     done
