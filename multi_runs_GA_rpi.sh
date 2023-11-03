#!/bin/bash

PROFILING_RESULTS="../../resources/profiling_results/rpi"
OUTPUT_FOLDER="../../resources/ILP_results"

MODEL="resources/models/example_models/MNIST/MNIST_full_quanitization.tflite"
file_basename="${MODEL%.*}"

model_type="MNIST"

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
    MODEL_SUMMARY="../../resources/model_summaries/example_summaries/${model_type}/${file_basename}_multi_${COUNT}.json"
    OUTPUT_NAME="results_rpi_P$POPULATION\_G$GENERATIONS\_$COUNT.csv"
    make dse USBMON=0 MODEL=$MODEL PROFILING_COSTS=$PROFILING_RESULTS MODEL_SUMMARY=$MODEL_SUMMARY \
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
