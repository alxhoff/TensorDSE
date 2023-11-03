#!/bin/bash

PROFILING_RESULTS="../../resources/profiling_results/rpi"
OUTPUT_FOLDER="../../resources/ILP_results"

MODEL="resources/models/example_models/MNIST/MNIST_full_quanitization.tflite"
file_basename="${MODEL%.*}"

model_type="MNIST"


for COUNT in 1 2 3; do
    MODEL_SUMMARY="../../resources/model_summaries/example_summaries/${model_type}/${file_basename}_multi_${COUNT}.json"
    OUTPUT_NAME="results_rpi_$COUNT.csv"
    make dse USBMON=0 MODEL=$MODEL PROFILING_COSTS=$PROFILING_RESULTS MODEL_SUMMARY=$MODEL_SUMMARY \
        OUTPUT_FOLDER=$OUTPUT_FOLDER OUTPUT_NAME=$OUTPUT_NAME ILP_MAPPING="true" VERBOSE="true"
    done
