#!/bin/bash

PROFILING_RESULTS="../../resources/profiling_results/rpi"
MODEL="MNIST_full_quanitization.json"
OUTPUT_FOLDER="../../resources/ILP_results"

for COUNT in 6; do
    MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$COUNT.json"
    OUTPUT_NAME="results_rpi_$COUNT.csv"
    make dse MODEL=$MODEL PROFILING_COSTS=$PROFILING_RESULTS MODEL_SUMMARY=$MODEL_SUMMARY \
        OUTPUT_FOLDER=$OUTPUT_FOLDER OUTPUT_NAME=$OUTPUT_NAME ILP_MAPPING="true" VERBOSE="true"
    done
