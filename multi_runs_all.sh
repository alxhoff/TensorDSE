#!/bin/bash

MODEL="resources/models/example_models/MNIST/MNIST_full_quanitization.tflite"
filename_with_extension="${MODEL##*/}"
file_basename="${filename_with_extension%.*}"

platform="rpi"

OUTPUT_FOLDER="resources/GA_results/${file_basename}"
if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

PROFILING_RESULTS="resources/profiling_results/${platform}/model_0_${file_basename}.json"

GENERATIONS=20
POPULATION=100

for COUNT in 1 2 3; do
    MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/${file_basename}_multi_$COUNT.json"
    OUTPUT_NAME="results_${platform}_P$POPULATION\_G$GENERATIONS\_$COUNT.csv"
    make dse USBMON=0 MODEL=$MODEL PROFILING_COSTS=$PROFILING_RESULTS MODEL_SUMMARY=$MODEL_SUMMARY \
        OUTPUT_FOLDER=$OUTPUT_FOLDER OUTPUT_NAME=$OUTPUT_NAME POPULATION_SIZE=$POPULATION GENERATIONS=$GENERATIONS \
        ILP_MAPPING="false" VERBOSE="true"
        
    MAPPINGS_FILE="${OUTPUT_FOLDER}/mappings.json"
    NEW_MAPPINGS_FILE="${OUTPUT_FOLDER}/results_${platform}_multi_${COUNT}_mappings.json"

    # Check if MAPPINGS_FILE exists and is a valid file path
    if [ -f "$MAPPINGS_FILE" ]; then
        # If it exists, then move it
        mv "$MAPPINGS_FILE" "$NEW_MAPPINGS_FILE"
        echo "Moved $MAPPINGS_FILE to $NEW_MAPPINGS_FILE"
    else
        echo "The file $MAPPINGS_FILE does not exist."
    fi
    done
