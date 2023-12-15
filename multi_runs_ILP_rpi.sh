#!/bin/bash

MODEL="resources/models/example_models/key_word_spotting/kws_ref_model.tflite"
filename_with_extension="${MODEL##*/}"
file_basename="${filename_with_extension%.*}"

model_type="key_word_spotting"

platform="rpi"

OUTPUT_FOLDER="resources/ILP_results/${file_basename}"
if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi


for COUNT in 3; do
    MODEL_SUMMARY="resources/model_summaries/example_summaries/${model_type}/${file_basename}_multi_${COUNT}.json"
    OUTPUT_NAME="results_${platform}_multi_${COUNT}.csv"
    make dse USBMON=0 MODEL=$MODEL PLATFORM=$platform MODEL_SUMMARY=$MODEL_SUMMARY \
        OUTPUT_FOLDER=$OUTPUT_FOLDER OUTPUT_NAME=$OUTPUT_NAME ILP_MAPPING="true" VERBOSE="true"
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
