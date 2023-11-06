#!/bin/bash

MODEL_SUMMARY_FOLDER="../../resources/model_summaries/example_summaries"
MODEL_SUMMARY_NAME="MLperf_ad.json"
PLATFORM="rpi"
PROFILING_COSTS_FOLDER="../../resources/profiling_results"
OUTPUT_FOLDER="../../resources/GA_results"
OUTPUT_NAME="results_single_run.csv"
POPULATION_SIZE=100
GENERATIONS=20
COUNT=20
VERBOSE="true"

for i in "$@"; do
    case $i in
    -p=* | --POPULATION=*)
        POPULATION="${i#*=}"
        shift # past argument=value
        ;;
    -g=* | --GENERATIONS=*)
        GENERATIONS="${i#*=}"
        shift # past argument=value
        ;;
    -m=* | --MODEL_SUMMARY_NAME=*)
        MODEL_SUMMARY_NAME="${i#*=}"
        shift # past argument=value
        ;;
    -o=* | --OUTPUT_FOLDER=*)
        OUTPUT_FOLDER="${i#*=}"
        shift # past argument=value
        ;;
    -p=* | --PROFILING_COSTS=*)
        PROFILING_COSTS="${i#*=}"
        shift # past argument=value
        ;;
    -l=* | --PLATFORM=*)
        PLATFORM="${i#*=}"
        shift # past argument=value
        ;;
    -d=* | --COUNT=*)
        COUNT="${i#*=}"
        shift # past argument=value
        ;;
    -d=* | --VERBOSE=*)
        VERBOSE="${i#*=}"
        shift # past argument=value
        ;;
    --default)
        DEFAULT=YES
        shift # past argument with no value
        ;;
    -* | --*) ;;
    *) ;;
    esac
done

MODEL_SUMMARY="$MODEL_SUMMARY_FOLDER/$MODEL_SUMMARY_NAME"
PROFILING_COSTS="$PROFILING_COSTS_FOLDER/$PLATFORM"

make dse USBMON=0 COUNT=$COUNT MODEL_SUMMARY=$MODEL_SUMMARY OUTPUT_FOLDER=$OUTPUT_FOLDER PROFILING_COSTS=$PROFILING_COSTS OUTPUT_NAME=$OUTPUT_NAME ILP_MAPPING="true" POPULATION_SIZE=$POPULATION_SIZE GENERATIONS=$GENERATIONS VERBOSE=$VERBOSE
