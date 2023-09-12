#!/bin/bash

for i in "$@"; do
    case $i in
    -m=* | --MODEL_SUMMARY=*)
        MODEL_SUMMARY="${i#*=}"
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

for COUNT in 1 2 3 4 5; do
    make nodeploy COUNT=20 MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$MODEL_SUMMARY.json" OUTPUT_FOLDER="../../resources/GA_results" OUTPUT_NAME="results_m$MODEL_SUMMARY_$COUNT.csv" ILP_MAPPING="false" POPULATION_SIZE=50 GENERATIONS=50
done
