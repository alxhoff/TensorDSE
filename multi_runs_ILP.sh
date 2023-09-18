#!/bin/bash

for COUNT in 1 2 3 4
do
    echo make nodeploy COUNT=20 MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$COUNT.json" OUTPUT_FOLDER="../../resources/ILP_results" OUTPUT_NAME="results_$COUNT.csv" ILP_MAPPING="true"
    make nodeploy COUNT=20 MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$COUNT.json" OUTPUT_FOLDER="../../resources/ILP_results" OUTPUT_NAME="results_$COUNT.csv" ILP_MAPPING="true" VERBOSE="true"
done
