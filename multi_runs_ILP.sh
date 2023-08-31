#!/bin/bash

for COUNT in 1 2 3 4 5
do
    echo make run COUNT=20 MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$COUNT.json" OUTPUT_FOLDER="../../resources/ILP_results" OUTPUT_NAME="results_$COUNT.csv" ILP_MAPPING="true"
    make run COUNT=20 MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$COUNT.json" OUTPUT_FOLDER="../../resources/ILP_results" OUTPUT_NAME="results_$COUNT.csv" ILP_MAPPING="true"
done
