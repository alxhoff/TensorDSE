#!/bin/bash

for COUNT in 1 2 3
do
    echo make run COUNT=20 MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$COUNT.json" OUTPUT_NAME="results_$COUNT.csv" POPULATION_SIZE=50 GENERATIONS=25
    make run COUNT=20 MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$COUNT.json" OUTPUT_NAME="results_$COUNT.csv" POPULATION_SIZE=50 GENERATIONS=25
done
