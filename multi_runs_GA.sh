#!/bin/bash

GENERATIONS=20
POPULATION=500
BENCHMARK_TEST_COUNT=20

for COUNT in 1
do
    MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$COUNT.json"
    make nodeploy USBMON=0 COUNT=$BENCHMARK_TEST_COUNT MODEL_SUMMARY=$MODEL_SUMMARY OUTPUT_NAME="results_$COUNT.csv" POPULATION_SIZE=$POPULATION GENERATIONS=$GENERATIONS
done
