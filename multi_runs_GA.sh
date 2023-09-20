#!/bin/bash

GENERATIONS=50
POPULATION=200
BENCHMARK_TEST_COUNT=20

for COUNT in 1 2 3 4 5
do
    # POPULATION=$(( COUNT * 40 ))
    MODEL_SUMMARY="../../resources/model_summaries/example_summaries/MNIST/MNIST_multi_$COUNT.json"
    echo make nodeploy USBMON=0 COUNT=$BENCHMARK_TEST_COUNT MODEL_SUMMARY=$MODEL_SUMMARY OUTPUT_NAME="results_$COUNT.csv" POPULATION_SIZE=$POPULATION GENERATIONS=$GENERATIONS
    make nodeploy USBMON=0 COUNT=$BENCHMARK_TEST_COUNT MODEL_SUMMARY=$MODEL_SUMMARY OUTPUT_NAME="results_$COUNT.csv" POPULATION_SIZE=$POPULATION GENERATIONS=$GENERATIONS
done
