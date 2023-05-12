#include<iostream>
#include<vector>
#include "backend.h"


int distributed_inference(uint8_t* input_data) {
    return 0;
}


std::vector<int> distributed_inference_wrapper(uint8_t* input_data, int benchmarking_count) {
    std::vector<int> inference_times(benchmarking_count, 0);
    for (int i = 0; i < benchmarking_count ; i++) {
        int inference_time = 0;
        inference_time = distributed_inference(input_data);
        inference_times.push_back(inference_time);
    }
    return inference_times;
}