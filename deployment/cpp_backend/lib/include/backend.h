#ifndef _BACKEND_H
#define _BACKEND_H

#include<vector>
#include<iostream>

std::vector<int> distributed_inference_wrapper(uint8_t* input_data, int benchmarking_count);

#endif