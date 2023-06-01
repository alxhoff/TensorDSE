#ifndef _BACKEND_H
#define _BACKEND_H

#include<vector>
#include<iostream>

int distributed_inference_wrapper(std::string tflite_model_path, int8_t* input_data,
                                                                 int8_t* output_data,
                                                                 const unsigned int input_data_size,
                                                                 const unsigned int output_data_size,
                                                                 std::string hardware_target,
                                                                 const unsigned int benchmarking_count);

#endif