#ifndef _BACKEND_H
#define _BACKEND_H

#include<vector>
#include<iostream>

int distributed_inference_wrapper(uint8_t* input_data, uint8_t* output_data, const unsigned int input_data_size, const unsigned int output_data_size);

#endif