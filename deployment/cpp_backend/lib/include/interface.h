#ifndef _INTERFACE_H
#define _INTERFACE_H

#include<iostream>
#include<vector>

#include "backend.h"

//typedef struct {
//    int* data;
//    int size;
//} IntVector;


extern "C" {
  extern int distributed_inference_interface(uint8_t* input_data, uint8_t* output_data, const unsigned int input_data_size, const unsigned int output_data_size) {
    //std::vector<int> result;
    //IntVector* output = new IntVector;
    int result = 0;
    result = distributed_inference_wrapper(input_data, output_data, input_data_size, output_data_size);
    return result;
  }
}

#endif