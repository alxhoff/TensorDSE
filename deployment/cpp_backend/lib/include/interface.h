#ifndef _INTERFACE_H
#define _INTERFACE_H

#include<iostream>
#include<vector>
#include "backend.h"

typedef struct {
    int* data;
    int size;
} IntVector;

#ifdef __cplusplus
extern "C" {
#endif
  extern void* distributed_inference_interface(uint8_t* input_data, int benchmarking_count)
  {
    std::vector<int> result;
    IntVector* output = new IntVector;

    result = distributed_inference_wrapper(input_data, benchmarking_count);

    output->data = result.data();
    output->size = result.size();

    return output;
  }
#ifdef __cplusplus
}
#endif

#endif