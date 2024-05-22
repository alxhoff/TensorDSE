#ifndef _BACKEND_H
#define _BACKEND_H

#include<vector>
#include<iostream>

enum DataType {
    INT8,
    UINT8
};

// Modify the wrapper function to accept the data type
int distributed_inference_wrapper(const std::string& tflite_model_path,
                                  void* input_data,
                                  void* output_data,
                                  uint32_t* inference_times,
                                  uint32_t* data_times,
                                  const unsigned int input_data_size,
                                  const unsigned int output_data_size,
                                  const std::string& hardware_target,
                                  const std::string& platform,
                                  const unsigned int benchmarking_count,
                                  const unsigned int core_index,
                                  DataType data_type);

#endif