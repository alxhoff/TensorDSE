#ifndef _BACKEND_H
#define _BACKEND_H

#include<vector>
#include<iostream>

int distributed_inference_wrapper(std::string& tflite_model_path, void* input_data,
                                                                  void* output_data,
                                                                  uint32_t* inference_times,
                                                                  uint32_t* data_times,
                                                                  const unsigned int input_data_size,
                                                                  const unsigned int output_data_size,
                                                                  std::string& hardware_target,
                                                                  std::string& platform,
                                                                  const unsigned int benchmarking_count,
                                                                  const unsigned int core_index,
                                                                  const unsigned int data_type);


#endif