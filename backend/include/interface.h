#ifndef _INTERFACE_H
#define _INTERFACE_H

#include <iostream>
#include <vector>
#include <string>
#include "backend.h"

extern "C" {
  extern int distributed_inference_interface(char* tflite_model_path_ptr, int8_t* input_data,
                                                                          int8_t* output_data,
                                                                          uint32_t* inference_times,
                                                                          const unsigned int input_data_size,
                                                                          const unsigned int output_data_size,
                                                                          char* hardware_target_ptr,
                                                                          char* platform_ptr,
                                                                          const unsigned int benchmarking_count,
                                                                          const unsigned int core_index) {
    std::string tflite_model_path(tflite_model_path_ptr);
    std::string hardware_target(hardware_target_ptr);
    std::string platform(platform_ptr);
    int result = 0;
    result = distributed_inference_wrapper(tflite_model_path, input_data,
                                                              output_data,
                                                              inference_times,
                                                              input_data_size,
                                                              output_data_size,
                                                              hardware_target,
                                                              platform,
                                                              benchmarking_count,
                                                              core_index);
    return result;
  }
}

#endif