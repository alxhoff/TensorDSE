#include<iostream>
#include<vector>
#include <chrono>

// Common
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// GPU
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.cc"

// TPU
#include "edgetpu.h"
#include "edgetpu_c.h"

#include "backend.h"

int distributed_inference(uint8_t* input_data, uint8_t* output_data, 
                                    const unsigned int input_data_size, const unsigned int output_data_size) {

    // Find TPU device.
    std::cout << "######################### Check Edge TPU USB Accelerator #########################" << "\n";
    std::cout << "\n" << std::endl;
    std::cout << " Detecting Edge TPUs Devices ..." << "\n";
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
    if (num_devices == 0) {
        std::cerr << "No connected USB Accelerator is found!" << std::endl;
        return 1;
    }
    const auto& device = devices.get()[0];
    const auto& available_tpus = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    std::cout << "Number of available Edge TPU USB Accelerators: " << available_tpus.size() << "\n"; // hopefully we'll see 1 here
    std::cout << "\n" << std::endl;

    auto inference_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < input_data_size; i++) {
        input_data[i] += i;
        std::cout << "Input Nr. " << i << ": " << static_cast<unsigned int>(input_data[i]) << "\n";
        output_data[i] += i+input_data_size;
        std::cout << "Output Nr. " << i << ": " << static_cast<unsigned int>(output_data[i]) << "\n";
    }
    auto inference_end = std::chrono::high_resolution_clock::now();
    auto total_inference_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(inference_end - inference_start).count();
    return total_inference_time_ns;
}


int distributed_inference_wrapper(uint8_t* input_data, uint8_t* output_data,
                                            const unsigned int input_data_size, const unsigned int output_data_size) {
    int result = 0;
    result = distributed_inference(input_data, output_data,
                            input_data_size, output_data_size);
    return result;
}