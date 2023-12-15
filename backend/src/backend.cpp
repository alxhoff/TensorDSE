#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <string>

// Common
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// GPU
#define CL_TARGET_OPENCL_VERSION 220
#include "tensorflow/lite/delegates/gpu/delegate.cc"
#include "tensorflow/lite/delegates/gpu/delegate.h"

// TPU
#include "edgetpu.h"
#include "edgetpu_c.h"

#include "backend.h"
#include "tflite_utils.h"

#include "logger.h"

int distributed_inference_cpu(std::string& tflite_model_path, int8_t* input_data,
                                                              int8_t* output_data,
                                                              uint32_t* inference_times,
                                                              const unsigned int input_data_size,
                                                              const unsigned int output_data_size,
                                                              const unsigned int benchmarking_count) {

    // Load the model
    spdlog::info("  [1]  Loading TFLite Model from: {}", tflite_model_path);
    std::unique_ptr<tflite::FlatBufferModel> model;
    model = LoadModelFile(tflite_model_path);
    if (model == nullptr) {
        spdlog::error("  [1]  Failed to load model from {}", tflite_model_path); 
        return -1;
    }
    spdlog::info("  [1]  Done!");

    // Create interpreter Object
    spdlog::info("  [2]  Creating Interpreter Object ...");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    spdlog::info("  [2]  Done!");


    // Allocate tensors
    spdlog::info("  [3]  Allocating Tensors ...");
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        spdlog::error("  [3]  Failed to allocate tensors!");
        return 1;
    }
    spdlog::info("  [3]  Done!");

    // Invoke
    spdlog::info("  [4]  Invoking Interpreter ...");
    std::vector<uint32_t> inference_times_vec(benchmarking_count, 0);

    for (int i = 0; i < benchmarking_count; i++) {
        std::vector<int8_t> randomVector
            = generateRandomVector(input_data_size);
        std::vector<int8_t> data(input_data, input_data + input_data_size);

        // Copy Input Data into the Input Tensor
        if (benchmarking_count > 1) {
            std::copy(randomVector.begin(), randomVector.end(),
                interpreter->typed_input_tensor<int8_t>(0));
        } else {
            std::copy(data.begin(), data.end(),
                interpreter->typed_input_tensor<int8_t>(0));
        }

        auto inference_start = std::chrono::high_resolution_clock::now();
        if (interpreter->Invoke() != kTfLiteOk) {
            spdlog::error("  [4]  Cannot invoke interpreter!");
            return 1;
        }
        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_time_ns
            = std::chrono::duration_cast<std::chrono::nanoseconds>(
                inference_end - inference_start)
                    .count();
        inference_times_vec[i] = inference_time_ns;
    }

    std::vector<int8_t> final_data 
        = GetTensorData(*interpreter->output_tensor(0));

    std::copy(final_data.begin(), final_data.end(), output_data);

    std::copy(inference_times_vec.begin(), inference_times_vec.end(),
        inference_times);

    int mean = calculateMean(inference_times_vec);

    spdlog::info("  [5]  Interpreter successfully invoked ({} times)!", benchmarking_count);


    return mean;
}


int distributed_inference_gpu(std::string& tflite_model_path, int8_t* input_data,
                                                              int8_t* output_data,
                                                              uint32_t* inference_times,
                                                              const unsigned int input_data_size,
                                                              const unsigned int output_data_size,
                                                              const unsigned int benchmarking_count) {


    // Load the model
    spdlog::info("  [1]  Loading TFLite Model from: {}", tflite_model_path);
    std::unique_ptr<tflite::FlatBufferModel> model;
    model = LoadModelFile(tflite_model_path);
    if (model == nullptr) {
        spdlog::error("Failed to load model from: {}", tflite_model_path);
        return -1;
    }
    spdlog::info("  [1]  Done!");

    // Create interpreter Object
    spdlog::info("  [2]  Creating Interpreter Object ...");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    spdlog::info("  [2]  Done!");

    // Allocate tensors
    spdlog::info("  [3]  Allocating Tensors ...");
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        spdlog::error("  [3]  Failed to allocate tensors!");
        return 1;
    }
    spdlog::info("  [3]  Done!");

    // Create GPU Delegate
    spdlog::info("  [4]  Creating GPU Delegate ...");
    const TfLiteGpuDelegateOptionsV2 options 
        = TfLiteGpuDelegateOptionsV2Default();
    auto* delegate = TfLiteGpuDelegateV2Create(&options);
    builder.AddDelegate(delegate);
    spdlog::info("  [4]  Done!");


    // Invoke
    spdlog::info("  [5]  Invoking Interpreter ...");
    std::vector<uint32_t> inference_times_vec(benchmarking_count, 0);

    for (int i = 0; i < benchmarking_count; i++) {
        std::vector<int8_t> randomVector
            = generateRandomVector(input_data_size);
        std::vector<int8_t> data(input_data, input_data + input_data_size);

        // Copy Input Data into the Input Tensor
        if (benchmarking_count > 1) {
            std::copy(randomVector.begin(), randomVector.end(), 
                interpreter->typed_input_tensor<int8_t>(0));
        } else {
            std::copy(data.begin(), data.end(),
                interpreter->typed_input_tensor<int8_t>(0));
        }

        auto inference_start = std::chrono::high_resolution_clock::now();
        if (interpreter->Invoke() != kTfLiteOk) {
            spdlog::error("  [5]  Cannot invoke interpreter");
            return 1;
        }
        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_time_ns 
            = std::chrono::duration_cast<std::chrono::nanoseconds>(
                inference_end - inference_start)
                .count();
        inference_times_vec[i] = inference_time_ns;
    }

    std::vector<int8_t> final_data 
        = GetTensorData(*interpreter->output_tensor(0));

    std::copy(final_data.begin(), final_data.end(), output_data);
    std::copy(inference_times_vec.begin(), inference_times_vec.end(),
        inference_times);

    int mean = calculateMean(inference_times_vec);

    spdlog::info("  [5]  Interpreter successfully invoked ({} times)!", benchmarking_count);

    return mean;
}


int distributed_inference_tpu_rpi(std::string& tflite_model_path, int8_t* input_data,
                                                              int8_t* output_data,
                                                              uint32_t* inference_times,
                                                              const unsigned int input_data_size,
                                                              const unsigned int output_data_size,
                                                              const unsigned int benchmarking_count,
                                                              const unsigned int core_index) {

    // Find TPU device.
    spdlog::info("  [1]  Detecting Edge TPUs Devices ...");
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
    if (num_devices == 0) {
        spdlog::error("  [1]  No connected USB Accelerator is found!");
        return -1;
    }
    const auto& device = devices.get()[core_index];
    const auto& available_tpus 
        = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    spdlog::info("  [1]  Number of available Edge TPU USB Accelerators: {}", available_tpus.size());// hopefully we'll see 1 here

    // Load the model
    spdlog::info("  [2]  Loading TFLite Model from: {}", tflite_model_path);
    std::unique_ptr<tflite::FlatBufferModel> model;
    model = LoadModelFile(tflite_model_path);
    if (model == nullptr) {
        spdlog::error("  [2]  Failed to load model from: {}!", tflite_model_path);
        return -1;
    }
    spdlog::info("  [2]  Done!");
    
    // Create interpreter.
    spdlog::info("  [3]  Creating Interpreter Object ...");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) 
        != kTfLiteOk) {
        spdlog::error("  [3]  Cannot create interpreter!");
        return -1;
    }
    spdlog::info("  [3]  Done!");

    spdlog::info("  [4]  Creating Edge TPU delegate ...");
    auto* delegate 
        = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
    interpreter->ModifyGraphWithDelegate(delegate);
    spdlog::info("  [4]  Done");

    // Allocate tensors 
    spdlog::info("  [5]  Allocating Tensors ...");
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        spdlog::error("  [5]  Failed to allocate tensors!");
        return -1;
    }
    spdlog::info("  [5]  Done!");

    // Invoke
    spdlog::info("  [6]  Invoking Interpreter ...");
    std::vector<uint32_t> inference_times_vec(benchmarking_count, 0);
    for (int i = 0; i < benchmarking_count; i++) {
        std::vector<int8_t> randomVector 
            = generateRandomVector(input_data_size);
        std::vector<int8_t> data(input_data, input_data + input_data_size);

        // Copy Input Data into the Input Tensor
        if (benchmarking_count > 1) {
            std::copy(randomVector.begin(), randomVector.end(), 
                interpreter->typed_input_tensor<int8_t>(0));
        } else {
            std::copy(data.begin(), data.end(), 
                interpreter->typed_input_tensor<int8_t>(0));
        }

        auto inference_start = std::chrono::high_resolution_clock::now();
        if (interpreter->Invoke() != kTfLiteOk) {
            spdlog::error("  [6]  Cannot invoke interpreter");
            return 1;
        }
        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_time_ns 
            = std::chrono::duration_cast<std::chrono::nanoseconds>(
                inference_end - inference_start)
                .count();
        inference_times_vec[i] = inference_time_ns;
    }

    std::vector<int8_t> final_data 
        = GetTensorData(*interpreter->output_tensor(0));

    std::copy(final_data.begin(), final_data.end(), output_data);
    std::copy(inference_times_vec.begin(), inference_times_vec.end(), 
        inference_times);

    int mean = calculateMean(inference_times_vec);

    spdlog::info("  [6]  Interpreter successfully invoked({} times)!", benchmarking_count);

    interpreter.reset();

    return mean;

}


int distributed_inference_tpu_std(std::string& tflite_model_path, int8_t* input_data,
                                                              int8_t* output_data,
                                                              uint32_t* inference_times,
                                                              const unsigned int input_data_size,
                                                              const unsigned int output_data_size,
                                                              const unsigned int benchmarking_count,
                                                              const unsigned int core_index) {

    // Find TPU device.
    spdlog::info("  [1]  Detecting Edge TPUs Devices ...");
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
    if (num_devices == 0) {
        std::cerr << "No connected USB Accelerator is found!" << std::endl;
        return -1;
    }
    const auto& device = devices.get()[core_index];
    const auto& available_tpus 
        = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    spdlog::info("  [1]  Number of available Edge TPU USB Accelerators: {}", available_tpus.size());

    // Load the model
    spdlog::info("  [2]  Loading TFLite Model from: {}", tflite_model_path);
    std::unique_ptr<tflite::FlatBufferModel> model;
    model = LoadModelFile(tflite_model_path);
    if (model == nullptr) {
        spdlog::error("  [2]  Failed to load model from {}!", tflite_model_path);
        return -1;
    }
    spdlog::info("  [2]  Done!");

    spdlog::info("  [3]  Creating Interpreter Object and Edge TPU Context ...");
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(available_tpus[0].type, available_tpus[0].path);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
    interpreter->SetNumThreads(4);
    spdlog::info("  [3]  Done!");

    // Allocate tensors 
    spdlog::info("  [4]  Allocating Tensors ...");
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        spdlog::error("  [4]  Failed to allocate tensors!");
        return -1;
    }
    spdlog::info("  [4]  Done!");

    // Invoke
    spdlog::info("  [5]  Invoking Interpreter ...");
    std::vector<uint32_t> inference_times_vec(benchmarking_count, 0);
    for (int i = 0; i < benchmarking_count; i++) {
        std::vector<int8_t> randomVector 
            = generateRandomVector(input_data_size);
        std::vector<int8_t> data(input_data, input_data + input_data_size);

        // Copy Input Data into the Input Tensor
        if (benchmarking_count > 1) {
            std::copy(randomVector.begin(), randomVector.end(), 
                interpreter->typed_input_tensor<int8_t>(0));
        } else {
            std::copy(data.begin(), data.end(), 
                interpreter->typed_input_tensor<int8_t>(0));
        }

        auto inference_start = std::chrono::high_resolution_clock::now();
        if (interpreter->Invoke() != kTfLiteOk) {
            spdlog::error("  [5]  Cannot invoke interpreter!");
            return 1;
        }
        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_time_ns 
            = std::chrono::duration_cast<std::chrono::nanoseconds>(
                inference_end - inference_start)
                .count();
        inference_times_vec[i] = inference_time_ns;
    }

    std::vector<int8_t> final_data 
        = GetTensorData(*interpreter->output_tensor(0));

    std::copy(final_data.begin(), final_data.end(), output_data);
    std::copy(inference_times_vec.begin(), inference_times_vec.end(), 
        inference_times);

    int mean = calculateMean(inference_times_vec);

    spdlog::info("  [5]  Interpreter successfully invoked ({} times)!", benchmarking_count);

    interpreter.reset();

    return mean;
}


int distributed_inference_wrapper(std::string& tflite_model_path, int8_t* input_data,
                                                                  int8_t* output_data,
                                                                  uint32_t* inference_times,
                                                                  const unsigned int input_data_size,
                                                                  const unsigned int output_data_size,
                                                                  std::string& hardware_target,
                                                                  std::string& platform,
                                                                  const unsigned int benchmarking_count,
                                                                  const unsigned int core_index
                                                                  ) {
   
    int result = 0;

    setup_logger();
    spdlog::info("Logger initialized!");
    
    spdlog::info("TFLite Model Path: {}", tflite_model_path);
    spdlog::info("HW Target: {}", hardware_target);
    spdlog::info("Execution Count: {}", benchmarking_count);
    spdlog::info("Platform: {}", platform);

    if (hardware_target.compare("cpu") == 0) {
        result = distributed_inference_cpu(tflite_model_path, input_data,
                                                              output_data,
                                                              inference_times,
                                                              input_data_size,
                                                              output_data_size,
                                                              benchmarking_count);
    } else if (hardware_target.compare("gpu") == 0) {
        result = distributed_inference_gpu(tflite_model_path, input_data,
                                                              output_data,
                                                              inference_times,
                                                              input_data_size,
                                                              output_data_size,
                                                              benchmarking_count);
    } else if (hardware_target.compare("tpu") == 0) {
        if (platform.compare("rpi") == 0) {
            result = distributed_inference_tpu_rpi(tflite_model_path, input_data,
                                                              output_data,
                                                              inference_times,
                                                              input_data_size,
                                                              output_data_size,
                                                              benchmarking_count,
                                                              core_index);

        } else {
            result = distributed_inference_tpu_std(tflite_model_path, input_data,
                                                              output_data,
                                                              inference_times,
                                                              input_data_size,
                                                              output_data_size,
                                                              benchmarking_count,
                                                              core_index);
        }
        
    } else {
        spdlog::error("This Hardware Target is not supported!");
        result = -1;
    }
    spdlog::drop("-----------------------------------------------------------------");
    return result;
}
    