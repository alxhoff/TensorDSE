#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

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

int distributed_inference_cpu(std::string& tflite_model_path,
    int8_t* input_data, int8_t* output_data, uint32_t* inference_times,
    const unsigned int input_data_size, const unsigned int output_data_size,
    const unsigned int benchmarking_count)
{

    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model;
    model = LoadModelFile(tflite_model_path);
    if (model == nullptr) {
        std::cerr << "Failed to load model from " << tflite_model_path
                  << std::endl;
        return -1;
    }

    // Create interpreter Object
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return 1;
    }

    // Invoke
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
            std::cerr << "Cannot invoke interpreter" << std::endl;
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

    return mean;
}

int distributed_inference_gpu(std::string& tflite_model_path,
    int8_t* input_data, int8_t* output_data, uint32_t* inference_times,
    const unsigned int input_data_size, const unsigned int output_data_size,
    const unsigned int benchmarking_count)
{

    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model;
    model = LoadModelFile(tflite_model_path);
    if (model == nullptr) {
        std::cerr << "Failed to load model from " << tflite_model_path
                  << std::endl;
        return -1;
    }

    // Create interpreter Object
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return 1;
    }

    // Create GPU Delegate
    const TfLiteGpuDelegateOptionsV2 options
        = TfLiteGpuDelegateOptionsV2Default();
    auto* delegate = TfLiteGpuDelegateV2Create(&options);
    builder.AddDelegate(delegate);

    // Invoke
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
            std::cerr << "Cannot invoke interpreter" << std::endl;
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

    return mean;
}

int distributed_inference_tpu(std::string& tflite_model_path,
    int8_t* input_data, int8_t* output_data, uint32_t* inference_times,
    const unsigned int input_data_size, const unsigned int output_data_size,
    const unsigned int benchmarking_count, const unsigned int core_index)
{

    // Find TPU device.
    std::cout << " Detecting Edge TPUs Devices ..." << std::endl;
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
    std::cout << "Number of available Edge TPU USB Accelerators: "
              << available_tpus.size()
              << std::endl; // hopefully we'll see 1 here

    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model;
    model = LoadModelFile(tflite_model_path);
    if (model == nullptr) {
        std::cerr << "Failed to load model from " << tflite_model_path
                  << std::endl;
        return -1;
    }

    const std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context
        = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
            available_tpus[0].type, available_tpus[0].path);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    interpreter->SetExternalContext(
        kTfLiteEdgeTpuContext, edgetpu_context.get());
    interpreter->SetNumThreads(1);

    std::cout << "Edge TPU Interpreter successfully built!" << std::endl;
    * /

        // Create interpreter.
        tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter)
        != kTfLiteOk) {
        std::cerr << "Cannot create interpreter" << std::endl;
        return -1;
    }

    auto* delegate
        = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
    interpreter->ModifyGraphWithDelegate(delegate);

    // Allocate tensors
    std::cout << "Allocating Tensors... " << std::endl;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return -1;
    }

    // Invoke
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
            std::cerr << "Cannot invoke interpreter" << std::endl;
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

    edgetpu_free_delegate(delegate);

    std::copy(final_data.begin(), final_data.end(), output_data);
    std::copy(inference_times_vec.begin(), inference_times_vec.end(),
        inference_times);

    int mean = calculateMean(inference_times_vec);

    return mean;
}

int distributed_inference_wrapper(std::string& tflite_model_path,
    int8_t* input_data, int8_t* output_data, uint32_t* inference_times,
    const unsigned int input_data_size, const unsigned int output_data_size,
    std::string& hardware_target, const unsigned int benchmarking_count,
    const unsigned int core_index)
{
    int result = 0;
    if (hardware_target.compare("cpu") == 0) {
        result = distributed_inference_cpu(tflite_model_path, input_data,
            output_data, inference_times, input_data_size, output_data_size,
            benchmarking_count);
    } else if (hardware_target.compare("gpu") == 0) {
        result = distributed_inference_gpu(tflite_model_path, input_data,
            output_data, inference_times, input_data_size, output_data_size,
            benchmarking_count);
    } else if (hardware_target.compare("tpu") == 0) {
        result = distributed_inference_tpu(tflite_model_path, input_data,
            output_data, inference_times, input_data_size, output_data_size,
            benchmarking_count, core_index);
    } else {
        std::cerr << "This Hardware Target is not supported!" << std::endl;
        result = -1;
    }
    return result;
}
