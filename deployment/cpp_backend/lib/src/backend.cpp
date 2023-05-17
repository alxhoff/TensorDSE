#include<iostream>
#include<vector>
#include <chrono>

// Common
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// GPU
#define CL_TARGET_OPENCL_VERSION 220
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.cc"

// TPU
#include "edgetpu.h"
#include "edgetpu_c.h"

#include "tflite_utils.h"
#include "backend.h"

int distributed_inference(uint8_t* input_data, uint8_t* output_data, 
                                    const unsigned int input_data_size, const unsigned int output_data_size) {
        // Initiating Distributed Inference
    std::cout << "##################################################################################" << "\n";
    std::cout << "#         Running Distributed Inference of an Image Classification Model         #" << "\n";
    std::cout << "##################################################################################" << "\n";
    std::cout << "\n" << std::endl;

    // Load labels.
    std::cout << "################################# Loading Labels #################################" << "\n";
    std::cout << "\n" << std::endl;
    const std::string label_file   = "/home/starkaf/Documents/uni/master/fp/TensorDSE/deployment/resources/labels/labels_mobilenet_quant_v1_224.txt";
    auto labels = ReadLabels(label_file);
    if (labels.empty()) {
        std::cerr << "Cannot read labels from " << label_file << std::endl;
        return 1;
    }
    std::cout << "Labels successfully loaded from: " << label_file << "\n";
    std::cout << "\n" << std::endl;

    // Load image.
    std::cout << "################################# Loading Image ##################################" << "\n";
    std::cout << "\n" << std::endl;
    int image_bpp, image_width, image_height;
    const std::string image_file = "/home/starkaf/Documents/uni/master/fp/TensorDSE/deployment/resources/images/resize_bird.bmp"; 

    const std::vector<uint8_t>& image =
        read_bmp(image_file.c_str(), &image_width, &image_height, &image_bpp);

    if (image.empty()) {
        std::cerr << "Cannot read image from " << image_file << std::endl;
        return 1;
    }
    std::cout << "Image successfully loaded from: " << image_file << "\n";
    std::cout << "\n" << std::endl;

    // Load models section
    std::cout << "################################## Loading Models ################################" << "\n";
    std::cout << "\n" << std::endl;

        std::unique_ptr<tflite::FlatBufferModel> model_0;
    model_0 = LoadModelFile("/home/starkaf/Documents/uni/master/fp/TensorDSE/deployment/resources/models/image_classification/mobilenet/mobilenet_v1_1_0_224_quant.tflite");
    if (!model_0) {
        std::cerr << "Cannot load model from " << "/home/starkaf/Documents/uni/master/fp/TensorDSE/deployment/resources/models/image_classification/mobilenet/mobilenet_v1_1_0_224_quant.tflite" << std::endl;
        return 1;
    }
    std::cout << "mobilenet_v1_1_0_224_quant.tflite  successfully loaded! " << std::endl;


    // Find TPU device.
    std::cout << "######################### Check Edge TPU USB Accelerator #########################" << "\n";
    std::cout << "\n" << std::endl;
    std::cout << "Detecting Edge TPUs Devices ..." << "\n";
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
    if (num_devices == 0) {
        std::cerr << "No connected USB Accelerator is found!" << std::endl;
        //return 1;
    }
    const auto& device = devices.get()[0];
    const auto& available_tpus = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    std::cout << "Number of available Edge TPU USB Accelerators: " << available_tpus.size() << "\n"; // hopefully we'll see 1 here
    std::cout << "\n" << std::endl;



    std::cout << "############################ Build GPU Interpreter ############################" << std::endl;
    std::cout << "\n" << std::endl;
    tflite::ops::builtin::BuiltinOpResolver resolver_0;
    tflite::InterpreterBuilder builder_1(*model_0, resolver_0);
    std::unique_ptr<tflite::Interpreter> interpreter_0;
    builder_1(&interpreter_0);
    // Allocate tensors 
    if (interpreter_0->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return 1;
    }
    std::cout << "Tensors successfully allocated." << "\n";  // NEW: Prepare GPU delegate.

    const TfLiteGpuDelegateOptionsV2 options_0 = TfLiteGpuDelegateOptionsV2Default();
    auto* delegate_0 = TfLiteGpuDelegateV2Create(&options_0);
    if (interpreter_0->ModifyGraphWithDelegate(delegate_0) != kTfLiteOk) { // Experimental: tflite::InterpreterBuilder::AddDelegate();
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
        return 1;
    }
    std::cout << "\n" << std::endl;

    // Load Data into Input Interpreter
    const auto* input_tensor = interpreter_0->input_tensor(0);
    if (input_tensor->type != kTfLiteUInt8 ||           //
        input_tensor->dims->data[0] != 1 ||             //
        input_tensor->dims->data[1] != image_height ||  //
        input_tensor->dims->data[2] != image_width ||   //
        input_tensor->dims->data[3] != image_bpp) {
        std::cerr << "Input tensor shape does not match input image" << std::endl;
        return 1;
    }

    std::copy(image.begin(), image.end(),
                interpreter_0->typed_input_tensor<uint8_t>(0));

    auto inference_start = std::chrono::high_resolution_clock::now();

    if (interpreter_0->Invoke() != kTfLiteOk) {
        std::cerr << "Cannot invoke interpreter" << std::endl;
        return 1;
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