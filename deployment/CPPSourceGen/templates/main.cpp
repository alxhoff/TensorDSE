#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <filesystem>
#include <experimental/filesystem>
#include <sys/stat.h>

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

namespace {
  constexpr size_t kBmpFileHeaderSize = 14;
  constexpr size_t kBmpInfoHeaderSize = 40;
  constexpr size_t kBmpHeaderSize = kBmpFileHeaderSize + kBmpInfoHeaderSize;
  
  std::vector<uint8_t> decode_bmp(const uint8_t* input, int row_size, int width,
                                  int height, int channels, bool top_down) {
    std::vector<uint8_t> output(height * width * channels);
    for (int i = 0; i < height; i++) {
      int src_pos;
      int dst_pos;
  
      for (int j = 0; j < width; j++) {
        if (!top_down) {
          src_pos = ((height - 1 - i) * row_size) + j * channels;
        } else {
          src_pos = i * row_size + j * channels;
        }
  
        dst_pos = (i * width + j) * channels;
  
        switch (channels) {
          case 1:
            output[dst_pos] = input[src_pos];
            break;
          case 3:
            // BGR -> RGB
            output[dst_pos] = input[src_pos + 2];
            output[dst_pos + 1] = input[src_pos + 1];
            output[dst_pos + 2] = input[src_pos];
            break;
          case 4:
            // BGRA -> RGBA
            output[dst_pos] = input[src_pos + 2];
            output[dst_pos + 1] = input[src_pos + 1];
            output[dst_pos + 2] = input[src_pos];
            output[dst_pos + 3] = input[src_pos + 3];
            break;
          default:
            std::cerr << "Unexpected number of channels: " << channels
                      << std::endl;
            std::abort();
            break;
        }
      }
    }
    return output;
  }
  
  std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
                                int* height, int* channels) {
    int begin, end;
  
    std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
    if (!file) {
      std::cerr << "input file " << input_bmp_name << " not found\n";
      std::abort();
    }
  
    begin = file.tellg();
    file.seekg(0, std::ios::end);
    end = file.tellg();
    size_t len = end - begin;
  
    std::vector<uint8_t> img_bytes(len);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(img_bytes.data()), len);
    const int32_t header_size =
        *(reinterpret_cast<const int32_t*>(img_bytes.data() + 10));
    *width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 18));
    *height = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 22));
    const int32_t bpp =
        *(reinterpret_cast<const int32_t*>(img_bytes.data() + 28));
    *channels = bpp / 8;
  
    // there may be padding bytes when the width is not a multiple of 4 bytes
    // 8 * channels == bits per pixel
    const int row_size = (8 * *channels * *width + 31) / 32 * 4;
  
    // if height is negative, data layout is top down
    // otherwise, it's bottom up
    bool top_down = (*height < 0);
  
    // Decode image, allocating tensor once the image size is known
    const uint8_t* bmp_pixels = &img_bytes[header_size];
    return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels,
                      top_down);
  }
  
  int32_t ToInt32(const char p[4]) {
    return (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
  }
  
  std::vector<uint8_t> ReadBmpImage(const char* filename,
                                    int* out_width = nullptr,
                                    int* out_height = nullptr,
                                    int* out_channels = nullptr) {
    assert(filename);
  
  
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      std::cerr << "Cannot Open File" << std::endl;
      return {};  // Open failed.
    }
    char header[kBmpHeaderSize];
    if (!file.read(header, sizeof(header))) return {};  // Read failed.
  
    const char* file_header = header;
    const char* info_header = header + kBmpFileHeaderSize;
  
    if (file_header[0] != 'B' || file_header[1] != 'M')
      return {};  // Invalid file type.
  
    const int channels = info_header[14] / 8;
    if (channels != 1 && channels != 3) return {};  // Unsupported bits per pixel.
  
    if (ToInt32(&info_header[16]) != 0) return {};  // Unsupported compression.
  
    const uint32_t offset = ToInt32(&file_header[10]);
    if (offset > kBmpHeaderSize &&
        !file.seekg(offset - kBmpHeaderSize, std::ios::cur))
      return {};  // Seek failed.
  
    int width = ToInt32(&info_header[4]);
    if (width < 0) return {};  // Invalid width.
  
    int height = ToInt32(&info_header[8]);
    const bool top_down = height < 0;
    if (top_down) height = -height;
  
    const int line_bytes = width * channels;
    const int line_padding_bytes =
        4 * ((8 * channels * width + 31) / 32) - line_bytes;
    std::vector<uint8_t> image(line_bytes * height);
    for (int i = 0; i < height; ++i) {
      uint8_t* line = &image[(top_down ? i : (height - 1 - i)) * line_bytes];
      if (!file.read(reinterpret_cast<char*>(line), line_bytes))
        return {};  // Read failed.
      if (!file.seekg(line_padding_bytes, std::ios::cur))
        return {};  // Seek failed.
      if (channels == 3) {
        for (int j = 0; j < width; ++j) std::swap(line[3 * j], line[3 * j + 2]);
      }
    }
  
    if (out_width) *out_width = width;
    if (out_height) *out_height = height;
    if (out_channels) *out_channels = channels;
    return image;
  }
  
  std::vector<std::string> ReadLabels(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) return {};  // Open failed.
  
    std::vector<std::string> lines;
    for (std::string line; std::getline(file, line);) lines.emplace_back(line);
    return lines;
  }
  
  std::string GetLabel(const std::vector<std::string>& labels, int label) {
    if (label >= 0 && label < labels.size()) return labels[label];
    return std::to_string(label);
  }
  
  std::vector<float> Dequantize(const TfLiteTensor& tensor) {
    const auto* data = reinterpret_cast<const uint8_t*>(tensor.data.data);
    std::vector<float> result(tensor.bytes);
    for (int i = 0; i < tensor.bytes; ++i)
      result[i] = tensor.params.scale * (data[i] - tensor.params.zero_point);
    return result;
  }
  
  std::vector<std::pair<int, float>> Sort(const std::vector<float>& scores,
                                          float threshold) {
    std::vector<const float*> ptrs(scores.size());
    std::iota(ptrs.begin(), ptrs.end(), scores.data());
    auto end = std::partition(ptrs.begin(), ptrs.end(),
                              [=](const float* v) { return *v >= threshold; });
    std::sort(ptrs.begin(), end,
              [](const float* a, const float* b) { return *a > *b; });
  
    std::vector<std::pair<int, float>> result;
    result.reserve(end - ptrs.begin());
    for (auto it = ptrs.begin(); it != end; ++it)
      result.emplace_back(*it - scores.data(), **it);
    return result;
  }
  
  std::unique_ptr<tflite::FlatBufferModel> LoadModelFile(std::string filepath) {
    std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(filepath.c_str());
    return model;
  }

}  // namespace


int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << argv[0]
              << " <label_file> <image_file> <threshold>"
              << std::endl;
    return 1;
  }

  const std::string label_file   = argv[1];
  const std::string image_file   = argv[2];
  const float threshold          = std::stof(argv[3]);
  
  // Initiating Distributed Inference
  std::cout << "##################################################################################" << "\n";
  std::cout << "#         Running Distributed Inference of an Image Classification Model         #" << "\n";
  std::cout << "##################################################################################" << "\n";
  std::cout << "\n" << std::endl;

  // Load labels.
  std::cout << "################################# Loading Labels #################################" << "\n";
  std::cout << "\n" << std::endl;
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

  {{load_models_section}}

  std::cout << "\n" << std::endl;


  // Find TPU device.
  std::cout << "######################### Check Edge TPU USB Accelerator #########################" << "\n";
  std::cout << "\n" << std::endl;
  std::cout << "Detecting Edge TPUs Devices ..." << "\n";
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

  // Create Interpreters
  std::cout << "##################################################################################" << "\n";
  std::cout << "#                               Build Interpreters                               #" << "\n";
  std::cout << "##################################################################################" << "\n";
  std::cout << "\n" << std::endl;

  {{build_interpreters_section}}

  // Load Data into Input Interpreter
  const auto* input_tensor = {{input_interpreter}}->input_tensor(0);
  if (input_tensor->type != kTfLiteUInt8 ||           //
      input_tensor->dims->data[0] != 1 ||             //
      input_tensor->dims->data[1] != image_height ||  //
      input_tensor->dims->data[2] != image_width ||   //
      input_tensor->dims->data[3] != image_bpp) {
    std::cerr << "Input tensor shape does not match input image" << std::endl;
    return 1;
  }

  std::copy(image.begin(), image.end(),
            {{input_interpreter}}->typed_input_tensor<uint8_t>(0));

  // Run Inference and pass intermediate tensors
  std::cout << "############################## Initiating Inference ###############################" << "\n";
  std::cout << "\n" << std::endl;

  auto total_inference_start = std::chrono::high_resolution_clock::now();
  {{invoke_interpreters_section}}
  auto total_inference_end = std::chrono::high_resolution_clock::now();
  
  auto total_inference_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(total_inference_end - total_inference_start).count();
  auto total_inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_inference_end - total_inference_start).count();

  {{exact_inference_time_section}}

  std::cout << "Inference is successful!" << "\n";
  std::cout << "\n" << std::endl;

  std::cout << "############################## Inference Times ##############################" << "\n";
  std::cout << "\n" << std::endl;
  std::cout << "Total Infernce Time in ms: " + std::to_string(total_inference_time_ms) << std::endl;
  std::cout << "Exact Infernce Time in ms: " + std::to_string(exact_inference_time_ms) << std::endl;
  std::cout << "Difference in ms: " + std::to_string(total_inference_time_ms - exact_inference_time_ms) << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "Total Infernce Time in ns: " + std::to_string(total_inference_time_ns) << std::endl;
  std::cout << "Exact Infernce Time in ns: " + std::to_string(exact_inference_time_ns) << std::endl;
  std::cout << "Difference in ns: " + std::to_string(total_inference_time_ns - exact_inference_time_ns) << std::endl;

  std::cout << "\n" << std::endl;
  

  // Read Final Results from Output Interpreter
  std::cout << "############################ Listing Inference Results ############################" << "\n";
  std::cout << "\n" << std::endl;
  auto results = Sort(Dequantize(*{{output_interpreter}}->output_tensor(0)), threshold);
  std::cout << "Results are sorted with decreasing likelihood:"<< "\n";
  int i = 1;
  for (auto& result : results) {
    /*std::cout << " - " << i << " - " << GetLabel(labels, result.first) << ": " << std::setw(7) << std::fixed << std::setprecision(5)
              << result.second << std::endl;*/
    std::cout << " - " << i << " - " << GetLabel(labels, result.first) << ": " << result.second * 100 << "%" << std::endl;
    i++;
  }
  std::cout << "\n" << std::endl;

  return 0;
}





