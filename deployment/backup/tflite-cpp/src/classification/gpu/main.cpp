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

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.cc"

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
}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << argv[0]
              << " <model_file> <label_file> <image_file> <threshold>"
              << std::endl;
    return 1;
  }

  const std::string model_file = argv[1];
  const std::string label_file = argv[2];
  const std::string image_file = argv[3];
  const float threshold = std::stof(argv[4]);

  // Load labels.
  std::cout << "Loading resources ..." << "\n";
  auto labels = ReadLabels(label_file);
  if (labels.empty()) {
    std::cerr << "Cannot read labels from " << label_file << std::endl;
    return 1;
  }
  std::cout << "Labels successfully loaded from: " << label_file << "\n";

  // Load image.
  int image_bpp, image_width, image_height;

  const std::vector<uint8_t>& image =
      read_bmp(image_file.c_str(), &image_width, &image_height, &image_bpp);

  if (image.empty()) {
    std::cerr << "Cannot read image from " << image_file << std::endl;
    return 1;
  }
  std::cout << "Image successfully loaded from: " << image_file << "\n";

  // Load model.
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
  if (!model) {
    std::cerr << "Cannot read model from " << model_file << std::endl;
    return 1;
  }
  std::cout << "Model successfully loaded from: " << model_file << "\n";
  std::cout << "\n" << std::endl;

  // Create interpreter.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);

  // Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  std::cout << "Tensors successfully allocated." << "\n";


  // NEW: Prepare GPU delegate.
  const TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
  auto* delegate = TfLiteGpuDelegateV2Create(&options);
  if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) { // Experimental: tflite::InterpreterBuilder::AddDelegate();
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
    return false;
  } 

  // Set interpreter input.
  const auto* input_tensor = interpreter->input_tensor(0);
  if (input_tensor->type != kTfLiteUInt8 ||           //
      input_tensor->dims->data[0] != 1 ||             //
      input_tensor->dims->data[1] != image_height ||  //
      input_tensor->dims->data[2] != image_width ||   //
      input_tensor->dims->data[3] != image_bpp) {
    std::cerr << "Input tensor shape does not match input image" << std::endl;
    return 1;
  }

  std::cout << "Loading Image into Interpreter's Input Tensor ..." << "\n";
  std::copy(image.begin(), image.end(),
            interpreter->typed_input_tensor<uint8_t>(0));
  std::cout << "\n" << std::endl;

  // Run inference.
  std::cout << "Initiating Inference ..." << "\n";
  std::chrono::steady_clock::time_point start, end;
  start = std::chrono::steady_clock::now();
  if (interpreter->Invoke() != kTfLiteOk) {
    std::cerr << "Cannot invoke interpreter" << std::endl;
    return 1;
  }
  end = std::chrono::steady_clock::now();
  auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Inference is successful" << "\n";
  std::cout << "\n" << std::endl;
  std::cout << "Infernce Time: " + std::to_string(inference_time) + "ms" << std::endl;
  std::cout << "\n" << std::endl;

  // Get interpreter output.
  auto results = Sort(Dequantize(*interpreter->output_tensor(0)), threshold);
  std::cout << "Results are sorted with decreasing likelihood:"<< "\n";
  int i = 1;
  for (auto& result : results) {
    /*std::cout << " - " << i << " - " << GetLabel(labels, result.first) << ": " << std::setw(7) << std::fixed << std::setprecision(5)
              << result.second << std::endl;*/
    std::cout << " - " << i << " - " << GetLabel(labels, result.first) << ": " << result.second * 100 << "%" << std::endl;
    i++;
  }
  // NEW: Clean up.
  TfLiteGpuDelegateV2Delete(delegate);

  return 0;
}