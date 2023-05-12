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
#include <jsoncpp/json/value.h>

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


namespace fs = std::experimental::filesystem;

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
  
  bool CheckEdgeAcc() {
    // Check if Coral USB Accelerator is connected!
    bool edgetpu_check = true;
    // Find TPU device.
    std::cout << "Detecting Edge TPUs Devices ..." << "\n";
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), 
                                                                            &edgetpu_free_devices);
    if (num_devices == 0) {
      std::cerr << "No connected TPU found" << std::endl;
      edgetpu_check=false;
    }

    const auto& available_tpus = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();

    if (available_tpus.size() < 1) {
      std::cerr << "Edge TPU USB Accelerator found, but not available" << std::endl;
      edgetpu_check=false;
    } 
    std::cout << "Number of available Edge TPUs: " << available_tpus.size() << "\n"; // hopefully we'll see 1 here
    std::cout << "\n" << std::endl;
    
    return edgetpu_check;
  }
  
  std::vector<std::string> ReadModelsPaths(std::string directory) {
  
    std::vector<std::string> result;
  
    // Read Models Directory
    std::vector<std::string> cpu_submodel_paths{};
    std::vector<std::string> gpu_submodel_paths{};
    std::vector<std::string> tpu_submodel_paths{};
    // This structure would distinguish a file from a directory
    struct stat sb;
    for (const auto& entry : fs::directory_iterator(directory)) {
      // Converting the path of a file to a const char *
      fs::path outfilename = entry.path();
      std::string outfilename_str = outfilename.string();
      const char* path = outfilename_str.c_str();
      // Check if path points to a non-directory. If True, display path
      if (stat(path, &sb) == 0 && !(sb.st_mode & S_IFDIR))
        if (outfilename_str.find("cpu") != std::string::npos) {
          //std ::cout << "CPU:" << outfilename_str << std::endl;
          cpu_submodel_paths.push_back(outfilename_str);
        } else if (outfilename_str.find("edgetpu") != std::string::npos) {
          //std ::cout << "TPU:" << outfilename_str << std::endl;
          tpu_submodel_paths.push_back(outfilename_str);
        } else if (outfilename_str.find("gpu") != std::string::npos) {
          //std ::cout << "GPU:" << outfilename_str << std::endl;
          gpu_submodel_paths.push_back(outfilename_str);
        } 
    }
    
    for (int i = 0; i < cpu_submodel_paths.size(); i++) {
      result.push_back(cpu_submodel_paths[i]);
    }

    for (int i = 0; i < gpu_submodel_paths.size(); i++) {
      result.push_back(gpu_submodel_paths[i]);
    }

    if (tpu_submodel_paths.size() != 0) {
      for (int i = 0; i < tpu_submodel_paths.size(); i++) {
        result.push_back(tpu_submodel_paths[i]);
      }
      bool check;
      check = CheckEdgeAcc();
    }
    
    std::sort(result.begin(), result.end());
    return result;
  }
  
  std::vector<std::vector<std::string>> ReadCsvMapping(std::string filepath) {
    std::vector<std::vector<std::string>> content;
    std::vector<std::string> row;
    std::string line, word;

    std::fstream file(filepath, std::ios::in);

    if(file.is_open()) {
      while(std::getline(file, line)) {
        row.clear();
        std::stringstream str(line);
        while(std::getline(str, word, ',')) {
          row.push_back(word);
        }
        content.push_back(row);
      }
    } else {
      std::cout<<"Could not open the file\n";
    }
      
    for(int i=0;i<content.size();i++) {
      for(int j=0;j<content[i].size();j++) {
        std::cout<<content[i][j]<<" ";
      }
      std::cout<<"\n";
    }

    return content;
  }

  // Could be separated from this namespace later
  std::unique_ptr<tflite::FlatBufferModel> LoadSubmodel(std::string filepath) {
    std::unique_ptr<tflite::FlatBufferModel> submodel =
    tflite::FlatBufferModel::BuildFromFile(filepath.c_str());
    return submodel;
  }

  bool CheckModelForTPU(const std::string model_path) {
    const std::string edgtpu_str = "edgetpu.tflite";
    if (std::strstr(model_path.c_str(), edgtpu_str.c_str())) {
      return true;
    }
    return false;
  }

  bool CheckModelForGPU(const std::string model_path) {
    const std::string gpu_str = "gpu";
    if (std::strstr(model_path.c_str(), gpu_str.c_str())) {
      return true;
    }
    return false;
  }

  int GetDevice(const std::string model_path) {
    int result;
    if (CheckModelForTPU(model_path)) {
      result = 2;
    } else if (CheckModelForGPU(model_path)) {
      result = 1;
    } else {
      result = 0;
    }
    return result;
  }

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << argv[0]
              << " <models_dir> <label_file> <image_file> <threshold>"
              << std::endl;
    return 1;
  }

  const std::string models_dir   = argv[1];
  const std::string label_file   = argv[2];
  const std::string image_file   = argv[3];
  const float threshold          = std::stof(argv[4]);

  // Find TPU device.
  std::cout << "Detecting Edge TPUs Devices ..." << "\n";
  size_t num_devices;
  std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
      edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

  if (num_devices == 0) {
    std::cerr << "No connected TPU found" << std::endl;
    //return 1;
  }
  const auto& device = devices.get()[0];

  const auto& available_tpus = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  std::cout << "Number of available Edge TPUs: " << available_tpus.size() << "\n"; // hopefully we'll see 1 here
  std::cout << "\n" << std::endl;

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

  std::vector<std::string> paths = {};

  paths = ReadModelsPaths(models_dir);

  for (int i = 0; i < paths.size(); i++) {
    // Check Target Device
    int device_id;
    device_id = GetDevice(paths[i]);
    
    // Check For Input/Output
    bool input,output;
    input  = i == 0;
    output = i == paths.size()-1;

    // Load Model
    std::unique_ptr<tflite::FlatBufferModel> submodel;
    submodel = LoadSubmodel(paths[i]);
    if (!submodel) {
      std::cerr << "Cannot load Submodel from " << paths[i] << std::endl;
      return 1;
    } else {
      std::cout << "Submodel successfully loaded from: " << paths[i] << "\n";
    }

    // Create Interpreter Object
    std::unique_ptr<tflite::Interpreter> interpreter;

    std::vector<uint8_t> intermediate_tensor;

    switch (device_id) {
      case 0:
        {
          // Create interpreter.
          tflite::ops::builtin::BuiltinOpResolver resolver;
          tflite::InterpreterBuilder builder(*submodel, resolver);
          builder(&interpreter);
          // Allocate tensors
          if (interpreter->AllocateTensors() != kTfLiteOk) {
            std::cerr << "Failed to allocate tensors." << std::endl;
          }
          std::cout << "Tensors successfully allocated." << "\n";
          break;
        }
      case 1:
        {
          // Create interpreter.
          tflite::ops::builtin::BuiltinOpResolver resolver;
          tflite::InterpreterBuilder builder(*submodel, resolver);
          builder(&interpreter);
          // Allocate tensors
          if (interpreter->AllocateTensors() != kTfLiteOk) {
            std::cerr << "Failed to allocate tensors." << std::endl;
          }
          std::cout << "Tensors successfully allocated." << "\n";
          // Prepare GPU delegate.
          const TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
          auto* delegate = TfLiteGpuDelegateV2Create(&options);
          if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
            fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
            return false;
          } 
          break;
        }
      case 2:
        {
          std::cout << "Initializing Edge TPU Context ... " << "\n";
          const std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
            edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(available_tpus[0].type, available_tpus[0].path);
          // Create interpreter.
          tflite::ops::builtin::BuiltinOpResolver resolver;
          resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
          std::cout << "Building Edge TPU Interpreter ... " << "\n";
          tflite::InterpreterBuilder builder(*submodel, resolver);
          builder(&interpreter);
          // Bind given context with interpreter.
          interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
          interpreter->SetNumThreads(1);
          if (interpreter->AllocateTensors() != kTfLiteOk) {
            std::cerr << "Failed to allocate tensors." << std::endl;
          }
          std::cout << "Tensors successfully allocated." << "\n";
          break;
        }
    }

    if (input) {
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
      std::copy(image.begin(), image.end(), interpreter->typed_input_tensor<uint8_t>(0));
      std::cout << "\n" << std::endl;
      // Invoke Interpreter
      if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Cannot invoke interpreter" << std::endl;
        return 1;
      }

      std::copy(interpreter->output_tensor(0)->begin(), interpreter->output_tensor(0)->end(), intermediate_tensor.begin());
      std::cout << interpreter->output_tensor(0)->bytes << std::endl;
    } 
//    else if (output) {
//      std::copy(intermediate_tensor.begin(), intermediate_tensor.end(), interpreter->typed_input_tensor<uint8_t>(0));
//      // Invoke Interpreter
//      if (interpreter->Invoke() != kTfLiteOk) {
//        std::cerr << "Cannot invoke interpreter" << std::endl;
//        return 1;
//      }
//      // Get interpreter output.
//      auto results = Sort(Dequantize(*interpreter->output_tensor(0)), threshold);
//      std::cout << "Results are sorted with decreasing likelihood:"<< "\n";
//      int i = 1;
//      for (auto& result : results) {
//        /*std::cout << " - " << i << " - " << GetLabel(labels, result.first) << ": " << std::setw(7) << std::fixed << std::setprecision(5)
//              << result.second << std::endl;*/
//        std::cout << " - " << i << " - " << GetLabel(labels, result.first) << ": " << result.second * 100 << "%" << std::endl;
//        i++;
//      }
//      break;
//
//    } else {
//      std::copy(intermediate_tensor.begin(), intermediate_tensor.end(), interpreter->typed_input_tensor<uint8_t>(0));
//      // Invoke Interpreter
//      if (interpreter->Invoke() != kTfLiteOk) {
//        std::cerr << "Cannot invoke interpreter" << std::endl;
//        return 1;
//      }
//      std::copy(interpreter->typed_output_tensor<uint8_t>(0).begin(), interpreter->typed_output_tensor<uint8_t>(0).end(), intermediate_tensor.begin());
//    }
     
  }
  return 0;
}