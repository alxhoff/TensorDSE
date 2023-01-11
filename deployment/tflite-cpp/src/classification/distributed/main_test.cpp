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
  std::cout << "\n" << std::endl;
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
    std::unique_ptr<tflite::FlatBufferModel> model_0;
  model_0 = LoadSubmodel("/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel0_cpu1_0_1_2_3_4.tflite");
  if (!model_0) {
    std::cerr << "Cannot load model from " << "/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel0_cpu1_0_1_2_3_4.tflite" << std::endl;
    return 1;
  }
  std::cout << "submodel0_cpu1_0_1_2_3_4.tflite successfully loaded! " << std::endl;
  std::unique_ptr<tflite::FlatBufferModel> model_1;
  model_1 = LoadSubmodel("/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel1_gpu1_5_6_7_8_9.tflite");
  if (!model_1) {
    std::cerr << "Cannot load model from " << "/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel1_gpu1_5_6_7_8_9.tflite" << std::endl;
    return 1;
  }
  std::cout << "submodel1_gpu1_5_6_7_8_9.tflite successfully loaded! " << std::endl;
  std::unique_ptr<tflite::FlatBufferModel> model_2;
  model_2 = LoadSubmodel("/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel2_tpu1_10_11_12_13_14_edgetpu.tflite");
  if (!model_2) {
    std::cerr << "Cannot load model from " << "/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel2_tpu1_10_11_12_13_14_edgetpu.tflite" << std::endl;
    return 1;
  }
  std::cout << "submodel2_tpu1_10_11_12_13_14_edgetpu.tflite successfully loaded! " << std::endl;
  std::unique_ptr<tflite::FlatBufferModel> model_3;
  model_3 = LoadSubmodel("/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel3_cpu2_15_16_17_18_19.tflite");
  if (!model_3) {
    std::cerr << "Cannot load model from " << "/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel3_cpu2_15_16_17_18_19.tflite" << std::endl;
    return 1;
  }
  std::cout << "submodel3_cpu2_15_16_17_18_19.tflite successfully loaded! " << std::endl;
  std::unique_ptr<tflite::FlatBufferModel> model_4;
  model_4 = LoadSubmodel("/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel4_gpu2_20_21_22_23_24.tflite");
  if (!model_4) {
    std::cerr << "Cannot load model from " << "/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel4_gpu2_20_21_22_23_24.tflite" << std::endl;
    return 1;
  }
  std::cout << "submodel4_gpu2_20_21_22_23_24.tflite successfully loaded! " << std::endl;
  std::unique_ptr<tflite::FlatBufferModel> model_5;
  model_5 = LoadSubmodel("/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel5_tpu2_25_26_27_28_29_30_edgetpu.tflite");
  if (!model_5) {
    std::cerr << "Cannot load model from " << "/home/starkaf/Documents/projects/EdgeTpuModelOptimizer/models/sub/tflite/submodel5_tpu2_25_26_27_28_29_30_edgetpu.tflite" << std::endl;
    return 1;
  }
  std::cout << "submodel5_tpu2_25_26_27_28_29_30_edgetpu.tflite successfully loaded! " << std::endl;

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
    std::cout << "############################ Build CPU1 Interpreter ############################" << std::endl;
  std::cout << "\n" << std::endl;
  tflite::ops::builtin::BuiltinOpResolver resolver_0;
  tflite::InterpreterBuilder builder_0(*model_0, resolver_0);
  std::unique_ptr<tflite::Interpreter> interpreter_0;
  builder_0(&interpreter_0);
  // Allocate tensors
  if (interpreter_0->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  std::cout << "Tensors successfully allocated." << "\n";
  std::cout << "\n" << std::endl;

  std::cout << "############################ Build GPU1 Interpreter ############################" << std::endl;
  std::cout << "\n" << std::endl;
  tflite::ops::builtin::BuiltinOpResolver resolver_1;
  tflite::InterpreterBuilder builder_1(*model_1, resolver_1);
  std::unique_ptr<tflite::Interpreter> interpreter_1;
  builder_1(&interpreter_1);
  // Allocate tensors 
  if (interpreter_1->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  std::cout << "Tensors successfully allocated." << "\n";  // NEW: Prepare GPU delegate.
  const TfLiteGpuDelegateOptionsV2 options_1 = TfLiteGpuDelegateOptionsV2Default();
  auto* delegate_1 = TfLiteGpuDelegateV2Create(&options_1);
  if (interpreter_1->ModifyGraphWithDelegate(delegate_1) != kTfLiteOk) { // Experimental: tflite::InterpreterBuilder::AddDelegate();
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
    return false;
  }
  std::cout << "\n" << std::endl;

  std::cout << "############################ Build TPU1 Interpreter ############################" << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "Initializing Edge TPU Context ... " << "\n";
  const std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context_2 = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(available_tpus[0].type, available_tpus[0].path);
  tflite::ops::builtin::BuiltinOpResolver resolver_2;
  resolver_2.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::cout << "Building Edge TPU Interpreter ... " << "\n";
  tflite::InterpreterBuilder builder_2(*model_2, resolver_2);
  std::unique_ptr<tflite::Interpreter> interpreter_2;
  builder_2(&interpreter_2);
  interpreter_2->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context_2.get());
  interpreter_2->SetNumThreads(1);
  // Allocate tensors 
  if (interpreter_2->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  std::cout << "Tensors successfully allocated." << "\n";  std::cout << "\n" << std::endl;

  std::cout << "############################ Build CPU2 Interpreter ############################" << std::endl;
  std::cout << "\n" << std::endl;
  tflite::ops::builtin::BuiltinOpResolver resolver_3;
  tflite::InterpreterBuilder builder_3(*model_3, resolver_3);
  std::unique_ptr<tflite::Interpreter> interpreter_3;
  builder_3(&interpreter_3);
  // Allocate tensors
  if (interpreter_3->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  std::cout << "Tensors successfully allocated." << "\n";
  std::cout << "\n" << std::endl;

  std::cout << "############################ Build GPU2 Interpreter ############################" << std::endl;
  std::cout << "\n" << std::endl;
  tflite::ops::builtin::BuiltinOpResolver resolver_4;
  tflite::InterpreterBuilder builder_4(*model_4, resolver_4);
  std::unique_ptr<tflite::Interpreter> interpreter_4;
  builder_4(&interpreter_4);
  // Allocate tensors 
  if (interpreter_4->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  std::cout << "Tensors successfully allocated." << "\n";  // NEW: Prepare GPU delegate.
  const TfLiteGpuDelegateOptionsV2 options_4 = TfLiteGpuDelegateOptionsV2Default();
  auto* delegate_4 = TfLiteGpuDelegateV2Create(&options_4);
  if (interpreter_4->ModifyGraphWithDelegate(delegate_4) != kTfLiteOk) { // Experimental: tflite::InterpreterBuilder::AddDelegate();
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
    return false;
  }
  std::cout << "\n" << std::endl;

  std::cout << "############################ Build TPU2 Interpreter ############################" << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "Initializing Edge TPU Context ... " << "\n";
  const std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context_5 = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(available_tpus[0].type, available_tpus[0].path);
  tflite::ops::builtin::BuiltinOpResolver resolver_5;
  resolver_5.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::cout << "Building Edge TPU Interpreter ... " << "\n";
  tflite::InterpreterBuilder builder_5(*model_5, resolver_5);
  std::unique_ptr<tflite::Interpreter> interpreter_5;
  builder_5(&interpreter_5);
  interpreter_5->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context_5.get());
  interpreter_5->SetNumThreads(1);
  // Allocate tensors 
  if (interpreter_5->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  std::cout << "Tensors successfully allocated." << "\n";  std::cout << "\n" << std::endl;



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

  // Run Inference and pass intermediate tensors
  std::cout << "############################## Initiating Inference ###############################" << "\n";
  std::cout << "\n" << std::endl;
  std::chrono::steady_clock::time_point start, end;
  start = std::chrono::steady_clock::now();
  
    if (interpreter_0->Invoke() != kTfLiteOk) {
    std::cerr << "Cannot invoke interpreter" << std::endl;
    return 1;
  }
  const auto* intermediate_tensor_0 = interpreter_0->output_tensor(0);

  *interpreter_1->input_tensor(0) = *intermediate_tensor_0;
  if (interpreter_1->Invoke() != kTfLiteOk) {
    std::cerr << "Cannot invoke interpreter" << std::endl;
    return 1;
  }
  const auto* intermediate_tensor_1 = interpreter_1->output_tensor(0);

  *interpreter_2->input_tensor(0) = *intermediate_tensor_1;
  if (interpreter_2->Invoke() != kTfLiteOk) {
    std::cerr << "Cannot invoke interpreter" << std::endl;
    return 1;
  }
  const auto* intermediate_tensor_2 = interpreter_2->output_tensor(0);

  *interpreter_3->input_tensor(0) = *intermediate_tensor_2;
  if (interpreter_3->Invoke() != kTfLiteOk) {
    std::cerr << "Cannot invoke interpreter" << std::endl;
    return 1;
  }
  const auto* intermediate_tensor_3 = interpreter_3->output_tensor(0);

  *interpreter_4->input_tensor(0) = *intermediate_tensor_3;
  if (interpreter_4->Invoke() != kTfLiteOk) {
    std::cerr << "Cannot invoke interpreter" << std::endl;
    return 1;
  }
  const auto* intermediate_tensor_4 = interpreter_4->output_tensor(0);

  *interpreter_5->input_tensor(0) = *intermediate_tensor_4;
  if (interpreter_5->Invoke() != kTfLiteOk) {
    std::cerr << "Cannot invoke interpreter" << std::endl;
    return 1;
  }



  end = std::chrono::steady_clock::now();
  auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Inference is successful!" << "\n";
  std::cout << "\n" << std::endl;

  std::cout << "############################ Estimating Inference Time ############################" << "\n";
  std::cout << "\n" << std::endl;
  std::cout << "Infernce Time: " + std::to_string(inference_time) + "ms" << std::endl;
  std::cout << "\n" << std::endl;
  

  // Read Final Results from Output Interpreter
  std::cout << "############################ Listing Inference Results ############################" << "\n";
  std::cout << "\n" << std::endl;
  auto results = Sort(Dequantize(*interpreter_5->output_tensor(0)), threshold);
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



