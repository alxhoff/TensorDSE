#ifndef _TFLITE_UTILS_H
#define _TFLITE_UTILS_H

#include <vector>
#include <stdint.h>
#include <string>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/c/common.h"

std::vector<uint8_t> decode_bmp(const uint8_t* input, int row_size, int width,
                                int height, int channels, bool top_down);

std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
                            int* height, int* channels);

int32_t ToInt32(const char p[4]);

std::vector<uint8_t> ReadBmpImage(const char* filename,
                                int* out_width,
                                int* out_height,
                                int* out_channels);

std::vector<std::string> ReadLabels(const std::string& filename);

std::string GetLabel(const std::vector<std::string>& labels, int label);

std::vector<float> Dequantize(const TfLiteTensor& tensor);

std::vector<std::pair<int, float>> Sort(const std::vector<float>& scores,
                                        float threshold);

std::unique_ptr<tflite::FlatBufferModel> LoadModelFile(std::string filepath);

int calculateMean(const std::vector<int>& values);

std::vector<uint8_t> GetTensorData(const TfLiteTensor& tensor);

#endif