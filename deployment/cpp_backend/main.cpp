#include <iostream>
#include <vector>

#include "backend.h" 

int main(int argc, char *argv[])
{
    std::vector<uint8_t> input_data(0, 100);
    std::vector<uint8_t> output_data(0, 100);

    const unsigned int input_data_size = 100;
    const unsigned int output_data_size = 100;
    
    unsigned int inference_time;

    inference_time = distributed_inference_wrapper("random", input_data.data(), output_data.data(), input_data_size, output_data_size, "TPU", 1);
    
    return 0;
}