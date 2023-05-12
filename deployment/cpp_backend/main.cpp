#include <iostream>
#include <vector>

#include "backend.h" 

int main(int argc, char *argv[])
{
    std::vector<uint8_t> input_data;
    std::vector<int> inference_times;

    inference_times = distributed_inference_wrapper(input_data.data(), 10);
    
    return 0;
}