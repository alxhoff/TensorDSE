import os, sys
import argparse
import numpy as np

import ModelLab.split as split
from ModelLab.logger import log
from ModelLab.utils import ReadJSON
#from CPPSourceGen.generate import GenerateSource

sys.path.append('py_backend')
import cpp_backend


DEPLOYMENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.dirname(DEPLOYMENT_DIR)
TEST_MODEL_SUMMARY_PATH = "/home/starkaf/Documents/uni/master/fp/TensorDSE/resources/model_summaries/example_summaries/pretrainedResnet_quant.json"

def ParseArgs():
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-mode", "--Mode", help = "Mode of Operation (benchmarking or deployment)", required=True)
    parser.add_argument("-model", "--Model", help = "Path to Source Model file to optimize", required=True)
    parser.add_argument("-map", "--Mapping", help = "Path to CSV file containing mapping", required=True)
    # Read arguments from command line
    try:
        args = parser.parse_args()
        return args
    except:
        print('Wrong or Missing argument!')
        print('Example Usage: main.py -mode <mode of operation> -model <path/to/model/file> -map <path/to/csv/file/containing/maping>')
        sys.exit(1)

def GenerateRandomData():
    np.random.seed(123)
    random_data = np.random.randint(0, 256, size=(1, 2, 2, 3), dtype=np.uint8)

def GetLayerDetails(summary_path:str, layer_index:int):
    summary = ReadJSON(summary_path)
    layer = summary["layers"][layer_index]
    
    input_tensor = layer["inputs"][0]
    input_tensor_shape = input_tensor["shape"]
    input_tensor_dtype = input_tensor["type"]

    output_tensor = layer["outputs"][0]
    output_tensor_shape = output_tensor["shape"]
    output_tensor_dtype = output_tensor["type"]

    hardware_target = layer["mapping"]

    return input_tensor_shape, input_tensor_dtype, output_tensor_shape, output_tensor_dtype, hardware_target

def main():

    #args = ParseArgs()
    #splitter = split.Splitter(args.Mode, args.Model, args.Mapping)

    #Test variables
    tflite_path_test_string = os.path.join(WORK_DIR, "resources/example_models/mobilenet_v1_1_0_224_quant.tflite")
    hardware_target_test_string = "CPU"

    GenerateRandomData()
    input_tensor_shape, input_tensor_dtype, output_tensor_shape, output_tensor_dtype, hardware_target = GetLayerDetails(TEST_MODEL_SUMMARY_PATH, 0)
    print(input_tensor_shape)
    print(input_tensor_dtype)
    print(output_tensor_shape)

    input_data_vector = np.zeros(150528).astype(np.uint8)
    output_data_vector = np.zeros(1001).astype(np.uint8)

    inference_time = cpp_backend.distributed_inference(tflite_path_test_string, input_data_vector,
                                                                                output_data_vector, 
                                                                                len(input_data_vector), 
                                                                                len(output_data_vector), 
                                                                                hardware_target_test_string, 
                                                                                1)
    
    #try:
    #    log.info("Running Model Splitter ...")
    #    #splitter.Run()
    #    log.info("Splitting Process Complete!\n")
    #    log.info("Generating Source File ...")
    #    #GenerateSource()
    #    log.info("Source File Generation Complete!\n")
    #except Exception as e:
    #    splitter.Clean(True)
    #    log.error("Failed to run splitter! {}".format(str(e)))
    #finally:
    #    del splitter

if __name__ == '__main__':
    main()
    pass