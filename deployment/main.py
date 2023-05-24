import os, sys
import argparse
import numpy as np

import ModelLab.split as split
from ModelLab.logger import log
from ModelLab.utils import LayerDetails
#from CPPSourceGen.generate import GenerateSource

sys.path.append('py_backend')


DEPLOYMENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.dirname(DEPLOYMENT_DIR)

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


def main():
    from backend.distributed_inference import distributed_inference

    #args = ParseArgs()
    #splitter = split.Splitter(args.Mode, args.Model, args.Mapping)

    #Test variables
    MODEL_PATH = os.path.join(WORK_DIR, "deployment/resources/models/keyword_spotting/kws_ref_model.tflite")
    MODEL_SUMMARY_PATH = os.path.join(WORK_DIR, "resources/model_summaries/kws_summary.json")

    details = LayerDetails(MODEL_SUMMARY_PATH)

    for layer in details.layers:
        details.ReadLayerDetails(layer)
        

    #input_data_vector = np.zeros(layer.GetTensorSize("Input")).astype(np.uint8)
    #output_data_vector = np.zeros(layer.GetTensorSize("Output")).astype(np.uint8)
#
    #inference_time = cpp_backend.distributed_inference(tflite_path_test_string, input_data_vector,
    #                                                                            output_data_vector, 
    #                                                                            len(input_data_vector), 
    #                                                                            len(output_data_vector), 
    #                                                                            hardware_target_test_string, 
    #                                                                            1)
    
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
