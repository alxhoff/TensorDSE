#import tensorflow as tf
import os
import logging
from pathlib import Path
import subprocess
import json
import urllib.request
from enum import Enum

class EdgeTPUBuiltinOperator(Enum):
  ADD = 0
  AVERAGE_POOL_2D = 1
  CONCATENATION = 2
  CONV_2D = 3
  DEPTHWISE_CONV_2D = 4
  FULLY_CONNECTED = 9
  L2_NORMALIZATION = 11
  LOGISTIC = 14
  MAX_POOL_2D = 17
  MUL = 18
  RELU = 19
  RELU_N1_TO_1 = 20
  RELU6 = 21
  RESHAPE = 22
  RESIZE_BILINEAR = 23
  SOFTMAX = 25
  SPACE_TO_DEPTH = 26
  TANH = 28
  CUSTOM = 32
  PAD = 34
  MEAN = 40
  SUB = 41
  SQUEEZE = 43
  STRIDED_SLICE = 45
  MAXIMUM = 55
  MINIMUM = 57
  SLICE = 65
  TRANSPOSE_CONV = 67
  EXPAND_DIMS = 70
  SUM = 74
  PACK = 83
  RESIZE_NEAREST_NEIGHBOR = 97
  QUANTIZE = 114

edgetpu_opcodes = [BuiltinOperator.value for BuiltinOperator in EdgeTPUBuiltinOperator]

mapping = [[0,-1],[1,0],[2,2],[3,2]]

#Converts a tflite model to JSON format and loads it
def load_tflite_as_json(log: logging.Logger, name: str):

    fn = "%s.tflite" % name

    if not Path("schema.fbs").exists():
        log.info("schema.fbs was not found, downloading")
        urllib.request.urlretrieve(
            "https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/schema/schema.fbs",
            "schema.fbs")
        log.info("Downloaded schema.fbs")
 
    
    """log.info("Converting the model from binary flatbuffers to JSON")
    echo_run("flatc", "-t", "--strict-json", "--defaults-json", "schema.fbs", "--", fn)"""

    #log.info("Patching the model in JSON")
    fn_json = str(Path(fn).with_suffix(".json"))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fn_json = os.path.join(dir_path, fn_json)

    with open(fn_json) as fin:
        model = json.load(fin)
    
    return model

#Classifies the operations in the model and returns an array of supported and unsupported ops
def classify_ops(model: dict):
    
    supported_opcodes = []
    unsupported_opcodes = []

    for i, c in enumerate(model["operator_codes"]):
        # Variable that saves the index of an operation in the operator_codes list
        # along with its deprecated_builtin_code
        op_tuple = []
        if c["deprecated_builtin_code"] in edgetpu_opcodes:
            op_tuple = [i,c["deprecated_builtin_code"]]
            supported_opcodes.append(op_tuple)
        else :
            op_tuple = [i,c["deprecated_builtin_code"]]
            unsupported_opcodes.append(op_tuple)
    
    result = [supported_opcodes,unsupported_opcodes]
    return result

def info_mapping(mapping: list):

    sequential_ops = []
    sequence = []
    sequence_flag = False

    for ops in mapping:
        if ops[1] == 2:
            if sequence_flag == False:
                sequence.append(ops[0])
                sequence_flag = True
                if ops[0] == len(mapping)-1:
                    sequential_ops.append(sequence)

            elif sequence_flag == True:
                sequence.append(ops[0])
                if ops[0] == len(mapping)-1:
                    sequential_ops.append(sequence)
        elif ops[1] != 2:
            if sequence_flag == True:
                sequential_ops.append(sequence)
                sequence = []
                sequence_flag = False
            else:
                continue
    
    return sequential_ops

def seperate_ops(model: dict, info: list):

    graph = model["subgraphs"][0]

    new_ops = []
    for i,op in enumerate(graph["operators"]):
        for info_op in info[0]:
            if info_op == i:
                new_ops.append(op)

    new_opcodes = []
    for i, op_code in enumerate(model["operator_codes"]):
        for new_op in new_ops:
            if i == new_op["opcode_index"]:
                new_opcodes.append(op_code)
                new_op["opcode_index"] = len(new_opcodes) - 1

    new_tensors = []
    tensor_indexes = []
    for new_op in new_ops:
        for i,op_input in enumerate(new_op["inputs"]):
            if op_input in tensor_indexes:
                continue
            else:
                tensor_indexes.append(op_input)
                new_tensors.append(graph["tensors"][op_input])
                new_op["inputs"][i] = len(new_tensors) - 1
        for j, op_output in enumerate(new_op["outputs"]):
            if op_output in tensor_indexes:
                continue
            else:
                tensor_indexes.append(op_output)
                new_tensors.append(graph["tensors"][op_output])
                new_op["outputs"][j] = len(new_tensors) - 1
    
    new_inputs = []
    new_outputs = []
    new_inputs.append(new_ops[0]["inputs"][0])
    new_outputs.append(new_ops[len(new_ops) - 1]["outputs"][0])

    new_buffers = []
    buffer_indexes = []
    for new_tensor in new_tensors:
        index = new_tensor["buffer"]
        if index in buffer_indexes:
            continue
        else:
            buffer_indexes.append(index)
            new_buffers.append(model["buffers"][index])
            new_tensor["buffer"] = len(new_buffers) - 1
    
def merge_ops(log: logging.Logger, model: dict, info: list, name: str):
    
    #echo_run("flatc", "-b", "schema.fbs", fn_json)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)

    tmp_model = load_tflite_as_json(log, name)
    seperate_ops(tmp_model,info)

    info.pop(0)
    return info,model

def optimize_edgetpu_model(log: logging.Logger, name: str):
    
    model = load_tflite_as_json(log, name)

    info = info_mapping(mapping)    
    
    if len(info) != 0:
        merge_flag = True
    while merge_flag:
        info,model = merge_ops(log, model, info, name)
        if len(info) == 0:
            merge_flag = False
            break

    #log.info("Generating the binary flatbuffers model from JSON")
    #echo_run("flatc", "-b", "schema.fbs", fn_json) """

def echo_run(*cmd):
    #Execute an arbitrary command and echo its output.
    p = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.decode()
    if output:
        print(output)
    p.check_returncode()

name = 'test_model_quant_edgetpu'
log = logging.getLogger(name)
logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

fn = "%s.tflite" % name
optimize_edgetpu_model(log, name)