#import tensorflow as tf
import os
import logging
from pathlib import Path
import subprocess
import json




def split_edgetpu_model(log: logging.Logger, name: str):

    fn = "%s.tflite" % name
    log.info("Patching the model in JSON")
    fn_json = str(Path(fn).with_suffix(".json"))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fn_json = os.path.join(dir_path, fn_json)

    with open(fn_json) as fin:
        model = json.load(fin)

    # Erase all opcodes except DEPTHWISE_CONV_2D.
    conv_opcode = -1
    new_opcodes = []
    for i, c in enumerate(model["operator_codes"]):
        if c["deprecated_builtin_code"] == 4:
            new_opcodes.append(c)
            conv_opcode = i
    assert conv_opcode >= 0
    model["operator_codes"] = new_opcodes
    print(new_opcodes)
    print('\n')

    # Fix the tensor dtypes which are int8 instead of uint8.
    # Also remove the multi-channel quantization which is not supported on Edge TPU.
    graph = model["subgraphs"][0]
    new_tensors = []
    index_map = {}
    for i, t in enumerate(graph["tensors"]):
        if t["type"] == "FLOAT32":
            continue
        if t["type"] == "INT8":
            t["type"] = "UINT8"
            t["quantization"]["zero_point"][0] = 0
        #t["quantization"]["scale"] = [t["quantization"]["scale"][0]]
        #t["quantization"]["zero_point"] = [t["quantization"]["zero_point"][0]]
        #t["quantization"]["quantized_dimension"] = 0
        index_map[i] = len(new_tensors)
        new_tensors.append(t)
        print(t)
        print('\n')
    graph["tensors"] = new_tensors
    

    """# Update the tensor indexes in the ops.
    new_ops = []
    for op in graph["operators"]:
        if op["opcode_index"] != conv_opcode:
            continue
        op["outputs"] = [index_map[i] for i in op["outputs"]]
        op["inputs"] = [index_map[i] for i in op["inputs"]]
        new_ops.append(op)
    graph["operators"] = new_ops


    
    
    # Update the global input and output tensor indexes.
    graph["inputs"][0] = new_ops[0]["inputs"][0]
    graph["outputs"][0] = new_ops[0]["outputs"][0]
    model["subgraphs"][0] = graph"""

    #with open(fn_json, "w") as fout:
        #json.dump(model, fout, indent=4)
    
    #log.info("Generating the binary flatbuffers model from JSON")
    #echo_run("flatc", "-b", "schema.fbs", fn_json)
    

def echo_run(*cmd):
    """Execute an arbitrary command and echo its output."""
    p = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.decode()
    if output:
        print(output)
    p.check_returncode()

name = 'test_model_quant'
log = logging.getLogger(name)
log.setLevel(logging.INFO)

fn = "%s.tflite" % name
split_edgetpu_model(log, name)   