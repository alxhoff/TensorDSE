import logging
import os
import json

def load_json_model(log: logging.Logger, file_dir_path: str, main_dir_path: str):

    tflite_file_exists = False
    json_file_exists = False

    schema_path = os.path.join(main_dir_path,"schema","schema.fbs")

    while json_file_exists == False:

        for file in os.listdir(file_dir_path):
            if file.endswith(".tflite"):
                file_path_tflite = os.path.join(file_dir_path, file)
                tflite_file_exists = True
            elif file.endswith(".json"):
                file_path_json = os.path.join(file_dir_path, file)
                json_file_exists = True
    
        if tflite_file_exists:
            if json_file_exists:
                with open(file_path_json) as fin:
                    model = json.load(fin)
                    log.info("Loading of model in JSON successful")
                    return model
            else:
                log.info("Converting the model from binary flatbuffers to JSON")
                convert_to_json(schema_path,file_path_tflite)
        else:
            if json_file_exists:
                with open(file_path_json) as fin:
                    model = json.load(fin)
                    log.info("Loading of shell submodel in JSON successful")
                    return model
            else:
                log.info("TFLite model not found.")
                break

def echo_run(*cmd):

    import subprocess
    
    #Execute an arbitrary command and echo its output.
    p = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.decode()
    if output:
        print(output)
    p.check_returncode()

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

def create_submodel(source_model: dict, submodel: dict, info: list, main_dir_path: str, submodel_filename: str):

    source_graph = source_model["subgraphs"][0]

    new_ops = []
    for i,op in enumerate(source_graph["operators"]):
        for info_op in info[0]:
            if info_op == i:
                new_ops.append(op)

    new_opcodes = []
    for i, op_code in enumerate(source_model["operator_codes"]):
        for new_op in new_ops:
            if i == new_op["opcode_index"]:
                new_opcodes.append(op_code)
                new_op["opcode_index"] = len(new_opcodes) - 1

    new_tensors = []
    tensor_indexes = []

    for new_op in new_ops:
        for i,op_input in enumerate(new_op["inputs"]):
            if op_input in tensor_indexes:
                new_op["inputs"][i] = new_tensors.index(source_graph["tensors"][op_input])
                continue
            else:
                tensor_indexes.append(op_input)
                new_tensors.append(source_graph["tensors"][op_input])
                new_op["inputs"][i] = len(new_tensors) - 1
        for j, op_output in enumerate(new_op["outputs"]):
            if op_output in tensor_indexes:
                new_op["outputs"][j] = new_tensors.index(source_graph["tensors"][op_output])
                continue
            else:
                tensor_indexes.append(op_output)
                new_tensors.append(source_graph["tensors"][op_output])
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
            new_buffers.append(source_model["buffers"][index])
            new_tensor["buffer"] = len(new_buffers) - 1
    
    submodel["operator_codes"] = new_opcodes
    submodel["subgraphs"][0]["tensors"]     =   new_tensors
    submodel["subgraphs"][0]["inputs"]      =   new_inputs
    submodel["subgraphs"][0]["outputs"]     =   new_outputs
    submodel["subgraphs"][0]["operators"]   =   new_ops
    submodel["buffers"]                     =   new_buffers
    submodel["metadata"][0]["buffer"]       =   len(new_buffers) - 1

    submodel_filepath = os.path.join(main_dir_path,"models","submodels",submodel_filename)
    
    with open(submodel_filepath,"w") as fout:
        json.dump(submodel, fout, indent=2)
    
    info.pop(0)

def check_for_source_model(models_dir: str):

    source_model_dir = os.path.join(models_dir,"source_model")
    for file in os.listdir(source_model_dir):
        if file.endswith(".tflite"):
            source_model_filename = file
            source_model_filepath = os.path.join(source_model_dir, file)
    
    return source_model_filename,source_model_filepath

def copy_file(file_path: str, target_path: str):
    echo_run("cp",file_path,target_path)

def delete_file(file_path: str):
    echo_run("rm", "-v", file_path)

def convert_to_json(schema_path: str, file_path: str):
    echo_run("flatc", "-t", "--strict-json", "--defaults-json", schema_path, "--", file_path)

def move_file()

def convert_to_tflite(schema_path: str, file_path: str):
    echo_run("flatc", "-b", schema_path, file_path)

def initialize_submodel(log: logging.Logger, main_dir_path: str, info: list):

    shell_model_path = os.path.join(main_dir_path, "models", 
                                                   "shell_models",
                                                   "source_model_shell.json")
    submodels_dir_path = os.path.join(main_dir_path, "models",
                                                     "submodels")
            
    log.info("Creating submodel file for operations: %s",info[0])
    submodel_number = '_'.join([str(elem) for elem in info[0]])
    submodel_filename = "submodel_" + submodel_number + ".json"
    submodel_path = os.path.join(submodels_dir_path,submodel_filename)
    copy_file(shell_model_path,submodel_path)
    submodel = load_json_model(log, submodels_dir_path,main_dir_path)
    log.info("Submodel file: %s created",submodel_filename)
    return submodel,submodel_filename

