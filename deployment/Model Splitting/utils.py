import logging
import os
import json
import csv

def read_op_mapping(csv_file_path: str):
    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        data = [list(map(int,rec)) for rec in reader]
        return data

def load_json_model(file_path: str):
    with open(file_path) as fin:
        model = json.load(fin)
        return model

def load_source_model(log: logging.Logger, file_dir_path: str, main_dir_path: str):

    tflite_file_exists = False
    json_file_exists = False

    schema_path = os.path.join(main_dir_path,"schema","schema.fbs")

    while json_file_exists == False:

        tflite_file_dir_path = os.path.join(file_dir_path,"tflite")
        for file in os.listdir(tflite_file_dir_path):
            if file.endswith(".tflite"):
                #file_path_tflite = os.path.join(tflite_file_dir_path, file)
                tflite_file_name = file
                tflite_file_exists = True
        
        json_file_dir_path = os.path.join(file_dir_path,"json")
        for file in os.listdir(json_file_dir_path):
            if file.endswith(".json"):
                file_path_json = os.path.join(json_file_dir_path, file)
                #json_file_name = file
                json_file_exists = True
    
        if tflite_file_exists:
            if json_file_exists:
                model = load_json_model(file_path_json)
                log.info("Loading of source model in JSON successful")
                return model
            else:
                log.info("Converting the model from binary flatbuffers to JSON")
                convert_to_json(schema_path,file_dir_path,tflite_file_name)
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

def optimize_mapping(mapping: list):

    sequential_ops = []
    sequence = []
    opt_mapping = []
    sequence_flag = False
    
    for ops in mapping:
        if ops[1] == 2:
            if sequence_flag == False:
                sequence.append(ops[0])
                sequence_flag = True
                if ops[0] == len(mapping)-1:
                    sequential_ops.append(sequence)
                    opt_mapping.append(sequence)
            elif sequence_flag == True:
                sequence.append(ops[0])
                if ops[0] == len(mapping)-1:
                    sequential_ops.append(sequence)
                    opt_mapping.append(sequence)
        elif ops[1] != 2:
            if sequence_flag == True:
                sequential_ops.append(sequence)
                opt_mapping.append(sequence)
                single_op = []
                single_op.append(ops[0])
                opt_mapping.append(single_op)
                sequence = []
                sequence_flag = False
            elif sequence_flag == False:
                single_op = []
                single_op.append(ops[0])
                opt_mapping.append(single_op)
                continue

    
    return sequential_ops,opt_mapping

def create_submodel(source_model: dict, submodel: dict, info: list, main_dir_path: str, submodel_filename: str):

    source_graph = source_model["subgraphs"][0]

    new_ops = []
    for info_op in info[0]:
        new_ops.append(source_graph["operators"][info_op])
            
    new_opcodes = []
    for new_op in new_ops:
        if source_model["operator_codes"][new_op["opcode_index"]] not in new_opcodes:
            new_opcodes.append(source_model["operator_codes"][new_op["opcode_index"]])
            new_op["opcode_index"] = len(new_opcodes) - 1
        else:
            new_op["opcode_index"] = new_opcodes.index(source_model["operator_codes"][new_op["opcode_index"]])

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
    for i,new_tensor in enumerate(new_tensors):
        buffer_index = new_tensor["buffer"]
        new_buffers.append(source_model["buffers"][buffer_index])
        new_tensor["buffer"] = len(new_buffers) - 1
    
    submodel["operator_codes"] = new_opcodes
    submodel["subgraphs"][0]["tensors"]     =   new_tensors
    submodel["subgraphs"][0]["inputs"]      =   new_inputs
    submodel["subgraphs"][0]["outputs"]     =   new_outputs
    submodel["subgraphs"][0]["operators"]   =   new_ops
    submodel["buffers"]                     =   new_buffers
    submodel["metadata"][0]["buffer"]       =   len(new_buffers) - 1

    submodel_filepath = os.path.join(main_dir_path,"models","submodels","json",submodel_filename)
    
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

def move_file(file_path: str, target_path: str):
    echo_run("mv",file_path,target_path)

def delete_file(file_path: str):
    echo_run("rm", file_path)

def convert_to_json(schema_path: str, file_dir: str, file_name: str):
    file_path = os.path.join(file_dir,"tflite",file_name)
    echo_run("flatc", "-t", "--strict-json", "--defaults-json", schema_path, "--", file_path)
    file_name = file_name.split(".",1)[0] + ".json"
    target_path = os.path.join(file_dir,"json",file_name)
    move_file(file_name,target_path)
    return file_name,target_path

def convert_to_tflite(schema_path: str, file_dir: str, file_name: str):
    file_path = os.path.join(file_dir,"json",file_name)
    echo_run("flatc", "-b", schema_path, file_path)
    file_name = file_name.split(".",1)[0] + ".tflite"
    target_path = os.path.join(file_dir,"tflite",file_name)
    move_file(file_name,target_path)
    return file_name,target_path

def initialize_submodel(log: logging.Logger, main_dir_path: str, info: list):

    shell_model_path = os.path.join(main_dir_path, "models", 
                                                   "shell_models",
                                                   "source_model_shell.json")
    submodels_dir_path = os.path.join(main_dir_path, "models",
                                                     "submodels")
            
    log.info("Creating submodel file for operations: %s",info[0])
    submodel_number = '_'.join([str(elem) for elem in info[0]])
    submodel_filename = "submodel_" + submodel_number + ".json"
    submodel_path = os.path.join(submodels_dir_path,"json",submodel_filename)
    copy_file(shell_model_path,submodel_path)
    submodel = load_json_model(submodel_path)
    log.info("Submodel file: %s created",submodel_filename)
    return submodel,submodel_filename

def initialize_optimized_model(log: logging.Logger, main_dir_path: str,):

    shell_model_path = os.path.join(main_dir_path, "models", 
                                                   "shell_models",
                                                   "optimized_model_shell.json")

    optimized_model_dir = os.path.join(main_dir_path, "models","optimized_model","json")

    source_model_dir = os.path.join(main_dir_path,"models","source_model","tflite")

    for file in os.listdir(source_model_dir):
        if file.endswith(".tflite"):
            source_model_name = file.split(".",1)[0]
            optimized_model_filename = source_model_name + "_opt" + "_edgetpu" + ".json"
    
    log.info("Creating optimized model...")
    optimized_model_path = os.path.join(optimized_model_dir,optimized_model_filename)
    copy_file(shell_model_path,optimized_model_path)
    optimized_model = load_json_model(optimized_model_path)
    log.info("Optimized model file: %s created",optimized_model_filename)
    return optimized_model,optimized_model_filename

def docker_start():
    echo_run("sudo","docker","start","debian-docker")

def docker_copy(file_path: str, direction: str, target_location: str):

    container = "debian-docker" + ":"

    if direction == "to_container" :
        target_location = container + target_location
    elif direction == "from_container" :
        file_path = container + file_path

    echo_run("sudo","docker","cp",file_path,target_location)

def docker_compile(file_path: str):
    compiling_command = "edgetpu_compiler -s %s" % file_path
    echo_run('sudo','docker','exec','-ti','debian-docker','sh','-c',"%s" % compiling_command)

def docker_clean():
    command = "find -type f -name '*submodel*' -delete"
    echo_run('sudo','docker','exec','-ti','debian-docker','sh','-c',"%s" % command)

def add_op(source_model: dict, compiled_submodel: dict, optimized_model: dict, op: list, edge_flag: bool):

    source_graph = source_model["subgraphs"][0]
    if len(compiled_submodel) != 0 :
        submodel_graph = compiled_submodel["subgraphs"][0]
    opt_graph = optimized_model["subgraphs"][0]

    opt_ops = opt_graph["operators"]
    opt_opcodes = optimized_model["operator_codes"]
    opt_tensors = optimized_model["subgraphs"][0]["tensors"]
    opt_tensor_names = [t["name"] for t in opt_tensors]
    opt_buffers = optimized_model["buffers"]

    if edge_flag == True:

        new_ops = []
        new_ops.append(submodel_graph["operators"][0])
        opt_ops.append(submodel_graph["operators"][0])

        new_opcodes = []
        new_opcodes.append(compiled_submodel["operator_codes"][0])
        if compiled_submodel["operator_codes"][0] not in opt_opcodes:
            opt_opcodes.append(compiled_submodel["operator_codes"][0])
        
        opt_ops[len(opt_ops) - 1]["opcode_index"] = opt_opcodes.index(compiled_submodel["operator_codes"][0])

        new_tensors = []
        
        for new_op in new_ops:
            op_index = opt_ops.index(new_op)
            for i,op_input in enumerate(new_op["inputs"]):
                if submodel_graph["tensors"][op_input]["name"] in opt_tensor_names:
                #if submodel_graph["tensors"][op_input] in opt_tensors:
                    t_index = [k for k,t in enumerate(opt_tensors) if opt_tensors[k]["name"] == submodel_graph["tensors"][op_input]["name"]][0]
                    #opt_tensors[t_index] = submodel_graph["tensors"][op_input]
                    opt_ops[op_index]["inputs"][i] = t_index
                    new_op["inputs"][i] = t_index
                    #opt_ops[op_index]["inputs"][i] = opt_tensors.index(submodel_graph["tensors"][op_input])
                    #new_op["inputs"][i] = opt_tensors.index(submodel_graph["tensors"][op_input])
                    continue
                else:
                    opt_tensors.append(submodel_graph["tensors"][op_input])
                    new_tensors.append(submodel_graph["tensors"][op_input])
                    opt_ops[op_index]["inputs"][i] = len(opt_tensors) - 1
                    new_op["inputs"][i] = len(opt_tensors) - 1
            for j, op_output in enumerate(new_op["outputs"]):
                if submodel_graph["tensors"][op_output]["name"] in opt_tensor_names:
                #if submodel_graph["tensors"][op_output] in opt_tensors:
                    t_index = [k for k,t in enumerate(opt_tensors) if opt_tensors[k]["name"] == source_graph["tensors"][op_input]["name"]][0]
                    opt_ops[op_index]["inputs"][i] = t_index
                    new_op["inputs"][i] = t_index
                    #opt_ops[op_index]["outputs"][j] = opt_tensors.index(submodel_graph["tensors"][op_output])
                    #new_op["outputs"][j] = opt_tensors.index(submodel_graph["tensors"][op_output])
                    continue
                else:
                    opt_tensors.append(submodel_graph["tensors"][op_output])
                    new_tensors.append(submodel_graph["tensors"][op_output])
                    opt_ops[op_index]["outputs"][j] = len(opt_tensors) - 1
                    new_op["outputs"][j] = len(opt_tensors) - 1
                
        new_inputs = []
        new_outputs = []
        new_inputs.append(opt_ops[0]["inputs"][0])
        new_outputs.append(opt_ops[len(opt_ops) - 1]["outputs"][0])
        optimized_model["subgraphs"][0]["inputs"]      =   new_inputs
        optimized_model["subgraphs"][0]["outputs"]     =   new_outputs
        
        for new_tensor in new_tensors:
            buffer_index = new_tensor["buffer"]
            opt_buffers.append(compiled_submodel["buffers"][buffer_index])
            tensor_index = opt_tensors.index(new_tensor)
            opt_tensors[tensor_index]["buffer"] = len(opt_buffers) - 1
        
    elif edge_flag == False:
        
        new_ops = []
        new_ops.append(source_graph["operators"][op[0]])
        opt_ops.append(source_graph["operators"][op[0]])

        new_opcodes = []
        source_opcode_index = new_ops[0]["opcode_index"]
        new_opcodes.append(source_model["operator_codes"][source_opcode_index])
        if new_opcodes[0] not in opt_opcodes:
            opt_opcodes.append(source_model["operator_codes"][source_opcode_index])
        
        opt_ops[len(opt_ops) - 1]["opcode_index"] = opt_opcodes.index(source_model["operator_codes"][source_opcode_index])
        
        new_tensors = []
        
        for new_op in new_ops:
            op_index = opt_ops.index(new_op)
            for i,op_input in enumerate(new_op["inputs"]):
                if source_graph["tensors"][op_input]["name"] in opt_tensor_names:
                #if source_graph["tensors"][op_input] in opt_tensors:
                    t_index = [k for k,t in enumerate(opt_tensors) if opt_tensors[k]["name"] == source_graph["tensors"][op_input]["name"]][0]
                    opt_ops[op_index]["inputs"][i] = t_index
                    new_op["inputs"][i] = t_index
                    #opt_ops[op_index]["inputs"][i] = opt_tensors.index(source_graph["tensors"][op_input])
                    #new_op["inputs"][i] = opt_tensors.index(source_graph["tensors"][op_input])
                    continue
                else:
                    opt_tensors.append(source_graph["tensors"][op_input])
                    #opt_tensors[len(opt_tensors) -1]["quantization"]["scale"][0] = 0
                    #opt_tensors[len(opt_tensors) -1]["quantization"]["zero_point"][0] = 0
                    new_tensors.append(source_graph["tensors"][op_input])
                    opt_ops[op_index]["inputs"][i] = len(opt_tensors) - 1
                    new_op["inputs"][i] = len(opt_tensors) - 1
            for j, op_output in enumerate(new_op["outputs"]):
                if source_graph["tensors"][op_output]["name"] in opt_tensor_names:
                #if source_graph["tensors"][op_output] in opt_tensors:
                    t_index = [k for k,t in enumerate(opt_tensors) if opt_tensors[k]["name"] == source_graph["tensors"][op_output]["name"]][0]
                    opt_ops[op_index]["outputs"][i] = t_index
                    new_op["outputs"][i] = t_index
                    #opt_ops[op_index]["outputs"][j] = opt_tensors.index(source_graph["tensors"][op_output])
                    #new_op["outputs"][j] = opt_tensors.index(source_graph["tensors"][op_output])
                    continue
                else:
                    opt_tensors.append(source_graph["tensors"][op_output])
                    #opt_tensors[len(opt_tensors) -1]["quantization"]["scale"][0] = 0
                    #opt_tensors[len(opt_tensors) -1]["quantization"]["zero_point"][0] = 0
                    new_tensors.append(source_graph["tensors"][op_output])
                    opt_ops[op_index]["outputs"][j] = len(opt_tensors) - 1
                    new_op["outputs"][j] = len(opt_tensors) - 1
                
        new_inputs = []
        new_outputs = []
        new_inputs.append(opt_ops[0]["inputs"][0])
        new_outputs.append(opt_ops[len(opt_ops) - 1]["outputs"][0])
        optimized_model["subgraphs"][0]["inputs"]      =   new_inputs
        optimized_model["subgraphs"][0]["outputs"]     =   new_outputs
        
        for new_tensor in new_tensors:
            buffer_index = new_tensor["buffer"]
            opt_buffers.append(source_model["buffers"][buffer_index])
            tensor_index = opt_tensors.index(new_tensor)
            opt_tensors[tensor_index]["buffer"] = len(opt_buffers) - 1
    
    options = ["FullyConnectedOptions","Conv2DOptions","DepthwiseConv2DOptions"]
    if opt_ops[len(opt_ops) - 1]["builtin_options_type"] in options:
        tensor_index_input_0 = opt_ops[len(opt_ops) - 1]["inputs"][0]
        tensor_index_input_1 = opt_ops[len(opt_ops) - 1]["inputs"][1]
        tensor_index_output = opt_ops[len(opt_ops) - 1]["inputs"][2]

        opt_tensors[tensor_index_output]["quantization"]["scale"][0] = opt_tensors[tensor_index_input_0]["quantization"]["scale"][0]*opt_tensors[tensor_index_input_1]["quantization"]["scale"][0]

    return optimized_model
        
def update_optimized_model(source_model: dict, compiled_submodel: dict, optimized_model: dict, mapping: list, submodel_created: bool, op_count: int):

    info,opt_mapping = optimize_mapping(mapping)
    
    for i,op in enumerate(opt_mapping):
        if i == op_count:
            if len(op) == 1:
                if op in info:
                    if submodel_created == True:
                        optimized_model = add_op(source_model,compiled_submodel,optimized_model,op,True)
                        op_count += 1
                        submodel_created = False
                    elif submodel_created == False:
                        break
                else:
                    optimized_model = add_op(source_model,compiled_submodel,optimized_model,op,False)
                    op_count += 1
            elif len(op) != 1:
                if submodel_created == True:
                    optimized_model = add_op(source_model,compiled_submodel,optimized_model,op,True)
                    op_count += 1
                    submodel_created = False
                elif submodel_created == False:
                    break
        else:
            continue
    return optimized_model,submodel_created,op_count
