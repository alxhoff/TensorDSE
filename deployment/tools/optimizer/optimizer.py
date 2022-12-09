import os
import json
import urllib
from utils import ParseArgs, LoggerInit, RunTerminalCommand, ReadCSV, ReadJSON, CopyFile, MoveFile, CompareKeys

log = LoggerInit()
WORK_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(WORK_DIR, "models")
MAPPING_DIR   = os.path.join(WORK_DIR, "mapping")
RESOURCES_DIR = os.path.join(WORK_DIR, "resources")
SOURCE_DIR    = os.path.join(MODELS_DIR, "source")
SUB_DIR       = os.path.join(MODELS_DIR, "sub")
FINAL_DIR     = os.path.join(MODELS_DIR, "final")

HW_ID = ["CPU", "GPU", "TPU"]

""" 
OperatorCode = {
    "deprecated_builtin_code": bytes(0),
    "custom_code": "",
    "version":1,
    "builtin_code": 0
}

Tensor = {
  "shape":[0],
  "type": 0,
  "buffer": 0,
  "name": "",
  "quantization": "QuantizationParameters",
  "is_variable": False,
  "sparsity": "SparsityParameters",
  "shape_signature":[0],
  "has_rank": False,
  "variant_tensors":["VariantSubType"]
}

Operator = {
    "opcode_index": 0,
    "inputs": [0],
    "outputs": [0],
    "builtin_options": "BuiltinOptions",
    "custom_options": [],
    "custom_options_format": "CustomOptionsFormat",
    "mutating_variable_inputs": [],
    "intermediates": [0]
}

Subgraph = {
    "tensors":[Tensor],
    "inputs":[0],
    "outputs":[0],
    "operators":[Operator],
    "name": ""
}

Buffer = {
    "data": []
}

Metadata = {
    "name": "",
    "buffer": 0
}

MODEL = {
    "version": 0,
    "operator_codes": [OperatorCode],
    "subgraphs": [Subgraph],
    "description": "",
    "buffers": [Buffer],
    "metadata": [Metadata],
    "signature_defs": []
} """

class Docker:
    def __init__(self, name):
        self.name = name
        self.Start()
        self.CreateSubDir()

    def CreateSubDir(self):
        compiling_command = "mkdir /home/sub"
        RunTerminalCommand('sudo', 'docker', 'exec', '-ti', 'debian-docker', 'sh', '-c', compiling_command)

    def Start(self):
        try:
            RunTerminalCommand("sudo", "docker", "start", self.name)
        except Exception as e:
            log.error(" Error: Failed to start Docker Container")
            log.error(e)

    def Copy(self, filename: str, src: str, trg: str):
        if [src, trg] == ["host", "docker"]:
            src_path = os.path.join(SUB_DIR, "tflite", filename)
            trg_path = "{0}:{1}".format(self.name, os.path.join("/home/sub", filename)) 
        else:
            src_path = "{0}:{1}".format(self.name, os.path.join("/home/sub", filename))
            trg_path = os.path.join(SUB_DIR, "tflite", filename)
             
        RunTerminalCommand("sudo", "docker", "cp", src_path, trg_path)
    
    def Compile(self, filename: str):
        compiling_command = "edgetpu_compiler -o /home/sub/ -s /home/sub/{0}".format(filename)
        RunTerminalCommand('sudo', 'docker', 'exec', '-ti', 'debian-docker', 'sh', '-c', compiling_command)

    def Clean(self):
        command = "find -type f -name '*submodel_tpu*' -delete"
        RunTerminalCommand('sudo', 'docker', 'exec', '-ti', 'debian-docker', 'sh', '-c', command)

    def __del__(self):
        command = "rm -rf /home/sub"
        RunTerminalCommand('sudo', 'docker', 'exec', '-ti', 'debian-docker', 'sh', '-c', command)
        
class Model:
    def __init__(self, path_to_model: str, schema_path: str):
        self.paths = {"json": "",
                      "tflite": ""}
        for ext in self.paths.keys():
            if path_to_model.endswith(ext):
                self.paths[ext] = path_to_model
                log.info("    " + ext.upper() + " Model path saved: " + self.paths[ext])
        self.schema = schema_path
    
    def Convert(self, source_ext: str, target_ext: str):
        log.info("Converting Model from " + source_ext.upper() + " to " + target_ext.upper() + " ...")
        if ([source_ext, target_ext] == ["json", "tflite"]):
            RunTerminalCommand("flatc", "-b", self.schema, self.paths["json"])
            tmp_filename = self.paths["json"].split("/")[len(self.paths["json"].split("/"))-1].split(".")[0] + ".tflite"
            self.paths["tflite"] = self.paths["json"].replace(source_ext, target_ext)
            MoveFile(tmp_filename, self.paths["tflite"])
        elif ([source_ext, target_ext] == ["tflite", "json"]):
            RunTerminalCommand("flatc", "-t", "--strict-json", "--defaults-json", self.schema, "--", self.paths["tflite"])
            tmp_filename = self.paths["tflite"].split("/")[len(self.paths["tflite"].split("/"))-1].split(".")[0] + ".json"
            self.paths["json"] = self.paths["tflite"].replace(source_ext, target_ext)
            MoveFile(tmp_filename, self.paths["json"])
            self.json = ReadJSON(self.paths["json"])
        log.info("    Model successfully converted.")

class Submodel(Model):
    def __init__(self, source_model_json: dict):
        CopyFile(os.path.join(RESOURCES_DIR, "shell", "shell_model.json"),
                 os.path.join(SUB_DIR, "json", "shell_model.json"))
        super().__init__(path_to_model=os.path.join(SUB_DIR, "json", "shell_model.json"), 
                          schema_path=os.path.join(RESOURCES_DIR, "schema", "schema.fbs"))
        self.json = ReadJSON(self.paths["json"])
        self.source_model_json = source_model_json

    def AddOps(self, ops_block):
        #Read Main Graph
        source_graph = self.source_model_json["subgraphs"][0]

        #Add version
        new_version = self.source_model_json["version"]

        #Add Operators from Main Graph and add them according to index op 
        new_ops = []
        for op_index in ops_block:
            new_ops.append(source_graph["operators"][op_index])

        #Add the OperatorCodes of the Newly added Operators and update their opcode_index        
        new_opcodes = []
        for new_op in new_ops:
            if self.source_model_json["operator_codes"][new_op["opcode_index"]] not in new_opcodes:
                new_opcodes.append(self.source_model_json["operator_codes"][new_op["opcode_index"]])
                new_op["opcode_index"] = len(new_opcodes) - 1
            else:
                new_op["opcode_index"] = new_opcodes.index(self.source_model_json["operator_codes"][new_op["opcode_index"]])
        
        #Add Tensors according to added Operators
        new_tensors = []
        tensor_indexes = []
        for new_op in new_ops:
            for entry in ["inputs", "outputs"]:
                for i, op_entry in enumerate(new_op[entry]):
                    if op_entry in tensor_indexes:
                        new_op[entry][i] = new_tensors.index(source_graph["tensors"][op_entry])
                        continue
                    else:
                        tensor_indexes.append(op_entry)
                        new_tensors.append(source_graph["tensors"][op_entry])
                        new_op[entry][i] = len(new_tensors) - 1

        #Add Submodel Input and Output Tensors
        new_inputs = []
        new_outputs = []
        new_inputs.append(new_ops[0]["inputs"][0])
        new_outputs.append(new_ops[-1]["outputs"][0])

        #Add Subgraph Name
        if "name" in source_graph.keys():
            new_subgraph_name = source_graph["name"]
            self.json["subgraphs"][0]["name"]        =   new_subgraph_name
        else:
            self.json["subgraphs"][0].pop("name", None)

        #Add Description
        new_description = self.source_model_json["description"]

        #Add Buffers according to the newly added Tensors
        new_buffers = []
        for i,new_tensor in enumerate(new_tensors):
            buffer_index = new_tensor["buffer"]
            new_buffers.append(self.source_model_json["buffers"][buffer_index])
            new_tensor["buffer"] = len(new_buffers) - 1
        
        #Add metadata
        new_metadata = []
        if "metadata" in self.source_model_json.keys():
            for i, source_metadata in enumerate(self.source_model_json["metadata"]):
                new_buffers.append(self.source_model_json["buffers"][source_metadata["buffer"]])
                new_metadata.append(source_metadata)
                new_metadata[-1]["buffer"] = len(new_buffers) - 1
            self.json["metadata"]                    =   new_metadata
        else:
            self.json.pop("metadata", None)

        #Add SignatureDefs
        #for i, sig_def in enumerate(self.source_model_json["signature_defs"]):
        #    for entry in ["inputs", "outputs"]:
        #        for this_entry in sig_def[entry]:
        #            if this_entry["tensor_index"] in tensor_indexes:
        #                self.json["signature_defs"][i] = self.source_model_json["signature_defs"][i].copy()
        #                self.json["signature_defs"][i][entry]["tensor_index"]


        #Update all fields
        self.json["version"]                     =   new_version
        self.json["operator_codes"]              =   new_opcodes

        self.json["subgraphs"][0]["tensors"]     =   new_tensors
        self.json["subgraphs"][0]["inputs"]      =   new_inputs
        self.json["subgraphs"][0]["outputs"]     =   new_outputs
        self.json["subgraphs"][0]["operators"]   =   new_ops
        
        
        self.json["description"]                 =   new_description
        self.json["buffers"]                     =   new_buffers

        #Save Submodel Number
        self.submodel_number = '_'.join([str(elem) for elem in ops_block])
        
    def Save(self, block_key: str):
        submodel_filename = "submodel_{0}_{1}.json".format(block_key.lower(), self.submodel_number)
        submodel_filepath = os.path.join(SUB_DIR, "json", submodel_filename)
        MoveFile(self.paths["json"], submodel_filepath)
        self.paths["json"] = submodel_filepath
        with open(submodel_filepath, "w") as fout:
            json.dump(self.json, fout, indent=2)

class Optimizer:
    def __init__(self, source_model_path, mapping_path):
        log.info("Initializing Environment ...")
        self.InitializeEnv(source_model_path, mapping_path)
        log.info("Initializing Source Model ...")
        self.source_model = Model(self.source_model_path, self.schema_path)
        log.info("Reading Mapping ...")
        self.mapping = ReadCSV(self.mapping_path)

    def CheckSchema(self):
        log.info("Checking schema ...")
        self.schema_path = os.path.join(RESOURCES_DIR,"schema","schema.fbs")
        if not os.path.exists(self.schema_path):    
            log.info("    File schema.fbs was not found, downloading...")
            urllib.request.urlretrieve("https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/schema/schema.fbs",
                                   "schema.fbs")
            log.info("    Downloaded schema.fbs")
        else:
            log.info("    File schema.fbs found.")

    def InitializeEnv(self, source_model_path, mapping_path):
        os.mkdir(MODELS_DIR)
        for directory in ["source", "sub", "final"]:
            sub_dir = os.path.join(MODELS_DIR, directory)
            os.mkdir(sub_dir)
            for ext in ["tflite", "json"]:
                ext_dir = os.path.join(sub_dir, ext)
                os.mkdir(ext_dir)

        model_filename = source_model_path.split("/")[len(source_model_path.split("/"))-1]
        self.source_model_path = os.path.join(SOURCE_DIR, "tflite", model_filename)
        CopyFile(source_model_path, self.source_model_path)

        os.mkdir(MAPPING_DIR)
        mapping_filename = mapping_path.split("/")[len(mapping_path.split("/"))-1]
        self.mapping_path = os.path.join(MAPPING_DIR, mapping_filename)
        CopyFile(mapping_path, MAPPING_DIR)

        self.CheckSchema()

    def Clean(self, all: bool):
        dirs_to_clean = []
        if all:
            dirs_to_clean.extend([MODELS_DIR, MAPPING_DIR])
        else:
            dirs_to_clean.append(MAPPING_DIR)

        for directory in dirs_to_clean:
            if os.path.isdir(directory): 
                RunTerminalCommand("rm", "-rf", directory)   

    def AnalyseMapping(self):
        log.info("Analysing Mapping ...")
        self.final_mapping = {}
        sequence = {
            "NEW": True,
            "IN" : False,
            "END": False
        }
        sequence_tracker = dict.fromkeys(HW_ID, 0)
        for i, op in enumerate(self.mapping):
            current_hw = HW_ID[op[1]]
            if sequence["NEW"]:
                sequence_tracker[current_hw]+=1
                self.final_mapping[current_hw+str(sequence_tracker[current_hw])] = [i]
                sequence["NEW"] = False
                sequence["IN"] = True
            elif sequence["IN"]:
                self.final_mapping[current_hw+str(sequence_tracker[current_hw])].append(i)

            if (i+1 == len(self.mapping)):
                sequence["NEW"] = False
                sequence["IN"]  = False
                sequence["END"] = True
            else:
                sequence["END"] = (op[1] != self.mapping[i+1][1])
                if sequence["END"]:
                    sequence["NEW"] = True
                    sequence["IN"]  = False
        log.info("Final Mapping Generated!")
        for e in self.final_mapping:
            ops = self.final_mapping[e]
            log.info("        Operations : " + ", ".join(str(op) for op in ops))
            log.info("        Target HW  : " + e[:3])

        final_mapping_path = os.path.join(RESOURCES_DIR, "final_mapping.json")
        with open(final_mapping_path, "w") as fout:
            json.dump(self.final_mapping, fout, indent=2)

    def ReadSourceModel(self):
        self.source_model.Convert("tflite", "json")

    def CreateSubmodels(self):
        for block_key in self.final_mapping.keys():
            submodel = Submodel(self.source_model.json)
            submodel.AddOps(self.final_mapping[block_key])
            submodel.Save(block_key)
            submodel.Convert("json", "tflite")

    def CompileForEdgeTPU(self):
        docker = Docker("debian-docker")
        for submodel in os.listdir(os.path.join(SUB_DIR, "tflite")):
            if submodel.startswith("submodel_tpu"):
                docker.Copy(submodel, "host", "docker")
                docker.Compile(submodel)
                compiled_name  = "{0}_edgetpu.tflite".format(submodel.split(".")[0])
                docker.Copy(compiled_name, "docker", "host")
                docker.Clean()
        del docker

    def Run(self):
        self.AnalyseMapping()
        self.ReadSourceModel()
        self.CreateSubmodels()
        self.CompileForEdgeTPU()
        
    def __del__(self):
        self.Clean(False)
           
def main():
    args = ParseArgs()
    optimizer = Optimizer(args.Model, args.Mapping)
    try:
        optimizer.Run()
    except Exception as e:
        optimizer.Clean(True)
        log.error(e)
        log.error("ERROR: Failed to run optimizer!")
    finally:
        del optimizer

if __name__ == '__main__':
    main()
    pass