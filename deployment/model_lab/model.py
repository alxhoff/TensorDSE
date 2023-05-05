import os
import json
from .utils import RunTerminalCommand, ReadJSON, CopyFile, MoveFile

WORK_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(WORK_DIR, "models")
MAPPING_DIR   = os.path.join(WORK_DIR, "mapping")
RESOURCES_DIR = os.path.join(WORK_DIR, "resources")
SOURCE_DIR    = os.path.join(MODELS_DIR, "source")
SUB_DIR       = os.path.join(MODELS_DIR, "sub")
FINAL_DIR     = os.path.join(MODELS_DIR, "final")

class Model:
    def __init__(self, path_to_model: str, schema_path: str):
        self.paths = {"json": "",
                      "tflite": "",
                      "edgetpu_tflite": ""}
        for ext in self.paths.keys():
            if path_to_model.endswith(ext):
                self.paths[ext] = path_to_model
        self.schema = schema_path
    
    def Convert(self, source_ext: str, target_ext: str):
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
    
    def Compile(self):
        compiling_command = "edgetpu_compiler -o {0} -s {1}".format(os.path.join(SUB_DIR, "tflite/"), self.paths["tflite"])
        self.RunDockerCommand(compiling_command)
        self.paths["edgetpu_tflite"] = self.paths["tflite"].split(".")[0] + "_edgetpu.tflite"

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
        
    def Save(self, block_key: str, index: int):
        submodel_filename = "submodel{0}_{1}_{2}.json".format(index, block_key.lower(), self.submodel_number)
        submodel_filepath = os.path.join(SUB_DIR, "json", submodel_filename)
        MoveFile(self.paths["json"], submodel_filepath)
        self.paths["json"] = submodel_filepath
        with open(submodel_filepath, "w") as fout:
            json.dump(self.json, fout, indent=2)
