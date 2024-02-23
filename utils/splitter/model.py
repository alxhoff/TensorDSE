"""
    Missing  Docstring: TODO
"""

import os
import json

from utils.logging.logger import log
from utils.splitter import RESOURCES_DIR, MODELS_DIR

from .utils import run_command_and_echo, read_json_file, copy_file, move_file

class Model:
    """
        Missing  Docstring: TODO
    """

    def __init__(self, path_to_model: str, schema_path: str):
        try:
            model_basename = os.path.basename(path_to_model)
            self.name = model_basename.split(".")[0]
            self.index = 0
        except IOError:
            log.fatal("Could not fetch model name from %s. Aborting!", path_to_model)

        self.paths = {"json": "",
                      "tflite": "",
                      "edgetpu_tflite": ""}
        for ext in self.paths:
            if path_to_model.endswith(ext):
                self.paths[ext] = path_to_model
        self.schema = schema_path
        self.json = None

    def convert(self, source_ext: str, target_ext: str):
        """
            Missing  Docstring: TODO
        """
        try:
            if ([source_ext, target_ext] == ["json", "tflite"]):
                run_command_and_echo("flatc", "-b", self.schema, self.paths["json"])
                tmp_filename = f"{self.name}.tflite"
                self.paths["tflite"] = self.paths["json"].replace(source_ext, target_ext)
                move_file(tmp_filename, self.paths["tflite"])
            elif ([source_ext, target_ext] == ["tflite", "json"]):
                run_command_and_echo(
                    "flatc",
                    "-t",
                    "--strict-json",
                    "--defaults-json",
                    self.schema,
                    "--",
                    self.paths["tflite"]
                    )
                tmp_filename = f"{self.name}.json"
                self.paths["json"] = self.paths["tflite"].replace(source_ext, target_ext)
                move_file(tmp_filename, self.paths["json"])
                self.json = read_json_file(self.paths["json"])
        except RuntimeError:
            log.fatal("An error occured while convert using flatc")

    def edgetpu_compile(self, parent_model_name: str):
        """
            Missing  Docstring: TODO
        """
        compiled_dir = os.path.join(MODELS_DIR, parent_model_name, "sub", "compiled")
        if not os.path.exists(compiled_dir):
            os.mkdir(compiled_dir)
        compiling_command = f'/usr/bin/edgetpu_compiler -o {compiled_dir} -s {self.paths["tflite"]}'
        os.system(compiling_command)
        self.paths["edgetpu_tflite"] = os.path.join(
            compiled_dir,
            os.path.basename(self.paths["tflite"]).split(".")[0] + "_edgetpu.tflite"
            )

class Submodel(Model):
    """
        Missing  Docstring: TODO
    """

    def __init__(self,
                 source_model: Model,
                 op_name: str,
                 target_hardware: str,
                 sequence_index: int):

        name = f'submodel_{sequence_index}_{op_name}_{"bm" if target_hardware.lower() == "" else target_hardware.lower()}'
        self.dirs = {
            "json": os.path.join(
                MODELS_DIR,
                f"model_{source_model.index}_{source_model.name}",
                "sub",
                "json",
                name
                ),
            "tflite": os.path.join(
                MODELS_DIR,
                f"model_{source_model.index}_{source_model.name}",
                "sub",
                "tflite",
                name
                )
            }
        os.mkdir(self.dirs["json"])
        os.mkdir(self.dirs["tflite"])
        copy_file(os.path.join(RESOURCES_DIR, "shell", "shell_model.json"),
                 os.path.join(self.dirs["json"], "shell_model.json"))
        super().__init__(path_to_model=os.path.join(self.dirs["json"], "shell_model.json"),
                          schema_path=os.path.join(RESOURCES_DIR, "schema", "schema.fbs"))
        self.json = read_json_file(self.paths["json"])
        self.source_model_name = source_model.name
        self.source_model_json = source_model.json
        self.name = name
        self.source_model_folder_name = f"model_{source_model.index}_{source_model.name}"

    def add_ops(self, layers):
        """Adds the appropriate operations, specified by the given layers, to
        the submodel in preparation for saving the submodel.

        Args:
            layers (_type_): List of layer mappings, ie. (layer index, layer
            type, target hardware) tupples, that should be converted into a submodel.
        """

        #Read Main Graph
        source_graph = self.source_model_json["subgraphs"][0].copy()

        #Add version
        new_version = self.source_model_json["version"]

        #Add Operators from Main Graph and add them according to index op
        new_ops = []
        for op_index in [layer[0] for layer in layers]:
            new_ops.append(source_graph["operators"][op_index])

        #Add the OperatorCodes of the Newly added Operators and update their opcode_index
        new_opcodes = []
        for new_op in new_ops:
            if self.source_model_json["operator_codes"][new_op["opcode_index"]] not in new_opcodes:
                new_opcodes.append(
                    self.source_model_json["operator_codes"][new_op["opcode_index"]].copy()
                    )
                new_op["opcode_index"] = len(new_opcodes) - 1
            else:
                new_op["opcode_index"] = new_opcodes.index(
                    self.source_model_json["operator_codes"][new_op["opcode_index"]]
                    )

        #Add Tensors according to added Operators
        new_tensors = []
        tensor_indexes = []
        for new_op in new_ops:
            for entry in ["inputs", "outputs"]:
                for i, op_entry in enumerate(new_op[entry]):
                    if op_entry in tensor_indexes:
                        new_op[entry][i] = new_tensors.index(source_graph["tensors"][op_entry])
                    else:
                        tensor_indexes.append(op_entry)
                        new_tensors.append(source_graph["tensors"][op_entry].copy())
                        new_op[entry][i] = len(new_tensors) - 1

        #Add Submodel Input and Output Tensors
        new_inputs = []
        new_outputs = []
        if new_opcodes[new_ops[0]["opcode_index"]]["deprecated_builtin_code"] == 0:
            for i_t in new_ops[0]["inputs"]:
                new_inputs.append(i_t)
        else:
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
        new_buffers = [{}]
        for i,new_tensor in enumerate(new_tensors):
            buffer_index = new_tensor["buffer"]
            if buffer_index > 0:
                new_buffers.append(self.source_model_json["buffers"][buffer_index].copy())
                new_tensor["buffer"] = len(new_buffers) - 1

        #Add metadata
        new_metadata = []
        if "metadata" in self.source_model_json.keys():
            for i, source_metadata in enumerate(self.source_model_json["metadata"]):
                new_buffers.append(
                    self.source_model_json["buffers"][source_metadata["buffer"]].copy()
                    )
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
        #                self.json["signature_defs"][i] =
        #    self.source_model_json["signature_defs"][i].copy()
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

    def save(self):
        """Saves a submodel to a JSON file, labeled using the target hardware,
        the index of the sequence, and the indexes of the layers (in their model)
        that are contained withing the submodel.
        Args:
            target_hardware (str): Name of the hardware that the submodel is to
            be executed on
            sequence_index (int): The index at where the sequence appears in the
            set of sequences to be run on the respective hardware
        """
        submodel_filename = f"{self.name}.json"
        submodel_filepath = os.path.join(self.dirs["json"], submodel_filename)
        move_file(self.paths["json"], submodel_filepath)
        self.paths["json"] = submodel_filepath
        with open(submodel_filepath, "w", encoding="utf-8") as fout:
            json.dump(self.json, fout, indent=2)
