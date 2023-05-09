import os
import json
import urllib
from utils import ParseArgs, LoggerInit, RunTerminalCommand, ReadJSON, CopyFile, MoveFile

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

    def RunDockerCommand(self, command: str):
        log.info(RunTerminalCommand('sudo', 'docker', 'exec', '-ti', 'debian-docker', 'sh', '-c', command, save_output=True))

    def CreateSubDir(self):
        create_subdir_command = "mkdir /home/sub"
        self.RunDockerCommand(create_subdir_command)

    def Start(self):
        try:
            log.info("Starting Docker Container: {0}".format(self.name))
            log.info(RunTerminalCommand("sudo", "docker", "start", self.name, save_output=True))
            log.info("OK\n")
        except Exception as e:
            log.error("Failed to start Docker Container! Potential Cause: {}".format(str(e)))

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
        self.RunDockerCommand(compiling_command)

    def Clean(self):
        command = "find -type f -name '*submodel_tpu*' -delete"
        self.RunDockerCommand(command)

    def __del__(self):
        command = "rm -rf /home/sub"
        self.RunDockerCommand(command)
        
class Model:
    def __init__(self, path_to_model: str, schema_path: str):
        self.paths = {"json": "",
                      "tflite": ""}
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

class Submodel(Model):
    def __init__(self, source_model_json: dict):
        CopyFile(os.path.join(RESOURCES_DIR, "shell", "shell_model.json"),
                 os.path.join(SUB_DIR, "json", "shell_model.json"))
        super().__init__(path_to_model=os.path.join(SUB_DIR, "json", "shell_model.json"),
                          schema_path=os.path.join(RESOURCES_DIR, "schema", "schema.fbs"))
        self.json = ReadJSON(self.paths["json"])
        self.source_model_json = source_model_json

    def AddOps(self, layers):
        """Adds the appropriate operations, specified by the given layers, to
        the submodel in preparation for saving the submodel.

        Args:
            layers (_type_): List of layer mappings, ie. (layer index, layer
            type, target hardware) tupples, that should be compiled into a submodel.
        """

        #Read Main Graph
        source_graph = self.source_model_json["subgraphs"][0]

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
        self.submodel_number = '_'.join([str(layer[0]) for layer in layers])

    def Save(self, target_hardware: str, sequence_index: int):
        """Saves a submodel to a JSON file, labeled using the target hardware,
        the index of the sequence, and the indexes of the layers (in their model)
        that are contained withing the submodel.

        Args:
            target_hardware (str): Name of the hardware that the submodel is to
            be executed on
            sequence_index (int): The index at where the sequence appears in the
            set of sequences to be run on the respective hardware
        """

        submodel_filename = "submodel{0}_{1}_{2}.json".format(sequence_index, target_hardware.lower(), self.submodel_number)
        submodel_filepath = os.path.join(SUB_DIR, "json", submodel_filename)
        MoveFile(self.paths["json"], submodel_filepath)
        self.paths["json"] = submodel_filepath
        with open(submodel_filepath, "w") as fout:
            json.dump(self.json, fout, indent=2)

class Optimizer:
    def __init__(self, source_model_path, mapping_path):
        import json

        log.info("Initializing Environment ...")
        self.InitializeEnv(source_model_path, mapping_path)
        log.info("Initializing Source Model ...")
        self.source_model = Model(self.source_model_path, self.schema_path)
        log.info("Source Model saved under: {}".format(os.path.join(MODELS_DIR, "source", "tflite")))
        log.info("Reading Mapping ...")
        with open(self.mapping_path, 'r') as f:
            self.mapping = json.load(f)

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
        self.CheckSchema()
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        for directory in ["source", "sub", "final"]:
            sub_dir = os.path.join(MODELS_DIR, directory)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            for ext in ["tflite", "json"]:
                ext_dir = os.path.join(sub_dir, ext)
                if not os.path.exists(ext_dir):
                    os.mkdir(ext_dir)

        model_filename = source_model_path.split("/")[len(source_model_path.split("/"))-1]
        self.source_model_path = os.path.join(SOURCE_DIR, "tflite", model_filename)
        CopyFile(source_model_path, self.source_model_path)

        if not os.path.exists(MAPPING_DIR):
            os.mkdir(MAPPING_DIR)
        mapping_filename = mapping_path.split("/")[len(mapping_path.split("/"))-1]
        self.mapping_path = os.path.join(MAPPING_DIR, mapping_filename)
        CopyFile(mapping_path, MAPPING_DIR)
        log.info("Mapping saved under: {}".format(os.path.join(MAPPING_DIR)))

    def Clean(self, all: bool):
        log.info("Cleaning Directory ...\n")
        dirs_to_clean = []
        if all:
            dirs_to_clean.extend([MODELS_DIR, MAPPING_DIR])
        else:
            dirs_to_clean.append(MAPPING_DIR)

        for directory in dirs_to_clean:
            if os.path.isdir(directory):
                RunTerminalCommand("rm", "-rf", directory)

    def AnalyseMappings(self) -> None:
        """ Parses the parsed JSON model summary containing the mappings to create
        a summarized list per model containing where each list contains a
        tupple of layer index in the model, layer type (eg. "conv_2d), and
        the device name of where the layer should be mapped (eg. "cpu1").
        """

        log.info("Analysing Mapping ...")

        self.final_mapping = []

        for model in self.mapping:
            model_layers = []
            self.final_mapping.append(model_layers)
            for i, layer in enumerate(model["layers"]):
                model_layers.append((i, layer["type"], layer["mapping"]))

        for i, model in enumerate(self.final_mapping):
            for j, layer in enumerate(model):
                log.info("Model #{}, layer #{} mapped to {}".format(i, j, layer))

    def CreateSubmodelLayerSequences(self) -> None:
        """ From the mappings created by AnalyseMappings, sequential layers
        that are mapped to the same hardware device are grouped to create
        layer sequences that will later be compiled into submodels for execution.
        This is done on a per-model basis.
        """

        self.model_layer_sequences = []

        for model in self.final_mapping:
            layer_seuqences = []
            self.model_layer_sequences.append(layer_seuqences)

            prev_layers_hardware = None
            current_sequence = []

            for layer in model:

                if prev_layers_hardware is not None and prev_layers_hardware != layer[2]:
                    layer_seuqences.append(current_sequence)
                    current_sequence = []

                current_sequence.append(layer)
                prev_layers_hardware = layer[2]

            layer_seuqences.append(current_sequence)


    def ReadSourceModel(self) -> None:
        log.info("Converting Source Model from TFLite to JSON ...")
        self.source_model.Convert("tflite", "json")
        log.info("OK\n")

    def CompileAndSaveSubmodel(self, layer_sequence, model_index, sequence_index) -> None:
        """Compiles a submodel from a sequence of layers

        Args:
            layer_sequence (list(tupples)): A list containing the layers that should
            be compiled into the submodel.
            model_index (int): Index showing which model in the list of all models
            is currently being compiled.
            sequence_index (int): The index at where the sequence appears in the
            set of sequences to be run on the respective hardware.
        """

        log.info("Initializing shell model for submodel {} from model {}...".format(sequence_index, model_index))
        submodel = Submodel(self.source_model.json)
        log.info("OK")
        log.info("Adding Operations (" + ", ".join(op[2] for op in layer_sequence) + ") to Shell Model ...")
        submodel.AddOps(layer_sequence)
        log.info("OK")
        log.info("Saving Model {} | Submodel {} | Operations: {} | Target HW: {} ...".format(
            model_index, sequence_index, ", ".join([str(layer[2]) for layer in layer_sequence]), layer_sequence[0][2]))
        submodel.Save(layer_sequence[0][2], sequence_index)
        log.info("OK")
        log.info("Converting Submodel {0} from JSON to TFLite ...".format(str(sequence_index)))
        submodel.Convert("json", "tflite")
        log.info("OK\n")

    def CreateSubmodels(self, sequences=False):
        """ Create the individual submodels for either sequences of sequential
        layers that are executed on the same hardware unit or for individual layers

        Args:
            sequences (bool, optional): If sequences should be compiled together
            or if individual layers should be compiled into submodels alone.
            Defaults to False.
        """

        # If we want to compile sequential layers mapped onto the same hardware
        # into single inference sessions
        if sequences:
            for i, model in enumerate(self.model_layer_sequences):
                for j, sequence in enumerate(model):
                    self.CompileAndSaveSubmodel(layer_sequence=sequence,
                                                model_index=i, sequence_index=j)
        # Else each individual layer will be run in its own inference session
        # which gives clearer overhead vs. execution time numbers
        else:
            for i, model in enumerate(self.final_mapping):
                for j, layer in enumerate(model):
                    self.CompileAndSaveSubmodel(layer_sequence=[layer],
                                                model_index=i, sequence_index=j)

    def CompileForEdgeTPU(self):
        docker = Docker("debian-docker")
        for i, submodel in enumerate(sorted(os.listdir(os.path.join(SUB_DIR, "tflite")))):
            if ((submodel.startswith("submodel{0}_tpu".format(i))) and (not(submodel.endswith("_edgetpu.tflite")))):
                docker.Copy(submodel, "host", "docker")
                docker.Compile(submodel)
                compiled_name  = "{0}_edgetpu.tflite".format(submodel.split(".")[0])
                docker.Copy(compiled_name, "docker", "host")
                docker.Clean()
        del docker

    def Run(self):
        self.AnalyseMappings()
        self.CreateSubmodelLayerSequences()
        self.ReadSourceModel()
        self.CreateSubmodels()
        self.CompileForEdgeTPU()
        
    def __del__(self):
        self.Clean(False)
           
def main():
    args = ParseArgs()
    optimizer = Optimizer(args.Model, args.Mapping)
    try:
        log.info("Running Optimizer ...")
        optimizer.Run()
        log.info("Optiizing Process Complete!\n")
    except Exception as e:
        optimizer.Clean(True)
        log.error("Failed to run optimizer! {}".format(str(e)))
    finally:
        del optimizer

if __name__ == '__main__':
    main()
    pass