import os
import json
import urllib
from .model import Model, Submodel
from .utils import ReadJSON, CopyFile, RunTerminalCommand
from .logger import log


MODEL_LAB_DIR       = os.path.dirname(os.path.abspath(__file__))
DEPLOYMENT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR          = os.path.join(MODEL_LAB_DIR, "models")
MAPPING_DIR         = os.path.join(MODEL_LAB_DIR, "mapping")
RESOURCES_DIR       = os.path.join(MODEL_LAB_DIR, "resources")
SOURCE_DIR          = os.path.join(MODELS_DIR, "source")
SUB_DIR             = os.path.join(MODELS_DIR, "sub")
FINAL_DIR           = os.path.join(MODELS_DIR, "final")

HW_ID = ["CPU", "GPU", "TPU", "XPU"]
MODES = ["benchmarking", "deployment"]

class Splitter:
    def __init__(self, mode_of_operation, source_model_path, mapping_path) -> None:
        log.info("Initializing Environment ...")
        self.InitializeEnv(source_model_path, mapping_path)
        log.info("Initializing Source Model ...")
        self.source_model = Model(self.source_model_path, self.schema_path)
        log.info("Source Model saved under: {}".format(os.path.join(MODELS_DIR, "source", "tflite")))
        if not(mapping_path == None):
            log.info("Reading Mapping ...")
            self.mapping = ReadJSON(self.mapping_path)
        self.mode = mode_of_operation
        
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
            os.mkdir(MODELS_DIR)
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

    def ReadSourceModel(self):
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
        for i, submodel in enumerate(self.submodel_list):
            submodel_path = submodel.paths["tflite"]
            if(self.mode == MODES[1]):
                if ((submodel_path.startswith("submodel{0}_tpu".format(i))) and (not(submodel_path.endswith("_edgetpu.tflite")))):
                    submodel.Compile()
                else:
                    continue
            elif(self.mode == MODES[0]):
                submodel.Compile()

    def Run(self):
        self.AnalyseMappings()
        self.CreateSubmodelLayerSequences()
        self.ReadSourceModel()
        self.CreateSubmodels()
        #self.CompileForEdgeTPU()
        
    def __del__(self):
        self.Clean(False)