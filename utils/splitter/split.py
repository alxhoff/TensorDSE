import os
import sys
import json
import urllib
import argparse
import multiprocessing

from .model import Model, Submodel
from .utils import CopyFile, RunTerminalCommand, ReadJSON
from utils.logging.logger import log


SPLITTER_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SPLITTER_DIR, "models")
MAPPING_DIR = os.path.join(SPLITTER_DIR, "mapping")

SOURCE_DIR = os.path.join(MODELS_DIR, "source")
SUB_DIR = os.path.join(MODELS_DIR, "sub")
LAYERS_DIR = os.path.join(SUB_DIR, "tflite")
COMPILED_DIR = os.path.join(MODELS_DIR, "sub", "compiled")
UTILS_DIR = os.path.dirname(SPLITTER_DIR)
WORK_DIR = os.path.dirname(UTILS_DIR)
RESOURCES_DIR = os.path.join(WORK_DIR, "resources")


class Splitter:
    def __init__(self, model_summary: dict) -> None:
        self.CheckSchema()
        self.models = []
        self.models_details = []
        self.summary = model_summary
        self.submodel_list = multiprocessing.Manager().list()


    def CheckSchema(self):
        log.info("Checking schema ...")
        self.schema_path = os.path.join(RESOURCES_DIR, "schema", "schema.fbs")
        if not os.path.exists(self.schema_path):
            log.info("    File schema.fbs was not found, downloading...")
            urllib.request.urlretrieve(
                "https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/schema/schema.fbs",
                "schema.fbs",
            )
            log.info("    Downloaded schema.fbs")
        else:
            log.info("    File schema.fbs found.")


    def InitializeEnv(self, source_model: Model):

        source_model_path = source_model.paths["tflite"]
        log.info(
            "Source Model saved under: {}".format(
                source_model_path
                )
            )
        if os.path.exists(MODELS_DIR):
            RunTerminalCommand("rm", "-rf", MODELS_DIR)
        os.mkdir(MODELS_DIR)
        os.mkdir(os.path.join(MODELS_DIR, source_model.name))
        
        for directory in ["source", "sub", "final"]:
            sub_dir = os.path.join(MODELS_DIR, source_model.name, directory)
            if os.path.exists(sub_dir):
                RunTerminalCommand("rm", "-rf", sub_dir)
            os.mkdir(sub_dir)
            for ext in ["tflite", "json"]:
                ext_dir = os.path.join(sub_dir, ext)
                if os.path.exists(ext_dir):
                    RunTerminalCommand("rm", "-rf", ext_dir)
                os.mkdir(ext_dir)

        target_path = os.path.join(os.path.join(MODELS_DIR, source_model.name, "source"), "tflite", "{}.tflite".format(source_model.name))
        CopyFile(source_model.paths["tflite"], target_path)
        source_model.paths["tflite"] = target_path


    def Clean(self, all: bool):
        dirs_to_clean = []
        if all:
            dirs_to_clean.extend([MODELS_DIR, MAPPING_DIR])
        else:
            dirs_to_clean.append(MAPPING_DIR)

        for directory in dirs_to_clean:
            if os.path.isdir(directory):
                RunTerminalCommand("rm", "-rf", directory)


    def CreateLayerMatrix(self) -> None:
        """Parses the parsed JSON model summary containing the mappings to create
        a summarized list per model containing where each list contains a
        tupple of layer index in the model, layer type (eg. "conv_2d), and
        the device name of where the layer should be mapped (eg. "cpu1").
        """

    
        for model in self.summary["models"]:
            layers = []
            for j, layer in enumerate(model["layers"]):
                layers.append((j, layer["type"], layer["mapping"]))
            self.models_details.append(layers)
            self.models.append(Model(model.get("path", ""), self.schema_path))


    def CreateSubmodelLayerSequences(self) -> None:
        """From the mappings created by CreateLayerMatrix, sequential layers
        that are mapped to the same hardware device are grouped to create
        layer sequences that will later be compiled into submodels for execution.
        This is done on a per-model basis.
        """

        self.model_layer_sequences = []

        for model in self.models_details:
            layer_seuqences = []
            self.model_layer_sequences.append(layer_seuqences)

            prev_layers_hardware = None
            current_sequence = []

            for layer in model:
                if (
                    prev_layers_hardware is not None
                    and prev_layers_hardware != layer[2]
                ):
                    layer_seuqences.append(current_sequence)
                    current_sequence = []

                current_sequence.append(layer)
                prev_layers_hardware = layer[2]

            layer_seuqences.append(current_sequence)


    def ReadSourceModels(self):
        for model in self.models:
            self.InitializeEnv(model)
            model.Convert("tflite", "json")


    def CompileAndSaveSubmodel(
        self, layer_sequence, model_index, sequence_index
    ) -> None:
        """Compiles a submodel from a sequence of layers

        Args:
            layer_sequence (list(tupples)): A list containing the layers that should
            be compiled into the submodel.
            model_index (int): Index showing which model in the list of all models
            is currently being compiled.
            sequence_index (int): The index at where the sequence appears in the
            set of sequences to be run on the respective hardware.
        """
        
        if len(layer_sequence) == 1:
            ops_range = layer_sequence[0][0]
        else:
            ops_range = '-'.join(map(str, [layer_sequence[0][0], layer_sequence[-1][0]]))
    
        ops_name = f"ops{ops_range}"

        submodel = Submodel(
            self.models[model_index],
            ops_name,
            layer_sequence[0][2],
            sequence_index,
        )
        log.info("OK")
        log.info(
            "Adding Operations of Index ("
            + ", ".join(str(op[0]) for op in layer_sequence)
            + ") to Shell Model ..."
        )
        submodel.AddOps(layer_sequence)
        log.info("OK")
        log.info(
            "Saving Model {} | Submodel {} | Operations: {} | Target HW: {} ...".format(
                model_index,
                sequence_index,
                ", ".join([str(layer[1]) for layer in layer_sequence]),
                layer_sequence[0][2],
            )
        )
        submodel.Save()
        log.info("OK")
        log.info(
            "Converting Submodel {0} from JSON to TFLite ...".format(
                str(sequence_index)
            )
        )
        submodel.Convert("json", "tflite")
        self.submodel_list.append(submodel)
        log.info("OK\n")


    def CreateSubmodels(self, sequences):
        """Create the individual submodels for either sequences of sequential
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
                with multiprocessing.Pool() as pool:
                    items = [(sequence, i, j) for j, sequence in enumerate(model)]
                    pool.starmap(self.CompileAndSaveSubmodel, items)


        # Else each individual layer will be run in its own inference session
        # which gives clearer overhead vs. execution time numbers
        else:
            for i, model in enumerate(self.models_details):
                with multiprocessing.Pool() as pool:
                    items = [([layer], i, j) for j, layer in enumerate(model)]
                    pool.starmap(self.CompileAndSaveSubmodel, items)


    def CompileForEdgeTPU(self, bm=True):
        for submodel in self.submodel_list:
            if bm:
                submodel.Compile(submodel.source_model_name)
            else:
                if ("tpu" in submodel.name):
                    submodel.Compile(submodel.source_model_name)


    def Run(self, sequences=False):
        log.info("[SPLIT] Started")
        self.CreateLayerMatrix()
        log.info("[SPLIT] Created matrix")
        self.CreateSubmodelLayerSequences()
        log.info("[SPLIT] Submodel layer sequences created")
        self.ReadSourceModels()
        log.info("[SPLIT] Source models read")
        self.CreateSubmodels(sequences=sequences)
        log.info("[SPLIT] Submodels created")


    def SaveSequences(self):
        seq_path = os.path.join(SPLITTER_DIR, "model_layer_sequences.json")
        with open(seq_path, 'w') as f:
            json.dump(self.model_layer_sequences, f)


    def __del__(self):
        self.Clean(False)


def GetArgs():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    
    parser.add_argument(
            "-s",
            "--summary",
            default="resources/model_summaries/example_summaries/MNIST/MNIST_full_quanitization_summary.json",
            help="File that contains a model summary with mapping annotations"
            )
    
    parser.add_argument(
                "-p",
                "--platform",
                default="desktop",
                help="Platform supporting the profiling/deployment process",
            )

    parser.add_argument(
                "-q",
                "--sequences",
                default=False,
                help="Flag signaling the splitter to split for deployment.",
            )

    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = GetArgs()

    if args.summary is not None:
        model_summary = ReadJSON(args.summary)
        if model_summary is None:
            log.error("The provided Model Summary is empty!")
            sys.exit(-1)

    # Create single operation models/layers from the operations in the provided model
        
    splitter = Splitter(model_summary=model_summary)

    try:
        log.info("Running Model Splitter ...")
        splitter.Run(sequences=args.sequences)
        log.info("Splitting Process Complete!\n")
        splitter.CompileForEdgeTPU()
        log.info("[SPLIT MODEL] Models successfully compiled!")
        if (args.sequences is True):
            splitter.SaveSequences()

    except Exception as e:
        splitter.Clean(True)
        log.error("Failed to run splitter! {}".format(str(e)))
        raise(e)
            
