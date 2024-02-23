"""
    Missing  Docstring: TODO
"""

import os
import json
import urllib
import argparse
import multiprocessing

from utils.logging.logger import log
from utils.splitter import SPLITTER_DIR, RESOURCES_DIR, MODELS_DIR, MAPPING_DIR

from .model import Model, Submodel
from .utils import copy_file, run_command_and_echo, read_json_file

class Splitter:
    """
    Missing  Docstring: TODO
    """
    def __init__(self, model_summary: dict) -> None:
        self.models = []
        self.models_details = []
        self.model_layer_sequences = []

        self.summary = model_summary
        self.submodel_list = multiprocessing.Manager().list()

        self.check_schema()


    def check_schema(self):
        """
            Missing  Docstring: TODO
        """
        log.info("Checking schema ...")
        self.schema_path = os.path.join(RESOURCES_DIR, "schema", "schema.fbs")
        if not os.path.exists(self.schema_path):
            log.info("    File schema.fbs was not found, downloading...")
            urllib.request.urlretrieve(
                "https://github.com/tensorflow/\
                    tensorflow/raw/master/tensorflow/lite/schema/schema.fbs",
                "schema.fbs",
            )
            log.info("    Downloaded schema.fbs")
        else:
            log.info("    File schema.fbs found.")


    def single_model_env_init(self, source_model: Model):
        """
            Missing  Docstring: TODO
        """
        source_model_path = source_model.paths["tflite"]
        log.info("Source Model saved under: %s", source_model_path)

        os.mkdir(
            os.path.join(
                MODELS_DIR,
                f"model_{source_model.index}_{source_model.name}")
        )

        for directory in ["source", "sub", "final"]:
            sub_dir = os.path.join(
                MODELS_DIR,
                f"model_{source_model.index}_{source_model.name}",
                directory
                )
            os.mkdir(sub_dir)
            for ext in ["tflite", "json"]:
                ext_dir = os.path.join(sub_dir, ext)
                if os.path.exists(ext_dir):
                    run_command_and_echo("rm", "-rf", ext_dir)
                os.mkdir(ext_dir)

        target_path = os.path.join(
            MODELS_DIR,
            f"model_{source_model.index}_{source_model.name}",
            "source",  
            "tflite",
            f"{source_model.name}.tflite"
            )
        copy_file(source_model.paths["tflite"], target_path)
        source_model.paths["tflite"] = target_path


    def clean(self, all_dirs: bool):
        """
            Missing  Docstring: TODO
        """
        dirs_to_clean = []
        if all_dirs is True:
            dirs_to_clean.extend([MODELS_DIR, MAPPING_DIR])
        else:
            dirs_to_clean.append(MAPPING_DIR)

        for directory in dirs_to_clean:
            if os.path.isdir(directory):
                run_command_and_echo("rm", "-rf", directory)


    def create_layer_matrix(self) -> None:
        """Parses the parsed JSON model summary containing the mappings to create
        a summarized list per model containing where each list contains a
        tupple of layer index in the model, layer type (eg. "conv_2d), and
        the device name of where the layer should be mapped (eg. "cpu1").
        """

        for i, model in enumerate(self.summary["models"]):
            layers = []
            for j, layer in enumerate(model["layers"]):
                layers.append((j, layer["type"], layer["mapping"]))
            self.models_details.append(layers)
            m = Model(model.get("path", ""), self.schema_path)
            m.index = i
            self.models.append(m)


    def create_submodel_layer_sequences(self) -> None:
        """From the mappings created by CreateLayerMatrix, sequential layers
        that are mapped to the same hardware device are grouped to create
        layer sequences that will later be compiled into submodels for execution.
        This is done on a per-model basis.
        """

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


    def multi_model_env_init(self):
        """
            Missing  Docstring: TODO
        """
        if os.path.exists(MODELS_DIR):
            run_command_and_echo("rm", "-rf", MODELS_DIR)
        os.mkdir(MODELS_DIR)

        for model in self.models:
            self.single_model_env_init(model)
            model.convert("tflite", "json")


    def compile_and_save_submodel_sequence(
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
        log.info("Adding Operations of Index (%s) to Shell Model ...",
                 ", ".join(str(op[0]) for op in layer_sequence))
        submodel.add_ops(layer_sequence)
        log.info("OK")
        log.info("Saving Model %s | Submodel %s | Operations: %s | Target HW: %s",
                model_index,
                sequence_index,
                ", ".join([str(layer[1]) for layer in layer_sequence]),
                layer_sequence[0][2]
                )
        submodel.save()
        log.info("OK")
        log.info("Converting Submodel %s from JSON to TFLite ...", str(sequence_index))
        submodel.convert("json", "tflite")
        self.submodel_list.append(submodel)
        log.info("OK\n")


    def create_submodels(self, sequences):
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
                    pool.starmap(self.compile_and_save_submodel_sequence, items)


        # Else each individual layer will be run in its own inference session
        # which gives clearer overhead vs. execution time numbers
        else:
            for i, model in enumerate(self.models_details):
                with multiprocessing.Pool() as pool:
                    items = [([layer], i, j) for j, layer in enumerate(model)]
                    pool.starmap(self.compile_and_save_submodel_sequence, items)


    def compile_for_edge_tpu(self, bm=True):
        """
            Missing  Docstring: TODO
        """
        for submodel in self.submodel_list:
            if bm:
                submodel.edgetpu_compile(submodel.source_model_folder_name)
            else:
                if "tpu" in submodel.name:
                    submodel.edgetpu_compile(submodel.source_model_folder_name)


    def run(self, sequences=False):
        """
            Missing  Docstring: TODO
        """

        log.info("[SPLIT] Started")
        self.create_layer_matrix()
        log.info("[SPLIT] Created matrix")
        self.create_submodel_layer_sequences()
        log.info("[SPLIT] Submodel layer sequences created")
        self.multi_model_env_init()
        log.info("[SPLIT] Source models read")
        self.create_submodels(sequences=sequences)
        log.info("[SPLIT] Submodels created")


    def save_sequences(self):
        """
            Missing  Docstring: TODO
        """
        seq_path = os.path.join(SPLITTER_DIR, "model_layer_sequences.json")
        with open(seq_path, 'w', encoding="utf-8") as f:
            json.dump(self.model_layer_sequences, f)


    def __del__(self):
        self.clean(False)


def get_arguments():
    """
        Missing  Docstring: TODO
    """
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

    parser.add_argument(
            "-s",
            "--summary",
            default="resources/model_summaries/example_summaries/MNIST/\
                MNIST_full_quanitization_summary.json",
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

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if args.summary is not None:
        model_summary_data = read_json_file(args.summary)
        if model_summary_data is None:
            log.fatal("The provided Model Summary is empty!")
    else:
        log.fatal("The provided Model Summary Path is not valid!")

    splitter = Splitter(model_summary=model_summary_data)

    try:
        log.info("Running Model Splitter ...")
        splitter.run(sequences=bool(args.sequences))
        log.info("Splitting Process Complete!\n")
        splitter.compile_for_edge_tpu()
        log.info("[SPLIT MODEL] Models successfully compiled!")
        if bool(args.sequences) is True:
            splitter.save_sequences()

    except RuntimeError:
        log.error("Failed to run splitter!")
        splitter.clean(True)
