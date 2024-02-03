"""
Missing  Docstring: TODO
"""

import os
import sys
import copy
import json
import argparse
import random
import traceback
import multiprocessing

import numpy as np

from utils.model import Model
from utils.logging.logger import log
from utils.splitter.utils import read_json_file
from utils.splitter.split import Splitter
from utils.splitter.split import MODELS_DIR
from utils.usb.process import process_streams

from utils.inference import Inference

from status_verifier import StatusVerifierSingleton as verifier


class Deployer:
    """
    Missing  Docstring: TODO
    """

    HARDWARE_LOCKS = {
        'cpu0': multiprocessing.Lock(),
        'gpu0': multiprocessing.Lock(),
        'tpu0': multiprocessing.Lock(),
    }


    def __init__(self,
                 native_deployment: bool,
                 data_module: str,
                 hardware: str) -> None:

        self.model_summary = verifier.get_model_summary(verifier)
        self.platform = verifier.get_platform(verifier)
        self.native_deployment = native_deployment
        self.data_module = data_module
        self.native_hardware = hardware

        self.models = []
        self.multi_models_sequences = []


    def split(self):
        """
        Missing  Docstring: TODO
        """

        model_layer_sequences = []

        if self.native_deployment is True:
            for model in self.model_summary["models"]:
                for layer in model["layers"]:
                    layer["mapping"] = self.native_hardware

        splitter = Splitter(self.model_summary)

        if self.platform == "desktop":
            try:
                log.info("Running Model Splitter ...")
                splitter.run(sequences=True)
                log.info("Splitting Process Complete!\n")
            except RuntimeError:
                splitter.clean(True)
                log.fatal("Failed to run splitter!")

            # Compiles created models/layers into Coral models for execution
            splitter.compile_for_edge_tpu(bm=False)
            log.info("Models successfully compiled!")
            model_layer_sequences = splitter.model_layer_sequences
        else:
            model_layer_sequences = read_json_file("utils/splitter/model_layer_sequences.json")

        self.multi_models_sequences = model_layer_sequences


    def set_random_input_data(self, m: Model):
        """
        Missing  Docstring: TODO
        """
        input_size = m.get_array_size_from_shape(m.input_shape)
        m.input_vector = np.array(
                np.random.random_sample(input_size), dtype=m.get_np_dtype(m.input_datatype)
                )


    def set_input_test_data_module(self, m: Model):
        """
        Missing  Docstring: TODO
        """
        module = __import__(self.data_module)
        for comp in self.data_module.split(".")[1:]:
            module = getattr(module, comp)
        dataset = module.GetData()["test_data"]
        m.input_vector = random.choice(dataset)


    def analyse_results(self):
        """
        Missing  Docstring: TODO
        """

        results_folder = os.path.join(os.getcwd(), "resources/deployment_results", self.platform)

        if not os.path.isdir(results_folder):
            log.error("%s is not a valid folder to store results!", results_folder)
            os.mkdir(results_folder)
            if not os.path.isdir(results_folder):
                log.fatal("Unable to create Results Folder to store Deployment results!")
                sys.exit(-1)

        for i, model in enumerate(self.models):
            results_path = os.path.join(results_folder, f"{model[0].parent}.json")
            model_results = dict()
            model_results["model_name"] = model[i].parent
            model_results["submodels"] = []
            model_results["total_inference_time (s)"] = 0
            for m in model:
                submodel = dict()
                submodel["name"] = m.model_name
                submodel["layers"] = m.details
                submodel["inference_time (s)"] = m.results[0]
                submodel["usb"] = dict()
                if not self.platform == "coral":
                    if "tpu" in m.delegate:
                        submodel["usb"] = process_streams(m.timers, m.results)
                model_results["submodels"].append(submodel)
                model_results["total_inference_time (s)"] += m.results[0]

            with open(results_path, 'w', encoding="utf-8") as json_file:
                json.dump(model_results, json_file, indent=4)


    def layer_deploy(self, m: Model):
        """
        Missing  Docstring: TODO
        """

        delegate_type  = m.delegate[:3]

        if delegate_type == "tpu":
            m.model_path = os.path.join(
                    MODELS_DIR,
                    m.parent,
                    "sub",
                    "compiled",
                    f"{m.model_name}_edgetpu.tflite"
                    )
        else:
            m.model_path = os.path.join(
                    MODELS_DIR,
                    m.parent,
                    "sub",
                    "tflite",
                    m.model_name,
                    f"{m.model_name}.tflite"
                    )

        inference_instance = Inference(model=m, count=1, platform=self.platform)
        m = inference_instance.invoke()

        return m


    def sequences_deploy(self, model_sequence, model_index):
        """
        Missing  Docstring: TODO
        """

        submodels = []
        model_name = os.path.basename(
            self.model_summary["models"][model_index]["path"]).split(".tflite")[0]

        for j, sequence in enumerate(model_sequence):
            layers = [self.model_summary["models"][model_index]["layers"][op[0]] for op in sequence]
            m = Model(layers, layers[0]["mapping"], model_name)

            if len(sequence) == 1:
                ops_range = sequence[0][0]
            else:
                ops_range = '-'.join(map(str, [sequence[0][0], sequence[-1][0]]))

            if self.native_deployment is False:
                m.model_name = f"submodel_{j}_{f'ops{ops_range}'}_{m.delegate}"
            else:
                m.model_name = model_name

            if j == 0:
                #Foced random input for rpi and dev board due to unsupported dataset installation
                if not self.platform == "desktop":
                    self.data_module = None
                else:
                    if self.data_module is None:
                        self.set_random_input_data(m)
                    else:
                        self.set_input_test_data_module(m)
            else:
                m.input_vector = copy.deepcopy(submodels[-1].output_vector)

            hardware_lock = self.HARDWARE_LOCKS[m.delegate]

            with hardware_lock:
                m = self.layer_deploy(m)

            submodels.append(m)
            log.info("Model %s | Layer %s deployed on %s", model_index, j, m.hardware)

        return submodels


    def multi_model_deploy(self):
        """
        Missing  Docstring: TODO
        """

        self.split()

        # Create a process pool with the number of processes equal to the number of CPU cores
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            # Asynchronously apply `DeploySequences` function to each sequence
            async_results = [
                pool.apply_async(self.sequences_deploy, (model_sequences, i))
                for i, model_sequences in enumerate(self.multi_models_sequences)
            ]

            # Collect the results as they complete
            self.models = [async_result.get() for async_result in async_results]

        log.info("All models deployed!")

        self.analyse_results()


def get_arguments():
    """
    Missing  Docstring: TODO
    """

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

    parser.add_argument(
            "-s",
            "--summarypath",
            default="resources/model_summaries/example_summaries/\
                MNIST/MNIST_full_quanitization_summary_w_mappings.json",
            help="File that contains a model summary with mapping annotations"
            )

    parser.add_argument(
            "-d",
            "--dataset",
            default="utils.datasets.MNIST",
            help="Dataset import module to be used for providing input data"
            )

    parser.add_argument(
            "-p",
            "--platform",
            default="desktop",
            help="Platform supporting the profiling/deployment process",
            )

    parser.add_argument(
            "-n",
            "--native",
            default=False,
            help="Native deployment on one single HW Target",
            )

    parser.add_argument(
            "-hw",
            "--hardware",
            default="cpu0",
            help="HW Target for native deployment",
            )

    return parser.parse_args()


if __name__ == "__main__":

    args = get_arguments()

    if verifier.verify_args_for_deployer(verifier, args=args) is True:
        deployer = Deployer(args.native, args.dataset, args.hardware)

    try:
        deployer.multi_model_deploy()
    except RuntimeError:
        log.error("An error occured in the deployment process.")
        TRACE_STR = traceback.format_exc()
        log.info("Traceback to the Excepton: %s", TRACE_STR)

    log.info("[DEPLOY] Finished")
