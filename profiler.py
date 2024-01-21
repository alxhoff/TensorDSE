"""
Missing  Docstring: TODO
"""

import os
import argparse
from typing import Dict

from utils.logging.logger import log

from utils.model import Model
from utils.splitter.split import Splitter
from utils.analysis import AnalyzeModelResults
from utils.inference import Inference
from status_verifier import StatusVerifierSingleton as verifier

class Proflier:
    """
    Missing  Docstring: TODO
    """

    def __init__(self,
                 count: int,
                 usbmon: int) -> None:

        self.count = count
        self.usbmon = usbmon
        self.model_summary = verifier.get_model_summary(verifier)
        self.hardware_summary = verifier.get_hardware_summary(verifier)
        self.platform = verifier.get_platform(verifier)

        self.splitter = Splitter(self.model_summary)

        self.hardware_to_benchmark = []


    def single_model_profile(self, model_summary: dict, parent_model_path: str) -> Dict:
        """
        Missing  Docstring: TODO
        """

        models = {}
        models["count"] = self.count

        for delegate in self.hardware_to_benchmark:
            models[delegate] = []
            for layer in model_summary["layers"]:
                inference_instance = Inference(
                    model=Model(layer, delegate, parent_model_path),
                    count=self.count,
                    platform=self.platform
                    )
                m = inference_instance.invoke()
                m.model_name = f"{m.model_name}_{m.index}_{'_'.join(map(str, m.input_shape))}"
                models["cpu"].append(m)
                log.info("[PROFILE MODEL LAYERS] TPUs profiled")
                #AnalyzeLayerResults(m, "delegate")

        return models


    def multi_model_profile(self):
        """
        Missing  Docstring: TODO
        """

        if (verifier.verify_hardware_for_profiling(verifier,
                                                   hardware_summary=self.hardware_summary)):
            self.hardware_to_benchmark = verifier.get_requested_hardware(verifier)
        else:
            log.fatal("The list for the Requested Hardware for profiling is empty!")

        # Create single operation models/layers from the operations in the provided model
        if self.platform == "desktop":
            try:
                log.info("Running Model Splitter ...")
                self.splitter.Run()
                log.info("Splitting Process Complete!\n")
            except RuntimeError:
                self.splitter.Clean(True)
                log.fatal("Failed to run splitter!")

            if "tpu" in self.hardware_to_benchmark:
                # Compiles created models/layers into Coral models for execution
                self.splitter.CompileForEdgeTPU()
                log.info("[PROFILE MODEL] Models successfully compiled \
                         for Benchmarking on the Edge TPU!")

        # Deploy the generated models/layers onto the target test hardware using docker
        for single_model_summary in self.model_summary.get("models"):
            if not single_model_summary.get("path").endswith(".tflite"):
                log.fatal("File: %s is not a tflite file!", single_model_summary.get("path"))

            model_path = os.path.basename(single_model_summary.get("path")).split(".tflite")[0]
            log.info("Benchmarking %s for %s time(s)", model_path, self.count)

            results_dict = self.single_model_profile(
                    parent_model_path=model_path,
                    model_summary=single_model_summary
                    )

            log.info("Model %s profiled!", model_path)

            # Process results
            log.info("[PROFILE MODEL] Analyzing profiling results for: %s", model_path)
            AnalyzeModelResults(model_path, results_dict, self.hardware_summary, self.platform)

            log.info("Analyzed and merged results")

            log.info("Final Clean up")
            if self.platform == "desktop":
                self.splitter.Clean(True)


def get_arguments() -> argparse.Namespace:
    """
    Argument parser, returns the Namespace containing all of the arguments.
    :raises: None

    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

    parser.add_argument(
            "-s",
            "--summarypath",
            default="resources/model_summaries/example/summaries/MNIST",
            help="Directory where model summary should be saved",
            )

    parser.add_argument(
            "-c",
            "--count",
            type=int,
            default=2,
            help="Number of times to measure inference.",
            )

    parser.add_argument(
            "-hs",
            "--hardwaresummary",
            type=str,
            default="resources/architecture_summaries/example_output_architecture_summary.json",
            help="Hardware summary file to tell benchmarking which devices to benchmark, by default\
                  all devices will be benchmarked",
            )

    parser.add_argument(
            "-p",
            "--platform",
            default="desktop",
            help="Platform supporting the profiling/deployment process",
            )

    parser.add_argument(
            "-u",
            "--usbmon",
            default="0",
            help="USB bus on which TPU is attached and thus which usbmon interface\
                  should be used for packet sniffing"
            )

    return parser.parse_args()


if __name__ == "__main__":

    args = get_arguments()

    if verifier.verify_args_for_profiler(verifier, args=args) is True:
        profiler = Proflier(
            count=args.count,
            usbmon=args.usbmon
        )
        profiler.multi_model_profile()
