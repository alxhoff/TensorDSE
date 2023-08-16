import os
import sys
import argparse
from utils.log import Log

MODELS_FOLDER = "resources/models/source/"
LAYERS_FOLDER = "resources/models/layers/"
COMPILED_MODELS_FOLDER = "resources/models/compiled/"
RESULTS_FOLDER = "resources/profiling_results/"

# custom logger to separate TF logs and Ours
# log = Log(os.path.join(RESULTS_FOLDER, "JOURNAL.log"))


def DisableTFlogging() -> None:
    """Disable the most annoying logging known to mankind"""
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"  # only report errors
    os.environ["KMP_WARNINGS"] = "0"  # disable warnings


def SummarizeModel(model: str, output_dir: str, output_name: str) -> None:
    from os import system

    command = "python3 resources/model_summaries/CreateModelSummary.py --model {} --outputdir {} --outputname {}".format(
        model, output_dir, output_name
    )

    print("Running summary command: {}".format(command))
    system(command)


def ProfileModel(
    model_path: str, count: int, hardware_summary_path: str, model_summary_path: str
) -> None:
    from utils.benchmark import BenchmarkModelLayers
    from utils.analysis import AnalyzeModelResults, MergeResults

    from utils.splitter.utils import ReadJSON
    from utils.splitter.logger import log
    from utils.splitter.split import Splitter

    if not model_path.endswith(".tflite"):
        raise Exception(f"File: {model_path} is not a tflite file!")

    model_name = (model_path.split("/")[-1]).split(".tflite")[0]
    log.info(f"Benchmarking {model_name} for {count} time(s)")

    hardware_to_benchmark = ["cpu", "gpu", "tpu"]

    if hardware_summary_path is not None:
        hardware_summary_json = ReadJSON(hardware_summary_path)

        if hardware_summary_json is not None:
            req_hardware = []

            if int(hardware_summary_json["CPU_cores"]) > 0:
                req_hardware.append("cpu")

            if int(hardware_summary_json["GPU_count"]) > 0:
                req_hardware.append("gpu")

            if int(hardware_summary_json["TPU_count"]) > 0:
                req_hardware.append("tpu")

            hardware_to_benchmark = req_hardware
        else:
            log.error("The provided Hardware Summary is empty!")
            sys.exit(-1)

    if model_summary_path is not None:
        model_summary_json = ReadJSON(model_summary_path)
        if model_summary_json is None:
            log.error("The provided Model Summary is empty!")
            sys.exit(-1)

    # Create single operation models/layers from the operations in the provided model
    splitter = Splitter(model_path, model_summary_json)
    try:
        log.info("Running Model Splitter ...")
        splitter.Run()
        log.info("Splitting Process Complete!\n")
    except Exception as e:
        splitter.Clean(True)
        log.error("Failed to run splitter! {}".format(str(e)))

    if "tpu" in hardware_to_benchmark:
        # Compiles created models/layers into Coral models for execution
        splitter.CompileForEdgeTPU()
        log.info("Models successfully compiled!")

    # Deploy the generated models/layers onto the target test hardware using docker
    results_dict = BenchmarkModelLayers(
        parent_model=model_name,
        hardware_list=hardware_to_benchmark,
        model_summary=model_summary_json,
        count=count,
    )

    log.info("Models deployed")

    # Process results
    AnalyzeModelResults(model_name, results_dict)

    log.info("Analyzed and merged results")


def GetArgs() -> argparse.Namespace:
    """Argument parser, returns the Namespace containing all of the arguments.
    :raises: None

    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-m",
        "--model",
        default="resources/models/example_models/MNIST_full_quanitization.tflite",
        help="File path to the SOURCE .tflite file.",
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
        help="Hardware summary file to tell benchmarking which devices to benchmark, by default all devices will be benchmarked",
    )

    parser.add_argument(
        "-o",
        "--summaryoutputdir",
        default="resources/model_summaries/example_summaries/MNIST",
        help="Directory where model summary should be saved",
    )

    parser.add_argument(
        "-n",
        "--summaryoutputname",
        default="MNIST_full_quanitization_summary",
        help="Name that the model summary should have",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    """Entry point to execute this script.

    Flags
    ---------
    -m or --model
        Target input tflite model to be processed and splitted.

    -c or --count
        Used in the tflite deployment that may occur directly after conversion.
        With count it is set the number of deployments done.
    """

    args = GetArgs()
    DisableTFlogging()

    SummarizeModel(args.model, args.summaryoutputdir, args.summaryoutputname)

    print("Profiling model")

    ProfileModel(
        args.model,
        args.count,
        args.hardwaresummary,
        os.path.join(args.summaryoutputdir, "{}.json".format(args.summaryoutputname)),
    )

    print()
    print("Finito ☜(⌒▽⌒)=b")
