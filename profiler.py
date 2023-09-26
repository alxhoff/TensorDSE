import os
import sys
import argparse
from utils.logging.logger import log

MODELS_FOLDER = "resources/models/source/"
LAYERS_FOLDER = "resources/models/layers/"
COMPILED_MODELS_FOLDER = "resources/models/compiled/"
RESULTS_FOLDER = "resources/profiling_results/"


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

    system(command)


def ProfileModel(
        model_path: str,
        count: int,
        hardware_summary_path: str,
        model_summary_path: str,
        platform: str,
        usbmon: int,
        ) -> None:

    from utils.benchmark import ProfileModelLayers
    from utils.analysis import AnalyzeModelResults

    from utils.splitter.utils import ReadJSON
    from utils.splitter.split import Splitter

    if not model_path.endswith(".tflite"):
        raise Exception(f"File: {model_path} is not a tflite file!")

    model_name = (model_path.split("/")[-1]).split(".tflite")[0]
    log.info(f"Benchmarking {model_name} for {count} time(s)")

    hardware_to_benchmark = ["cpu", "gpu", "tpu"]

    hardware_summary_json = None

    if hardware_summary_path is not None:
        hardware_summary_json = ReadJSON(hardware_summary_path)

        if hardware_summary_json is not None:

            req_hardware = []
            if int(hardware_summary_json["CPU_count"]) > 0:
                req_hardware.append("cpu")

            if int(hardware_summary_json["GPU_count"]) > 0:
                req_hardware.append("gpu")

            if int(hardware_summary_json["TPU_count"]) > 0:
                req_hardware.append("tpu")

            hardware_to_benchmark = req_hardware
        else:
            log.error("Could not read provided hardware summary")
            sys.exit(-1)
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

    log.info("[PROFILE MODEL] Splitter created")

    if "tpu" in hardware_to_benchmark:
        # Compiles created models/layers into Coral models for execution
        splitter.CompileForEdgeTPU()
        log.info("[PROFILE MODEL] Models successfully compiled!")

    # Deploy the generated models/layers onto the target test hardware using docker

    results_dict = ProfileModelLayers(
            parent_model=model_name,
            hardware_list=hardware_to_benchmark,
            model_summary=model_summary_json,
            count=count,
            platform=platform,
            usbmon_bus=usbmon
            )

    log.info("Models deployed")

    # Process results
    print("[PROFILE MODEL] Analyzing model: {}".format(model_name))
    AnalyzeModelResults(model_name, results_dict, hardware_summary_json)

    log.info("Analyzed and merged results")

    log.info("Final Clean up")
    splitter.Clean(True)


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
            help="USB bus on which TPU is attached and thus which usbmon interface should be used for packet sniffing"
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

    log.info("[PROFILER] Starting")

    SummarizeModel(args.model, args.summaryoutputdir, args.summaryoutputname)

    log.info("[PROFILER] Model {} summarized".format(args.model))
    print("[PROFILER] Model summarized")

    if args.count < 2:
        print("Count MUST be greater than 2")
        sys.exit('Count was not greater than 2')

    ProfileModel(
        args.model,
        args.count,
        args.hardwaresummary,
        os.path.join(args.summaryoutputdir, "{}.json".format(args.summaryoutputname)),
        args.platform,
        args.usbmon,
    )

    log.info("[PROFILER] Model {} profiled".format(args.model))
    print("[PROFILER] Finished")