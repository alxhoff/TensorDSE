import argparse
import os
from utils.log import Log

MODELS_FOLDER = "resources/models/source/"
LAYERS_FOLDER = "resources/models/layers/"
COMPILED_MODELS_FOLDER = "resources/models/compiled/"
RESULTS_FOLDER = "resources/results/"

# custom logger to separate TF logs and Ours
#log = Log(os.path.join(RESULTS_FOLDER, "JOURNAL.log"))


def DisableTFlogging() -> None:
    """Disable the most annoying logging known to mankind"""
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"  # only report errors
    os.environ["KMP_WARNINGS"] = "0"  # disable warnings


def SummarizeModel(model: str, output_dir: str, output_name: str) -> None:
    from os import system

    command = "python resources/model_summaries/CreateModelSummary.py --model {} --outputdir {} --outputname {}".format(
        model, output_dir, output_name
    )

    print("Running summary command: {}".format(command))
    system(command)


def BenchmarkModel(model: str, count: int, hardware_summary: str, model_summary: str) -> None:
    from utils.deploy import DeployModels
    from utils.convert import ImportTFLiteModules, SplitTFLiteModel
    from utils.compile import CompileTFLiteModelsForCoral
    from utils.analysis import AnalyzeModelResults, MergeResults

    from utils.model_lab.utils import ReadJSON, LayerDetails
    from utils.model_lab.logger import log
    from utils.model_lab.split import Splitter


    import json

    if not model.endswith(".tflite"):
        raise Exception(f"File: {model} is not a tflite file!")

    model_name = (model.split("/")[model.count("/")]).split(".tflite")[0]
    log.info(f"Benchmarking {model_name} for {count} time(s)")

    hardware_to_benchmark = ["cpu", "gpu", "tpu"]

    if hardware_summary is not None:
        hardware_summary_json = ReadJSON(hardware_summary)

        req_hardware = []

        if int(hardware_summary_json["CPU_cores"]) > 0:
            req_hardware.append("cpu")

        if int(hardware_summary_json["GPU_count"]) > 0:
            req_hardware.append("gpu")

        if int(hardware_summary_json["TPU_count"]) > 0:
            req_hardware.append("tpu")

        hardware_to_benchmark = req_hardware

    # Create single operation models/layers from the operations in the provided model
    splitter = Splitter(model, model_summary)
    try:
        log.info("Running Model Splitter ...")
        splitter.Run()
        log.info("Splitting Process Complete!\n")
    except Exception as e:
        splitter.Clean(True)
        log.error("Failed to run splitter! {}".format(str(e)))

    if "tpu" in hardware_to_benchmark:
        #Compiles created models/layers into Coral models for execution
        splitter.CompileForEdgeTPU()
        log.info("Models successfully compiled!")

    # Deploy the generated models/layers onto the target test hardware using docker
    results_dict = DeployModels(
        hardware_list=hardware_to_benchmark,
        model_summary=model_summary,
        count=count
    )

    #print("Models deployed")

    # Process results
    #AnalyzeModelResults(model_name, results_dict)

    #print("Analyzed results")

    #MergeResults(model_name, layers, clean=True)

    #print("Results merged")


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
        default="resources/example_models/kws_ref_model.tflite",
        help="File path to the SOURCE .tflite file.",
    )

    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=10,
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
        default="resources/model_summaries",
        help="Directory where model summary should be saved",
    )

    parser.add_argument(
        "-n",
        "--summaryoutputname",
        default="kws_ref_model",
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

    #SummarizeModel(args.model, args.summaryoutputdir, args.summaryoutputname)
    print("Model summarized")

    BenchmarkModel(args.model, args.count, args.hardwaresummary, os.path.join(args.summaryoutputdir, "{}.json".format(args.summaryoutputname)))
    print("Model benchmarked")

    ## Run DSE
    #import os
#
    os.chdir(os.path.join(os.getcwd(), "DSE/TensorDSE"))
    print(os.getcwd())
    model_summary = (
       "../../resources/model_summaries/example_summaries/MNIST_multi_1.json"
    )
    architecture_summary = "../../resources/architecture_summaries/example_output_architecture_summary.json"
    benchmarking_results = (
       "../../resources/benchmarking_results/example_benchmark_results.json"
    )
    output_folder = "src/main/resources/output"
    ilp_mapping = "true"
    runs = "1"
    crossover = "0.9"
    population_size = 100
    parents_per_generation = 50
    offspring_per_generation = 50
    generations = 25
    verbose = "false"
    os.system(
       'gradle6 run --args="--modelsummary {} --architecturesummary {} --benchmarkingresults {} --outputfolder {} --ilpmapping {} --runs {} --crossover {} --populationsize {} --parentspergeneration {} --offspringspergeneration {} --generations {} --verbose {}"'.format(
           model_summary,
           architecture_summary,
           benchmarking_results,
           output_folder,
           ilp_mapping,
           runs,
           crossover,
           population_size,
           parents_per_generation,
           offspring_per_generation,
           generations,
           verbose,
       )
    )
    os.chdir(os.path.join(os.getcwd(), "../.."))

    # Deploy

    print()
    print("Finito ☜(⌒▽⌒)=b")
