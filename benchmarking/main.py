import argparse
from utils.log import Log

# custom logger to separate TF logs and Ours
log = Log("results/info.log")

MODELS_FOLDER           = "models/source/"
LAYERS_FOLDER           = "models/layers/"
COMPILED_MODELS_FOLDER  = "models/compiled/"
RESULTS_FOLDER          = "results/"

def DisableTFlogging() -> None:
    """Disable the most annoying logging known to mankind
    """
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' # only report errors
    os.environ['KMP_WARNINGS'] = '0'         # disable warnings


def BenchmarkModel(model:str, count:int):
    from deploy import DeployModels
    from convert import ImportTFLiteModules, SplitTFLiteModel
    from compile import CompileTFLiteModelsForCoral
    from analysis import AnalyzeModelResults

    if not model.endswith(".tflite"):
        raise Exception(f"File: {model} is not a tflite file!")

    model_name = (model.split("/")[model.count("/")]).split(".tflite")[0]
    log.info(f"Benchmarking {model_name} for {count} time(s)")

    # Imports modules found in the tflite folder, generated from the fattbuffer compiler
    ImportTFLiteModules()

    # Create single operation models/layers from the operations in the provided model
    SplitTFLiteModel(model=model)

    # Compiles created models/layers into Coral models for execution
    CompileTFLiteModelsForCoral()

    # Deploy the generated models/layers onto the target test hardware using docker
    ret = DeployModels(model_name, count=count)

    # Process results
    AnalyzeModelResults(model_name, ret)


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
        default="models/source/MNIST.tflite",
        help="File path to the SOURCE .tflite file.",
    )

    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=1000,
        help="Number of times to measure inference.",
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
    BenchmarkModel(args.model, args.count)
