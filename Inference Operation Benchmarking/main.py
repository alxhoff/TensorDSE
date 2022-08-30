import argparse
import logging

log = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO, filename="debug.log")

MODELS_FOLDER = "models/layers/"
COMPILED_MODELS_FOLDER = "models/compiled/"

def BenchmarkModel(model:str, count:int):
    from deploy import DeployModels
    from convert import ImportTFLiteModules, SplitTFLiteModel
    from utils.compile import CompileTFLiteModelsForCoral
    from analysis.analyze import AnalyzeModelResults

    # Imports modules found in the tflite folder, generated from the fattbuffer compiler
    ImportTFLiteModules()

    # Create single operation models from the operations in the provided model
    SplitTFLiteModel(model=model)

    # Compiles created models into Coral models for execution
    CompileTFLiteModelsForCoral(log)

    # Deploy the generated models onto the target test hardware using docker
    DeployModels(count=count)

    # Process results
    AnalyzeModelResults()


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
    BenchmarkModel(args.model, args.count)
