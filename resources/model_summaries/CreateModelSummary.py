#!/bin/python
import sys
import argparse
from utils.logging.logger import log
from utils.splitter.utils import ReadJSON
    
def GetArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pass in the model file to be summarized")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="../models/examplemodels/MNIST_full_quanitization.tflite",
        help="Path to the TFLite model that is to be loaded and summarized",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        default="resources/model_summaries/example_summaries/MNIST",
        help="Output directory for JSON file, defults to modelsummaries folder in java resources",
    )
    parser.add_argument(
        "--outputname",
        type=str,
        required=False,
        default="MNIST_full_quanitization",
        help="Filename of output JSON file",
    )

    args = parser.parse_args()

    return args


def main() -> int:

    from summarize import SummarizeGraph

    args = GetArgs()

    log.info(
        "Converting {0} into {1}/{2}.json...".format(args.model, args.outputdir, args.outputname)
    )

    SummarizeGraph(args.model, args.outputdir, args.outputname)


    log.info("Graph summarized")

    return 0


if __name__ == "__main__":
    sys.exit(main())
