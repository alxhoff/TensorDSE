#!/bin/python
import sys
import argparse

parser = argparse.ArgumentParser(description="Pass in the model file to be summarized")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    default="../examplemodels/MNIST.tflite",
    help="Path to the TFLite model that is to be loaded and summarized",
)
parser.add_argument(
    "--outputdir",
    type=str,
    default=".",
    help="Output directory for JSON file, defults to modelsummaries folder in java resources",
)
parser.add_argument(
    "--outputname",
    type=str,
    required=True,
    default="output",
    help="Filename of output JSON file",
)

args = parser.parse_args()

def main() -> int:
    from summarize import SummarizeGraph

    print(args.model)
    print(args.outputdir)
    print(args.outputname)

    SummarizeGraph(args.model, args.outputdir, args.outputname)

    return 0

if __name__ == "__main__":
    sys.exit(main())
