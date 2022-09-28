#!/bin/python
import sys
import argparse

parser = argparse.ArgumentParser(description="Creates a architecture summary from prompted information")
parser.add_argument("--outputdir", type=str, default="src/main/resources/architecturesummaries",
                    help="Output directory for JSON file, defults to architecturesummaries folder in java resources")
parser.add_argument("--output", type=str, default="outputarchitecturesummary.json",
                    help="Filename of output JSON file")

args = parser.parse_args()

def main() -> int:
    import json

    CPU_cores = input("How many CPU cores are available for use?\n")
    GPU_count = input("How many GPUs are available for use?\n")
    TPU_count = input("How many USB TPUs are available for use?\n")

    summary = {"CPU_cores": CPU_cores, "GPU_count": GPU_count, "TPU_count": TPU_count}

    with open(args.outputdir + "/" + args.output, "w+") as outfile:
        json.dump(summary, outfile, indent=4)

if __name__ == "__main__":
    sys.exit(main())