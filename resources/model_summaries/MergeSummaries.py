import os
import json
import argparse

from utils.splitter.utils import ReadJSON
from utils.logging.logger import log

def MergeSummaries(sum_dir, output_file):
    merged_data = {"models": []}
    
    for sum_file in sum_dir:
        data = ReadJSON(sum_file)
        model = data.get("models", [])
        merged_data["models"].extend(model)
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    log.info(f"Merged Summary saved to {output_file}")


def GetArgs() -> argparse.Namespace:
    """Argument parser, returns the Namespace containing all of the arguments.
    :raises: None

    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

    parser.add_argument(
            "-s",
            "--summariesdir",
            default="resources/artifacts/models_summaries",
            type=str,
            help="Workload directory path where multiple single model summaries are saved",
            )

    parser.add_argument(
            "-o",
            "--outputdir",
            default="resources/workloads",
            type=str,
            help="Output directory path of the merged multi model summary",
            )
    
    parser.add_argument(
            "-w",
            "--workload",
            default="MNIST",
            type=str,
            help="Output directory path of the merged multi model summary",
            )
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = GetArgs()
    
    output_file = os.path.join(args.outputdir, args.workload ,"{}_multi_model_summary.json".format(args.workload))
    
    MergeSummaries(args.summariesdir, output_file)