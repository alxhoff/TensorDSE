"""
Module Docstring: TODO
"""

import os
import json
import argparse

from utils.splitter.utils import read_json_file
from utils.logging.logger import log

def merge_summaries(sum_dir, output_file):
    """
    Merges multiple Summaries into one and saves it under a path of the user's choosing.
    """
    merged_data = {"models": []}
    for sum_file in sum_dir:
        data = read_json_file(sum_file)
        model = data.get("models", [])
        merged_data["models"].extend(model)

    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)
    log.info("Merged Summary saved to %s", output_file)


def get_arguments() -> argparse.Namespace:
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

    return parser.parse_args()


if __name__ == "__main__":

    args = get_arguments()
    output_file_path = os.path.join(
        args.outputdir,
        args.workload ,
        f"{args.workload}_multi_model_summary.json")

    merge_summaries(args.summariesdir, output_file_path)
