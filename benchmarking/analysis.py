import argparse
from typing import Dict

from utils.model import Model
from utils.analysis import Analyzer

def AnalyzeModelResults(parent_model:str, models_dict:Dict):
    import json
    from os.path import join, isdir
    from main import log, RESULTS_FOLDER

    data = {
            "models": [{
                "name"      : parent_model,
                "runs"      : models_dict["count"],
                "layers"    : {}
        }]
    }

    for delegate in ("cpu", "gpu", "tpu"):
        if not isdir(RESULTS_FOLDER):
            log.error(f"{RESULTS_FOLDER} is not a valid folder to store results!")
            break

        if not models_dict[delegate]:
            log.info(f"Models dictionary does not contain results for delegate: {delegate}")
            continue

        for m in models_dict[delegate]:
            log.info(f"Analyzing results of operation: {m.model_name} ran on {delegate}")
            a = Analyzer(m)
            a.get_basic_statistics()
            a.get_distribution()

            model = data["models"][0] # hacky for now
            if not m.model_name in  model["layers"].keys():
                d = {
                    "name"                  : m.model_name,
                    "path"                  : { m.delegate : m.model_path },
                    "delegates"             : {
                        m.delegate: {
                            "mean"                      : a.mean,
                            "median"                    : a.median,
                            "standard_deviation "       : a.std_deviation,
                            "avg_absolute_deviation"    : a.avg_absolute_deviation,
                            "distribution"              : a.distribution_name
                        }
                    }
                }

                if m.delegate == "tpu":
                    d["delegates"][m.delegate]["usb"] = m.usb_statistics

                model["layers"][m.model_name] = d
                data["models"][0] = model
                continue

            if not m.delegate in  model["layers"][m.model_name]["delegates"].keys():
                if m.delegate not in model["layers"][m.model_name]["path"].keys():
                    model["layers"][m.model_name]["path"][m.delegate] = m.model_path

                d = {
                            "mean"                      : a.mean,
                            "median"                    : a.median,
                            "standard_deviation "       : a.std_deviation,
                            "avg_absolute_deviation"    : a.avg_absolute_deviation,
                            "distribution"              : a.distribution_name
                }

                if m.delegate == "tpu":
                    d["usb"] = m.usb_statistics

                model["layers"][m.model_name]["delegates"][m.delegate] = d
                data["models"][0] = model
                continue

            raise Exception(f"Apparently attempt to overwrite data from model: {m.model_name} run on: {delegate}!")


    with open(join(RESULTS_FOLDER, f"{parent_model}.json"), "w") as json_file:
        json_data = data
        json.dump(json_data, json_file, indent=4)


def GetArgs() -> argparse.Namespace:
    """ Description
    :raises:

    :rtype:
    """
    from main import RESULTS_FOLDER
    from os.path import join

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--file', required=False,
                        default="results/results.json",
                        help='Path to file .')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = GetArgs()

