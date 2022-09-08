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
                "layers"    : []
        }]
    }

    if not isdir(RESULTS_FOLDER):
       import sys
       log.error(f"{RESULTS_FOLDER} is not a valid folder to store results!")
       sys.exit(-1)

    for delegate in ("cpu", "gpu", "tpu"):
        if not models_dict[delegate]:
            log.warning(f"Models dictionary does not contain results for delegate: {delegate}")
            continue

        for m in models_dict[delegate]:
            log.info(f"Analyzing results of operation: {m.model_name} ran on {delegate}")
            a = Analyzer(m)
            a.get_basic_statistics()
            a.get_distribution()

            model = data["models"][0] # hacky for now
            names = [i["name"] for i in model["layers"]]

            if not m.model_name in names:
                d = {
                    "device"                : m.model_name,
                    "path"                  : { m.delegate : m.model_path },
                    "delegates"             : [
                        {
                            "name"                      : m.delegate,
                            "mean"                      : a.mean,
                            "median"                    : a.median,
                            "standard_deviation"        : a.std_deviation,
                            "avg_absolute_deviation"    : a.avg_absolute_deviation,
                            "distribution"              : a.distribution_name
                        }
                    ]
                }

                if m.delegate == "tpu":
                    d["delegates"][m.delegate]["usb"] = m.usb_statistics

                model["layers"].append(d)
                data["models"][0] = model
                continue

            model_dict  = [ d for d in model["layers"] if d["name"] == m.model_name][0]
            delegates   = [ d["name"] for d in model_dict["delegates"]]
            paths       = [ d["path"] for d in model_dict["delegates"]]
            if not m.delegate in delegates:
                if m.delegate not in paths:
                    for i,j in enumerate(model["layers"]):
                        if j["name"] == m.model_name:
                            model["layers"][i]["path"][m.delegate] = m.model_path
                            break
                d = {
                            "name"                      : m.model_name,
                            "mean"                      : a.mean,
                            "median"                    : a.median,
                            "standard_deviation"        : a.std_deviation,
                            "avg_absolute_deviation"    : a.avg_absolute_deviation,
                            "distribution"              : a.distribution_name
                }

                if m.delegate == "tpu":
                    d["usb"] = m.usb_statistics

                for i,j in enumerate(model["layers"]):
                    if j["name"] == m.model_name:
                        model["layers"][i]["delegates"].append(d)
                        data["models"][0] = model
                        break
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

