import argparse
from typing import Dict

from utils.model import Model
from utils.analysis import Analyzer

def AnalyzeModelResults(parent_model:str, models:Dict):
    import json
    from os.path import join, isdir
    from main import log, RESULTS_FOLDER

    data = {
            parent_model: {
                "name"   : parent_model,
                "layers" : {}
        }
    }

    for delegate in ("cpu", "gpu", "tpu"):
        if not isdir(RESULTS_FOLDER):
            log.error(f"{RESULTS_FOLDER} is not a valid folder to store results!")
            break

        if not models[delegate]:
            log.info(f"Models dictionary does not contain results for delegate: {delegate}")
            continue

        for m in models[delegate]:
            log.info(f"Analyzing results of operation: {m.model_name} ran on {delegate}")
            a = Analyzer(m)
            a.get_basic_statistics()
            a.get_distribution()

            if not m.model_name in  data[parent_model]["layers"].keys():
                l = {
                    "name"                  : m.model_name,
                    "path"                  : m.model_path,
                    "delegates"             : {
                        m.delegate: {
                            "mean time"             : a.mean,
                            "median"                : a.median,
                            "standard deviation "   : a.std_deviation,
                            "distribution"          : a.distribution_name
                        }
                    }
                }

                data[parent_model]["layers"][m.model_name] = l
                continue

            if not m.delegate in  data[parent_model]["layers"][m.model_name]["delegates"].keys():
                data[parent_model]["layers"][m.model_name]["delegates"][m.delegate] = {
                            "mean time"             : a.mean,
                            "median"                : a.median,
                            "standard deviation "   : a.std_deviation,
                            "distribution"          : a.distribution_name
                }
                continue

            raise Exception(f"Apparently attempt to overwrite data from model: {m.model_name} run on: {delegate}!")


    with open(join(RESULTS_FOLDER, f"results_{parent_model}.json"), "w") as json_file:
        json_data = json.dumps(data)
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

