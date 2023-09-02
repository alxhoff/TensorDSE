from typing import Dict, List

from utils.model import Model

def ExportResults(file:str, data:Dict) -> None:
    import json
    with open(file, "w") as json_file:
        json_data = data
        json.dump(json_data, json_file, indent=4)

def AnalyzeModelResults(parent_model:str, models_dict:Dict, hardware_summary:Dict):

    from utils.splitter.logger import log
    from utils.usb.process import process_streams
    from utils.analysis.analysis import Analyzer
    import os
    
    RESULTS_FOLDER = os.path.join(os.getcwd(), "resources/profiling_results")
    results_path = os.path.join(RESULTS_FOLDER, f"{parent_model}.json")
    print("Results file: {} from {}".format(results_path, os.getcwd()))

    if not os.path.isdir(RESULTS_FOLDER):
        import sys
        print("Results folder {} doesn't exist, creating".format(RESULTS_FOLDER))
        log.error(f"{RESULTS_FOLDER} is not a valid folder to store results!")
        os.mkdir(RESULTS_FOLDER)
        if not os.path.isdir(RESULTS_FOLDER):
            sys.exit(-1)
    
    data = {
            "models": [{
                "name"      : parent_model,
                "runs"      : models_dict["count"],
                "layers"    : []
        }]
    }

    # layers
    from itertools import groupby
    delegates = ["cpu", "gpu", "tpu"]
    models_dict = {l: {m.delegate: m for m in list(v)} for l, v in groupby(sorted([model for d in delegates for model in list(models_dict[d])], key=lambda x:x.model_name), lambda x:x.model_name)}

    for layer_name, layer in models_dict.items():

        layer_dict = {
            "name"                  : layer_name,
            "path"                  : {},
            "delegates"             : []
            }

        for delegate_name, delegate in layer.items():

            a = Analyzer(delegate.results, find_distribution=True)

            delegate_dict = {
                "device"                      : delegate_name,
                "count"                       : hardware_summary["{}_count".format(delegate_name.upper())],
                "input"                     : {
                        "shape"   : delegate.input_shape,
                        "type"    : delegate.input_datatype
                    },
                "mean"                      : a.mean,
                "median"                    : a.median,
                "standard_deviation"        : a.std_deviation,
                "avg_absolute_deviation"    : a.avg_absolute_deviation,
                "distribution"              : a.distribution_name,
                "usb"                       : process_streams(delegate.timers, delegate.results)         
            }

            layer_dict["delegates"].append(delegate_dict)
            layer_dict["path"][delegate_name] = delegate.model_path

        for delegate in list(set(delegates) - set(layer.keys())):

            delegate_dict = {
                "device"                    : delegate,
                "count"                     : 0,
            }
            layer_dict["delegates"].append(delegate_dict)

        data["models"][0]["layers"].append(layer_dict)

    print("Exporting profiling results to: {}".format(results_path))
    ExportResults(results_path, data)

def MergeResults(parent_model:str, layers:dict, clean:bool=True):
    from main import RESULTS_FOLDER
    from ..splitter.logger import log
    from utils import load_json
    import os

    data = load_json(os.path.join(RESULTS_FOLDER, f"{parent_model}.json"))
    d = data["models"][0]["layers"]
    names = [i["name"] for i in d]

    for device in ("cpu", "gpu", "tpu"):
        for l in layers[device]:
            file = os.path.join(RESULTS_FOLDER, f"layer_{l.index}_{l.model_name}_{device.upper()}_USB.json")
            name = f"{l.model_name}_{l.index}"
            if (os.path.isfile(file) and name in names):
                    j = load_json(file)
                    for k,v in enumerate(d):
                        delegates = [i["device"] for i in v["delegates"]]
                        if (v["name"] == name):
                            d[k]["path"][device] = j["path"][device]
                            if device in delegates:
                                for x,dele in enumerate(d[k]["delegates"]):
                                    if dele["device"] == device:
                                        d[k]["delegates"][x] = j["delegates"][0]
                                        break
                            else:
                                d[k]["delegates"].append(j["delegates"][0])
                            print(f"Merging {name} run on {device} onto the results from {parent_model}!")

                    if clean:
                        os.system(f"rm -f {file}")


    data["models"][0]["layers"] = d
    ExportResults(os.path.join(RESULTS_FOLDER, f"{parent_model}.json"), data)
