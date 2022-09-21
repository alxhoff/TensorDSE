from typing import Dict, List

from utils.model import Model

def ExportResults(file:str, data:Dict) -> None:
    import json
    with open(file, "w") as json_file:
        json_data = data
        json.dump(json_data, json_file, indent=4)

def AnalyzeModelResults(parent_model:str, models_dict:Dict):
    from main import log, RESULTS_FOLDER
    from utils import cpu_count
    from usb.process import process_timers
    from analysis.analysis import Analyzer
    import os

    data = {
            "models": [{
                "name"      : parent_model,
                "runs"      : models_dict["count"],
                "layers"    : []
        }]
    }

    if not os.path.isdir(RESULTS_FOLDER):
       import sys
       log.error(f"{RESULTS_FOLDER} is not a valid folder to store results!")
       sys.exit(-1)

    for delegate in ("cpu", "gpu", "tpu"):
        if not models_dict[delegate]:
            log.warning(f"Models dictionary does not contain results for delegate: {delegate}")
            continue

        for m in models_dict[delegate]:
            log.info(f"Analyzing results of operation: {m.model_name} ran on {delegate}")
            a = Analyzer(m.results, find_distribution=True)

            model = data["models"][0] # hacky for now
            names = [i["name"] for i in model["layers"]]

            if not m.model_name in names:
                d = {
                    "name"                  : m.model_name,
                    "path"                  : { m.delegate : m.model_path },
                    "delegates"             : [
                        {
                            "device"                      : m.delegate,
                            "input"                     : {
                                    "shape"   : m.input_shape,
                                    "type"    : m.input_datatype
                             },
                            "mean"                      : a.mean,
                            "median"                    : a.median,
                            "standard_deviation"        : a.std_deviation,
                            "avg_absolute_deviation"    : a.avg_absolute_deviation,
                            "distribution"              : a.distribution_name,
                            "usb"                       : process_timers(m.timers)
                        }
                    ]
                }

                if m.delegate == "cpu":
                    d["delegates"][0]["cpu_count"] = cpu_count()

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
                            "device"                    : m.delegate,
                            "input"                     : {
                                    "shape"   : m.input_shape,
                                    "type"    : m.input_datatype
                             },
                            "mean"                      : a.mean,
                            "median"                    : a.median,
                            "standard_deviation"        : a.std_deviation,
                            "avg_absolute_deviation"    : a.avg_absolute_deviation,
                            "distribution"              : a.distribution_name,
                            "usb"                       : process_timers(m.timers)
                }

                if m.delegate == "cpu":
                    d["cpu_count"] = cpu_count()

                for i,j in enumerate(model["layers"]):
                    if j["name"] == m.model_name:
                        model["layers"][i]["delegates"].append(d)
                        data["models"][0] = model
                        break
                continue

            raise Exception(f"Apparently attempt to overwrite data from model: {m.model_name} run on: {delegate}!")

    ExportResults(os.path.join(RESULTS_FOLDER, f"{parent_model}.json"), data)

def AnalyzeLayerResults(m:Model, delegate:str):
    from main import RESULTS_FOLDER
    from usb.process import process_timers
    from analysis.analysis import Analyzer
    import os

    a = Analyzer(m.results, find_distribution=True)
    data = {
        "name"                  : m.model_name,
        "path"                  : { m.delegate : m.model_path },
        "delegates"             : [
            {
                "device"                      : m.delegate,
                "input"                     : {
                        "shape"   : m.input_shape,
                        "type"    : m.input_datatype
                 },
                "mean"                      : a.mean,
                "median"                    : a.median,
                "standard_deviation"        : a.std_deviation,
                "avg_absolute_deviation"    : a.avg_absolute_deviation,
                "distribution"              : a.distribution_name,
                "usb"                       : process_timers(m.timers)
            }
        ]
    }

    ExportResults(os.path.join(RESULTS_FOLDER, f"{m.model_name}_LAYER.json"), data)

def MergeResults(parent_model:str, layers:List):
    from main import RESULTS_FOLDER, log
    from utils import load_json
    import os

    data = load_json(os.path.join(RESULTS_FOLDER, f"{parent_model}.json"))
    d = data["models"][0]["layers"]
    names = [i["name"] for i in d]

    for l in layers:
        if (os.path.isfile(os.path.join(RESULTS_FOLDER, f"{l.upper()}_LAYER.json"))
            and l.upper() in names):
                j = load_json(os.path.join(RESULTS_FOLDER, f"{l.upper()}_LAYER.json"))
                for k,v in enumerate(d):
                    if v["name"] == l.upper():
                        d[k]["path"]["tpu"] = j["path"]["tpu"]
                        d[k]["delegates"].append(j["delegates"][0])


    data["models"][0]["layers"] = d
    ExportResults(os.path.join(RESULTS_FOLDER, f"{parent_model}.json"), data)
