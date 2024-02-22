"""
Missing Docstring: TODO
"""

import os
from typing import Dict
from itertools import groupby

import json

from utils.logging.logger import log
from utils.splitter.utils import run_command_and_echo
from utils.usb.process import process_streams
from utils.analysis.analysis import Analyzer


def export_results(file:str, data:Dict) -> None:
    """
    Missing Docstring: TODO
    """
    with open(file, "w", encoding="utf-8") as json_file:
        json_data = data
        json.dump(json_data, json_file, indent=4)

def analyse_model_results(parent_model:str, models_dict:Dict, hardware_summary:Dict, platform: str):
    """
    Missing Docstring: TODO
    """

    results_folder = os.path.join(os.getcwd(), f"resources/profiling_results/{platform}")
    if os.path.exists(results_folder):
        run_command_and_echo("rm", "-rf", results_folder)
    os.mkdir(results_folder)
    results_path = os.path.join(results_folder, f"{parent_model}.json")
    log.info("Results file: %s from %s", results_path, os.getcwd())

    data = {
            "models": [{
                "name"      : parent_model,
                "runs"      : models_dict["count"],
                "layers"    : []
                }]
            }

    # layers
    delegates = ["cpu", "gpu", "tpu"]
    models_dict = {
        l: {
            m.delegate: m for m in list(v)
            } for l, v in groupby(
                sorted([model for d in delegates for model in list(models_dict[d])],
                        key=lambda x:x.model_name), lambda x:x.model_name)
        }

    for layer_name, layer in models_dict.items():

        layer_dict = {
                "name"                  : layer_name,
                "path"                  : {},
                "delegates"             : []
                }

        for delegate_name, delegate in layer.items():

            a = Analyzer(delegate.results, find_distribution=True)

            delegate_dict = {
                    "device"                   : delegate_name,
                    "count"                    : hardware_summary[f"{delegate_name.upper()}_count"],
                    "input"                    : {
                        "shape"   : delegate.input_shape,
                        "type"    : delegate.input_datatype
                        },
                    "mean"                     : a.mean,
                    "median"                   : a.median,
                    "standard_deviation"       : a.std_deviation,
                    "avg_absolute_deviation"   : a.avg_absolute_deviation,
                    "distribution"             : a.distribution_name,
                    "usb"                      : process_streams(delegate.timers, delegate.results),
                    "mean_computation"         : 0
                    }

            if delegate == "tpu":
                delegate_dict["mean"] = delegate_dict["mean"] - (delegate_dict["usb"]["communication"]["mean"])

            layer_dict["delegates"].append(delegate_dict)
            layer_dict["path"][delegate_name] = delegate.model_path

        for delegate in list(set(delegates) - set(layer.keys())):

            delegate_dict = {
                    "device"                    : delegate,
                    "count"                     : 0,
                    }
            layer_dict["delegates"].append(delegate_dict)

        data["models"][0]["layers"].append(layer_dict)

    export_results(results_path, data)
