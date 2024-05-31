"""
Missing Docstring: TODO
"""

import os
import sys
from typing import Dict
from itertools import groupby


import json
import numpy as np

import matplotlib.pyplot as plt

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


def analyse_model_results_profile(parent_model:str,
                                  models_dict:Dict, hardware_summary:Dict, platform: str):
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
                    "cpp_preprocessing"        : delegate.cpp_times[0],
                    "cpp_postprocessing"       : delegate.cpp_times[1],
                    "backend_inference_time"   : delegate.total_inf_time,
                    "mean_computation"         : 0
                    }

            if delegate == "tpu":
                delegate_dict["mean"] = delegate_dict["mean"] - (
                    delegate_dict["usb"]["communication"]["mean"]
                    )

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


def softmax(x):
    """
    Missing  Docstring: TODO
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def interpret_output(output_data, data_type, labels_np, index):
    """
    Missing  Docstring: TODO
    """

    result = False
    output_data = output_data.astype(np.float32)
    if data_type == np.int8:
        output_data = (output_data + 128) / 255.0  # Dequantize int8
    elif data_type == np.uint8:
        output_data = output_data / 255.0  # Dequantize uint8

    probabilities = softmax(output_data)
    predicted_label = np.argmax(probabilities)

    if predicted_label == labels_np[index]:
        result = True
    return result, predicted_label


def save_image(folder, image, predicted_label, image_index, platform, prediction):
    """
    Missing  Docstring: TODO
    """

    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')

    res = "right" if prediction else "wrong"

    file_name = f'image_{image_index}_predicted_{res}_{predicted_label}.png'
    file_path = os.path.join(folder, file_name)
    plt.savefig(file_path)
    log.info('Saved Image under: %s', file_name)


def analyse_model_results_deploy(models, platform):
    
    """
    Missing  Docstring: TODO
    """

    results_folder = os.path.join(os.getcwd(), "resources/deployment_results", platform)
    models_len = len(models)
    results_folder_multi = os.path.join(results_folder, f"multi_{models_len}")

    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
        if not os.path.isdir(results_folder):
            log.fatal("Unable to create Results Folder to store Deployment results!")
            sys.exit(-1)

    if not os.path.isdir(results_folder_multi):
        os.mkdir(results_folder_multi)
        if not os.path.isdir(results_folder_multi):
            log.fatal("Unable to create Results Folder to store Deployment results!")
            sys.exit(-1)


    inference_results = dict()
    input_image_idx = 0

    for i, model in enumerate(models):
        results_path = os.path.join(results_folder_multi, f"model_{i}_{model[0].parent}.json")
        model_results = {
            "model_name": model[0].parent,
            "submodels": [],
            "total_inference_time (c++)": 0,
            "total_pre_processing_time (c++)": 0,
            "total_post_processing_time (c++)": 0,
            "total_inference_time (python)": 0,
            "Prediction Result": "F"
        }

        for m in model:
            submodel = {
                "name": m.model_name,
                "layers": m.details,
                "inference_time (c++)": m.results[0],
                "pre_processing_time (c++)": m.cpp_times[0],
                "post_processing_time (c++)": m.cpp_times[1],
                "inference_time (python)": m.total_inf_time,
                "usb": {}
            }

            if platform != "coral" and "tpu" in m.delegate:
                submodel["usb"] = process_streams(m.timers, m.results)

            model_results["submodels"].append(submodel)

            model_results["total_inference_time (c++)"] += m.results[0]
            model_results["total_pre_processing_time (c++)"] += m.cpp_times[0]
            model_results["total_post_processing_time (c++)"] += m.cpp_times[1]
            model_results["total_inference_time (python)"] += m.total_inf_time

            if platform == "desktop":
                if m.first is True:
                    input_image_idx = m.input_image_idx

                if m.final is True:
                    prediction, predicted_label = interpret_output(m.output_vector,
                                        m.get_np_dtype(m.output_datatype),
                                        m.labels, input_image_idx)
                    if prediction is True:
                        log.info("Image Prediction Result: Positive")
                        inference_results[f"{i}"] = 1
                    else:
                        log.info("Image Prediction Result: Negative")
                        inference_results[f"{i}"] = -1

                    save_image(results_folder_multi, m.images[input_image_idx],
                                predicted_label, input_image_idx,
                                platform, prediction)
        if platform == "desktop":
            model_results["Prediction Result"] = "T" if prediction else "F"
        with open(results_path, 'w', encoding="utf-8") as json_file:
            json.dump(model_results, json_file, indent=4)
    return inference_results
