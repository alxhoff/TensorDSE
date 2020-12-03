source_model_filename = "models/source_model/MNIST_model.tflite"

def tflite_model_optimization():

    #log.info("Patching the model in JSON")
    fn_json = str(Path(source_model_filename).with_suffix(".json"))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fn_json = os.path.join(dir_path, fn_json)

    with open(fn_json) as fin:
        model = json.load(fin)

if __name__ == '__main__':

    import os
    import logging
    from pathlib import Path
    import subprocess
    import json
    import urllib.request
    from enum import Enum

    tflite_model_optimization()

    pass