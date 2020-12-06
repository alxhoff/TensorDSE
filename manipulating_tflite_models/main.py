import os
import json
import logging
from pathlib import Path

import utils

models_dir = "./models/source_model/"

def check_dir(models_dir: str):
    for file in os.listdir(models_dir):
        if file.endswith(".tflite"):
            source_model_filename = file
            source_model_filepath = os.path.join(models_dir, file)
    
    return source_model_filename,source_model_filepath

def tflite_model_optimization(log: logging.Logger, source_model_filepath: str):

    log.info("Patching the original model in JSON...")
    source_model_filepath_json = str(Path(source_model_filepath).with_suffix(".json"))

    with open(source_model_filepath_json) as fin:
        original_model = json.load(fin)
    
    log.info("Creating tmp files...")
    utils.echo_run("mkdir", "models/submodels/tmp_folder")
    if not Path("models/submodels/tmp_folder").exists():
        print("fuck")
    else:
        print("yes")
        utils.echo_run("rm", "-rf", "models/submodels/tmp_folder")

if __name__ == '__main__':
    
    source_model_filename,source_model_filepath = check_dir(models_dir)

    log = logging.getLogger(source_model_filename)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    tflite_model_optimization(log, source_model_filepath)

    pass