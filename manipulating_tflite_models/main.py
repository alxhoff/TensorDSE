import os
import json
import logging
from pathlib import Path
import urllib.request

import utils

FROM_CONTAINER = "from_container"
TO_CONTAINER   = "to_container"
CONTAINER_SUBMODELS_PATH = "/home/deb/TensorDSE/submodels"

mapping = [[0,2],[1,2],[2,2],[3,0],[4,2],[5,2],[6,2]]

def tflite_model_optimization(log: logging.Logger, main_dir_path: str):
    
    log.info("Checking submodels directory...")
    log.info("Pre-existing submodels found.")
    log.info("Deleting...")
    submodels_dir_path = os.path.join(main_dir_path,"models","submodels")
    for file in os.listdir(submodels_dir_path):
        if file.endswith(".json") or file.endswith(".tflite"):
            file_path = os.path.join(submodels_dir_path,file)
            utils.delete_file(file_path)

    log.info("Checking schema...")
    schema_path = os.path.join(main_dir_path,"schema","schema.fbs")
    if not os.path.exists(schema_path):    
        log.info("schema.fbs was not found, downloading...")
        urllib.request.urlretrieve("https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/schema/schema.fbs",
                                   "schema.fbs")
        log.info("Downloaded schema.fbs")
    else:
        log.info("schema.fbs found.")
        
    log.info("Patching source model in JSON...")
    source_model_dir_path = os.path.join(main_dir_path, "models", "source_model")
    source_model = utils.load_json_model(log,source_model_dir_path,main_dir_path)
    
    log.info("Analyzing mapping...")
    info = utils.info_mapping(mapping)
    log.info("Operations to be merged have following indexes: %s" % info)

    if len(info) != 0:
        merge_flag = True
    else:
        merge_flag = False
    
    while merge_flag:

        submodel,submodel_filename = utils.initialize_submodel(log, main_dir_path, info)
        utils.create_submodel(source_model, submodel, info, main_dir_path,submodel_filename)
        submodel_file_path = os.path.join(submodels_dir_path,submodel_filename)
        utils.convert_to_tflite(schema_path,submodel_file_path)

        if len(info) == 0:
            merge_flag = False
            break
    
    utils.docker_start()

    for file in os.listdir(main_dir_path):
        if file.endswith(".tflite"):
            submodel_tflite_file_path = os.path.join(main_dir_path,file)
            utils.docker_copy(submodel_tflite_file_path,TO_CONTAINER,CONTAINER_SUBMODELS_PATH)
            file_to_compile_path = os.path.join(CONTAINER_SUBMODELS_PATH,file)
            utils.docker_compile(file_to_compile_path)
            submodel_tflite_file_target_path = os.path.join(main_dir_path,"models","submodels",file)
            utils.move_file(submodel_tflite_file_path,submodel_tflite_file_target_path)

    log.info("Initializing optimized model...")
    optimized_model,optimized_model_filename = utils.initialize_optimized_model(log,main_dir_path)
    log.info("Optimized model initialized.")    


if __name__ == '__main__':
    
    main_dir_path = os.path.dirname(os.path.abspath(__file__))

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    tflite_model_optimization(log, main_dir_path)

    pass