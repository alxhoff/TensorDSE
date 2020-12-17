import os
import json
import logging
from pathlib import Path
import urllib.request

import utils

mapping = [[0,2],[1,2],[2,2],[3,0],[4,2],[5,2],[6,2]]

def tflite_model_optimization(log: logging.Logger, main_dir_path: str):
    
    schema_path = os.path.join(main_dir_path,"schema","schema.fbs")
    if not os.path.exists(schema_path):    
        log.info("schema.fbs was not found, downloading...")
        urllib.request.urlretrieve("https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/schema/schema.fbs",
                                   "schema.fbs")
        log.info("Downloaded schema.fbs")
        
    log.info("Patching source model in JSON...")
    source_model_dir_path = os.path.join(main_dir_path, "models", "source_model")
    source_model = utils.load_json_model(log,source_model_dir_path,main_dir_path)
    
    log.info("Deleting existing submodels...")
    submodels_dir_path = os.path.join(main_dir_path,"models","submodels")
    for file in os.listdir(submodels_dir_path):
        if file.endswith(".json"):
            file_path = os.path.join(submodels_dir_path,file)
            utils.delete_file(file_path)

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
    
    #log.info("")


    
    """log.info("Creating tmp files...")
    tmp_folder_dir = os.path.dirname(os.path.dirname(source_model_filepath))
    tmp_folder_path = os.path.join(tmp_folder_dir, "submodels", "tmp_files")

    utils.echo_run("mkdir", tmp_folder_path)

    if not os.path.exists(tmp_folder_path):
        print("no")
    else:
        print("yes")
        utils.echo_run("rm", "-rf", tmp_folder_path)"""

if __name__ == '__main__':
    
    main_dir_path = os.path.dirname(os.path.abspath(__file__))

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    tflite_model_optimization(log, main_dir_path)

    pass