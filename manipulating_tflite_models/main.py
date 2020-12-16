import os
import json
import logging
from pathlib import Path

import utils

mapping = [[0,2],[1,2],[2,2],[3,0],[4,2],[5,2],[6,2]]

def tflite_model_optimization(log: logging.Logger, main_dir_path: str):
    
    log.info("Patching original model in JSON...")
    source_model_dir_path = os.path.join(main_dir_path, "models", "source_model")
    original_model = utils.load_json_file(log,source_model_dir_path)
    
    log.info("Analyzing mapping...")
    info = utils.info_mapping(mapping)
    log.info("Operations to be merged have following indexes: %s" % info)

    if len(info) != 0:
        merge_flag = True
    else:
        merge_flag = False
    

    while merge_flag:
        submodel,submodel_filename = utils.initialize_submodel_file(log, main_dir_path, info)
        info, submodel = utils.merge_ops(original_model, submodel, info, main_dir_path,submodel_filename)
        if len(info) == 0:
            merge_flag = False
            break

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
    """models_dir = os.path.join(main_dir_path, "models")
    
    source_model_filename,source_model_filepath = utils.check_for_source_model(models_dir)"""

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    tflite_model_optimization(log, main_dir_path)

    pass