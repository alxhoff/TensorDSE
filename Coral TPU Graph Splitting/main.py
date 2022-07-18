import os
import json
import logging
from pathlib import Path
import urllib.request

import utils

FROM_CONTAINER = "from_container"
TO_CONTAINER   = "to_container"
CONTAINER_SUBMODELS_PATH = "/home/deb/TensorDSE/submodels"

old_mapping = [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],
[7,0],[8,0],[9,0],[10,0],[11,0],[12,0],[13,0],
[14,0],[15,0],[16,0],[17,0],[18,0],[19,0],[20,0],
[21,2],[22,2],[23,2],[24,2],[25,2],[26,2],[27,2],
[28,2],[29,2],[30,2]]

"mapping = [[0,1],[1,2],[2,1],[3,2],[4,2],[5,1],[6,1]]"

def tflite_model_optimization(log: logging.Logger, main_dir_path: str):
    
    log.info("Starting docker...")
    utils.docker_start()

    log.info("Cleaning pre-existing files...")

    submodels_dir_path = os.path.join(main_dir_path,"models","submodels")
    json_submodels_dir = os.path.join(main_dir_path,"models","submodels","json")
    for file in os.listdir(json_submodels_dir):
        json_file_path = os.path.join(json_submodels_dir,file)
        os.remove(json_file_path)
    tflite_submodels_dir = os.path.join(main_dir_path,"models","submodels","tflite")
    for file in os.listdir(tflite_submodels_dir):
        tflite_file_path = os.path.join(tflite_submodels_dir,file)
        os.remove(tflite_file_path)
    
    optimized_model_dir = os.path.join(main_dir_path,"models","optimized_model")
    json_opmodel_dir = os.path.join(optimized_model_dir,"json")
    for file in os.listdir(json_opmodel_dir):
        optimized_model_file_path = os.path.join(json_opmodel_dir,file)
        os.remove(optimized_model_file_path)
    tflite_opmodel_dir = os.path.join(optimized_model_dir,"tflite")
    for file in os.listdir(tflite_opmodel_dir):
        optimized_model_file_path = os.path.join(tflite_opmodel_dir,file)
        os.remove(optimized_model_file_path)

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
    source_model = utils.load_source_model(log,source_model_dir_path,main_dir_path)
    
    log.info("Analyzing mapping...")
    info,opt_mapping = utils.optimize_mapping(mapping)
    

    if (len(info) == 0):

        log.info("The model does not contain operations which can be merged")

    elif (len(info[0]) == len(mapping)):

        log.info("Preparing to merge all the operations in the model.")
        #TODO

    else:

        merge_flag = True

        log.info("Operations to be merged have following indexes: %s" % info)

        log.info("Initializing optimized model...")
        optimized_model,optimized_model_filename = utils.initialize_optimized_model(log,main_dir_path)
        
        log.info("Optimized model initialized.")

        compiled_submodel = {}
        op_count = 0
        submodel_created = False
        optimized_model,submodel_created,op_count = utils.update_optimized_model(source_model,compiled_submodel,optimized_model,mapping,submodel_created,op_count)

        while merge_flag:

            submodel,submodel_filename = utils.initialize_submodel(log, main_dir_path, info)
            utils.create_submodel(source_model, submodel, info, main_dir_path,submodel_filename)
            tflite_submodel_filename,tflite_submodel_path = utils.convert_to_tflite(schema_path,
                                                                                submodels_dir_path,
                                                                                submodel_filename)
            file_to_compile_path = os.path.join(CONTAINER_SUBMODELS_PATH,tflite_submodel_filename)
            utils.docker_copy(tflite_submodel_path,TO_CONTAINER,file_to_compile_path)
            utils.docker_compile(file_to_compile_path)
            submodel_name = tflite_submodel_filename.split(".",1)[0]
            compiled_model_filename = submodel_name + "_edgetpu" + ".tflite"
            compiled_file_target_path = os.path.join(main_dir_path,"models",
                                                                   "submodels",
                                                                   "tflite",
                                                                   compiled_model_filename)
            utils.docker_copy(compiled_model_filename, FROM_CONTAINER, compiled_file_target_path)
            utils.docker_clean()
            log.info("Converting compiled submodel to JSON...")
            json_compiled_submodel_filename,json_compiled_submodel_path = utils.convert_to_json(schema_path,
                                                                                                submodels_dir_path,
                                                                                                compiled_model_filename)
            log.info("JSON file: %s for compiled submodel created." % json_compiled_submodel_filename)

            json_compiled_submodel = utils.load_json_model(json_compiled_submodel_path)
            submodel_created = True
            optimized_model,submodel_created,op_count = utils.update_optimized_model(source_model,json_compiled_submodel,optimized_model,mapping,submodel_created,op_count)

            if len(info) == 0:
                merge_flag = False
                optimized_model_file_path = os.path.join(main_dir_path,"models","optimized_model","json",optimized_model_filename)
                with open(optimized_model_file_path,"w") as fout:
                    json.dump(optimized_model, fout, indent=2)
                utils.convert_to_tflite(schema_path,optimized_model_dir,optimized_model_filename)
                break
        

if __name__ == '__main__':
    
    main_dir_path = os.path.dirname(os.path.abspath(__file__))
    mapping_file_path = os.path.join(main_dir_path,"mappings","mapping_30_ops.csv")
    mapping = utils.read_op_mapping(mapping_file_path)
    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)
    tflite_model_optimization(log, main_dir_path)
    
    pass