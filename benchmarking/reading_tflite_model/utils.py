
gen_op_name=""
gen_op_in_shape=[]
gen_op_in_type="float32"

def retrieve_folder_path(path, folder):
    return path.split(folder + "/")[0] + folder + "/"


def fetch_file(directory, ending):
    import os
    from os import listdir
    from os.path import isfile, join

    for f in listdir(directory): 
        if (isfile(join(directory, f)) and f.endswith(ending)):
            return f
    
    return None
    

def get_numpy_type(tensor_type):
    import numpy as np

    TF_TYPE_DICT = {
      'float32': np.float32,
      'int32': np.int32
    }

    default=None
    return TF_TYPE_DICT.get(tensor_type, default)
    

def get_activation_id(activ_func):

    options_dict = {
        "RELU" : "relu",
        "SOFTMAX" : "softmax",
        "DROPOUT" : None
    }

    default=None
    return options_dict.get(activ_func, default)


def create_csv_file(path_file, folder_name, results):
    import os
    import csv

    csv_dir = path_file + folder_name + "/"
    csv_file = csv_dir + "/" + "Results.csv"

    if (os.path.exists(path_file)): 
        if (not os.path.exists(csv_dir)): 
            mkdir_cmd="mkdir " + csv_dir
            os.system(mkdir_cmd)
        else:
            clean_up_cmd="rm -r " + csv_dir + "*"
            os.system(clean_up_cmd)
    else:
        raise NotImplentedError

    with open(csv_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(results)):
            fw.writerow([results[i][0], results[i][1]])


def extend_directory(path_to_dir, extended_dir, parent_dir):
    #TODO Hardcoded for 'Nix Systems
    import os

    if (os.path.exists(path_to_dir)): 
        ext_dir=path_to_dir + extended_dir
        if (not os.path.exists(ext_dir)): 
            mkdir_cmd="mkdir " + ext_dir
            os.system(mkdir_cmd)
        else:
            rm_cmd = "rm -r " + ext_dir
            mkdir_cmd="mkdir " + ext_dir
            os.system(rm_cmd)
            os.system(mkdir_cmd)
    else:
        ext_dir=path_to_dir + extended_dir
        mkdir_cmd="mkdir -p" + ext_dir
        os.system(mkdir_cmd)

    return ext_dir + "/"


def clean_directory(path_to_dir):
    #TODO Hardcoded for 'Nix Systems
    import os

    if (os.path.exists(path_to_dir)): 
        if (os.listdir(path=path_to_dir)!=0):
            rm_cmd = "rm -r " + path_to_dir
            os.system(rm_cmd)
    else:
        raise NotImplentedError

def prep_dataset_generator(op_name, input_place):
    import random

    global gen_op_name
    global gen_op_in_shape
    global gen_op_in_type

    gen_op_in_shape = []
    gen_op_name=op_name
    gen_op_in_type=get_numpy_type(str(input_place.dtype).split(" '")[1].split("'>")[0])
    
    for i in range(len(input_place.shape)):
        gen_op_in_shape.append(input_place.shape[i])

    gen_op_in_shape = tuple(gen_op_in_shape)


def generator():
    import tensorflow as tf
    import numpy as np

    for _ in range(100):
        input_data = np.array(np.random.random_sample(gen_op_in_shape), dtype=gen_op_in_type)
        input_data = tf.convert_to_tensor(input_data)
        yield [input_data]


def tflite_quantization(converter):
    import tensorflow as tf

    converter.experimental_new_converter = False

    # This enables Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # This sets the representative dataset for quantization
    converter.representative_dataset = tf.lite.RepresentativeDataset(generator) 

    # This ensures that if OPS cant be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # For full integer quantization, through supported types defaults to int8, declared for clarity
    converter.target_spec.supported_types = [ tf.int8 ]
    converter.inference_input_type = tf.uint8  # or tf.uint8
    converter.inference_output_type = tf.uint8  # or tf.uint8

    return converter


def save_session(session, operation_name, operation, op_dir, input_place):
    import tensorflow as tf

    # Clears saved model directory.
    tmp_model_saved_dir=extend_directory(op_dir, "tmp", operation_name)

    # Saving Model into the saved model directory.
    tf.compat.v1.saved_model.simple_save(session,
                                         tmp_model_saved_dir,                                          
                                         inputs={operation_name+"_input":input_place},                      
                                         outputs={operation_name+"_op":operation})                            
    return tmp_model_saved_dir
                                                                                                   
def tflite_conversion(op_dir, model_saved_dir, operation_name, operation, input_place):
    import tensorflow as tf

    tf_model_filename = op_dir + operation_name + ".tflite"
    prep_dataset_generator(operation_name, input_place)

    edge_dir=extend_directory(op_dir,"quant", operation_name)
    edge_tf_model_filename = edge_dir + "quant_"+ operation_name + ".tflite"

    # Creates Converter Object.
    converter=tf.lite.TFLiteConverter.from_saved_model(model_saved_dir)
    # Creates another Converter Object.
    edge_converter=tf.lite.TFLiteConverter.from_saved_model(model_saved_dir)
    # Quanitzation of tflite model, necessary for edge
    edge_converter=tflite_quantization(edge_converter)
    # Performs tflite conversion with it.
    tflite_model=converter.convert()
    # Performs quantized tflite conversion with it.
    edge_tflite_model=edge_converter.convert()

    # Writes Conversion to pre-defined tflite folder.
    open(tf_model_filename, "wb").write(tflite_model)
    # Writes Conversion to pre-defined tflite folder
    open(edge_tf_model_filename, "wb").write(edge_tflite_model)

    clean_directory(model_saved_dir)

