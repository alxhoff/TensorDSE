GEN_OP_NAME=""
GEN_OP_IN_SHAPE=[]
GEN_OP_IN_TYPE="float32"

def deduce_op(compiled_file):
    f = compiled_file
    op = f.split("edge_")[1]
    op = op.split("_edgetpu.tflite")[0]

    return op

def fetch_file(directory, ending):
    import os
    from os import listdir
    from os.path import isfile, join

    for f in listdir(directory): 
        if (isfile(join(directory, f)) and f.endswith(ending)):
            return f
    
    return None

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
    
def activation_id(activ_func):

    options_dict = {
        "RELU" : "relu",
        "SOFTMAX" : "softmax",
        "DROPOUT" : None
    }

    default=None
    return options_dict.get(activ_func, default)

def prep_edge_dir(path_to_dir, op_name):
    #TODO 
    #Hardcoded for 'Nix Systems
    #Should be combined with similar functions in a generic way.

    import os

    if (os.path.exists(path_to_dir)): 
        edge_dir=path_to_dir + "edge"
        if (not os.path.exists(edge_dir)): 
            mkdir_cmd="mkdir " + edge_dir
            os.system(mkdir_cmd)
        else:
            rm_cmd = "rm -r " + edge_dir
            mkdir_cmd="mkdir " + edge_dir
            os.system(rm_cmd)
            os.system(mkdir_cmd)
    else:
        raise NotImplentedError("No folder for the specified operation available.")

    return path_to_dir+"edge/"

def prep_op_dir(path_to_dir, op_name):
    #TODO Hardcoded for 'Nix Systems
    import os

    if (os.path.exists(path_to_dir)): 
        tmp_dir=path_to_dir + "tmp"
        if (not os.path.exists(tmp_dir)): 
            mkdir_cmd="mkdir " + tmp_dir
            os.system(mkdir_cmd)
        else:
            rm_cmd = "rm -r " + tmp_dir
            mkdir_cmd="mkdir " + tmp_dir
            os.system(rm_cmd)
            os.system(mkdir_cmd)
    else:
        mkdir_cmd="mkdir " + path_to_dir + "op_name"
        os.system(mkdir_cmd)

    return path_to_dir+"tmp/"

def clean_op_dir(path_to_dir):
    #TODO Hardcoded for 'Nix Systems
    import os

    if (os.path.exists(path_to_dir)): 
        if (os.listdir(path=path_to_dir)!=0):
            rm_cmd = "rm -r " + path_to_dir
            os.system(rm_cmd)
    else:
        raise NotImplentedError

def save_session(session, operation_name, operation, op_dir, input_place):
    import tensorflow as tf

    tmp_model_saved_dir=prep_op_dir(op_dir, operation_name)                     #Clears SavedModelDir -- Necessary
    tf.compat.v1.saved_model.simple_save(session,                               #Saving Model into SavedModelDir
                                         tmp_model_saved_dir,                                          
                                         inputs={operation_name+"_input":input_place},                      
                                         outputs={operation_name+"_op":operation})                            
    return tmp_model_saved_dir
                                                                                                   
def prep_generator(op_name, input_place):
    import random

    global GEN_OP_NAME
    global GEN_OP_IN_SHAPE
    global GEN_OP_IN_TYPE

    GEN_OP_NAME=op_name
    GEN_OP_IN_TYPE=input_place.dtype
    
    for i in range(len(input_place.shape)):
        GEN_OP_IN_SHAPE.append(input_place.shape[i])

def generator():
    import numpy as np
    import tensorflow as tf

    (data_train, labels_train), (data_test, labels_test) = tf.keras.datasets.mnist.load_data()
    data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], data_train.shape[2], 1)
    data_train = data_train.astype("float32")
    data_train /= 255

    for input_value in tf.data.Dataset.from_tensor_slices(data_train).batch(1).take(1000):
        yield [input_value]

def tflite_quantization(converter):
    import tensorflow as tf

    converter.experimental_new_converter = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]                        #This enables Quantization
    converter.representative_dataset = tf.lite.RepresentativeDataset(generator) #This sets the representative dataset for quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] #This ensures that if OPS cant be quantized, the converter throws an error
    converter.target_spec.supported_types = [ tf.int8 ]                         #For full integer quantization, through supported types defaults to int8, declared for clarity
    converter.inference_input_type = tf.uint8 #or tf.uint8                      #These set the input and output tensors to uint8
    converter.inference_output_type = tf.uint8 #or tf.uint8

    return converter


def tflite_conversion(op_dir, model_saved_dir, operation_name, operation, input_place):
    import tensorflow as tf

    prep_generator(operation_name, input_place)
    edge_dir=prep_edge_dir(op_dir, operation_name)

    edge_tf_model_filename = edge_dir + "edge_"+ operation_name + ".tflite"
    tf_model_filename = op_dir + operation_name + ".tflite"

    converter=tf.lite.TFLiteConverter.from_saved_model(model_saved_dir)         #Creates Converter Object
    edge_converter=tflite_quantization(converter)                                    #Quanitzation of tflite model, necessary for edge

    edge_tflite_model=edge_converter.convert()                                            #Performs tflite conversion with it
    tflite_model=converter.convert()                                            #Performs tflite conversion with it

    clean_op_dir(model_saved_dir)

    open(tf_model_filename, "wb").write(tflite_model)                           #Writes Conversion to pre-defined tflite folder
    open(edge_tf_model_filename, "wb").write(edge_tflite_model)                           #Writes Conversion to pre-defined tflite folder

