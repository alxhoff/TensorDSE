def fetch_file(directory, ending):
    import os
    from os import listdir
    from os.path import isfile, join

    for f in listdir(directory): 
        if (isfile(join(directory, f)) and f.endswith(ending)):
            return f
    
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

def representative_dataset_gen():
    import tensorflow as tf
    (data_train, labels_train), (data_test, labels_test) = tf.keras.datasets.mnist.load_data()
    data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], data_train.shape[2], 1)

    for input_value in tf.data.Dataset.from_tensor_slices(data_train).batch(1).take(100):
        yield [input_value]

def tflite_conversion(session, operation_name, operation, op_dir, input_place):
    import tensorflow as tf
    tf_model_filename = op_dir + operation_name + ".tflite"

    tmp_model_saved_dir=prep_op_dir(op_dir, operation_name)                     #Clears SavedModelDir -- Necessary

    tf.compat.v1.saved_model.simple_save(session,                               #Saving Model into SavedModelDir
                                         tmp_model_saved_dir,                                          
                                         inputs={operation_name+"_input":input_place},                      
                                         outputs={operation_name+"_op":operation})                            
                                                                                                   
    converter=tf.lite.TFLiteConverter.from_saved_model(tmp_model_saved_dir)     #Creates Converter Object
    #converter.experimental_new_converter = False
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.representative_dataset = representative_dataset_gen
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.inference_input_type = input_place.dtype # or tf.uint8
    #converter.inference_output_type = operation.dtype # or tf.uint8

    tflite_model=converter.convert()                                            #Performs tflite conversion with it
    clean_op_dir(tmp_model_saved_dir)
    
    open(tf_model_filename, "wb").write(tflite_model)                           #Writes Conversion to pre-defined tflite folder

