
def import_data():
    import tensorflow as tf
    import numpy as np

    #Data Retrieval
    (data_train, labels_train), (data_test, labels_test) = tf.keras.datasets.mnist.load_data()

    #Reshaping Data
    data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], data_train.shape[2], 1)
    data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], data_test.shape[2], 1)

    #Defining Input Tensor Shape
    input_tensor_shape = (data_test.shape[1], data_test.shape[2], 1)

    #Defining tensor type
    data_train = data_train.astype('float32')
    data_test = data_test.astype('float32')

    #Normalizing/Standardization
    data_train /= 255
    data_test /= 255

    return data_test

def prep_op_dir(path_to_dir):
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
        #TODO Hardcoded for 'Nix Systems
        raise NotImplentedError

    return path_to_dir+"tmp/"

def clean_op_dir(path_to_dir):
    import os

    if (os.path.exists(path_to_dir)): 
        if (os.listdir(path=path_to_dir)!=0):
            rm_cmd = "rm -r " + path_to_dir
            os.system(rm_cmd)
    else:
        #TODO Hardcoded for 'Nix Systems
        raise NotImplentedError

def tflite_conversion(session, operation_name, operation, op_dir, input_place):
    import tensorflow as tf
    tf_model_filename = op_dir + operation_name + ".tflite"

    tmp_model_saved_dir=prep_op_dir(op_dir)                            #Clears SavedModelDir -- Necessary

    tf.compat.v1.saved_model.simple_save(session,                               #Saving Model into SavedModelDir
                                         tmp_model_saved_dir,                                          
                                         inputs={operation_name+"_input":input_place},                      
                                         outputs={operation_name+"_op":operation})                            
                                                                                                   
    converter=tf.lite.TFLiteConverter.from_saved_model(tmp_model_saved_dir)     #Creates Converter Object
    tflite_model=converter.convert()                                            #Performs tflite conversion with it

    clean_op_dir(tmp_model_saved_dir)
    
    open(tf_model_filename, "wb").write(tflite_model)                      #Writes Conversion to pre-defined tflite folder
