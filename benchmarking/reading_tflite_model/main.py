# tflite folder generated using tflite schema and the flattbuffer compiler
# See  https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html

from utils import *
from gets import *

model_filename = "MNIST_model.tflite"
models_folder = "models/"
saved_models_folder = "SavedModelDir/"
tflite_models_folder = "TfliteModel/"

# Functions to process each operation take the form of "process_" + the builtin opcode name that can
# be found in the TFLite schema under `BuiltinOperator`. This way the functions can be resolved using `eval` and
# the resolved builtin operator name.

OP_ARRAY = []

def process_CONV_2D(options, io):

    #TODO DILLATION
    """ Conv2D layers have three inputs: 
        Input
        Weights
        Bias
    """

    conv_dir = models_folder + "CONV_2D/"
    conv_ckpt = conv_dir + 'graph.ckpt'
    model_saved_dir = conv_dir + saved_models_folder
    op_model_filename = conv_dir + tflite_models_folder + "CONV2D_model.tflite"

    #Constants
    batch_size = 1
    filter_count = 32

    input_shape = get_input_tensor_shape(io)        #Gets inputs shape
    kernel_shape = get_kernel_shape(io)             #Gets kernel shape

    padding = get_padding(options)                  #Gets Padding
    strides = get_strides(options)                  #Gets Strides
    activ_func = get_activation_function(options)   #Gets Activation Function

    conv_graph = tf.Graph()

    with conv_graph.as_default(), tf.compat.v1.Session() as sess:

        #Defining test data
        test_input = np.random.rand(input_shape[0],input_shape[1],
                                    input_shape[2],input_shape[3])
        #Defining Kernel
        kernel_place = tf.Variable(tf.random.normal([3,3,1,filter_count], 
                                   dtype="float32"), 
                                   dtype=tf.float32) 

        #Defining input
        input_place = tf.raw_ops.Placeholder(dtype=tf.float32, 
                                             shape=input_shape, 
                                             name="CONV2D_input") 
        #Model Creation/Instantiation
        conv_2d = tf.nn.conv2d(input_place, filters=kernel_place,               
                               strides=strides, padding=padding, 
                               name="CONV2D_op")

        init = tf.compat.v1.global_variables_initializer()                          #Initialize global variables
        sess.run(init)                                                              #Runs Sessions initialization

        saver = tf.compat.v1.train.Saver()                                          #Declaring Saver Object
        saver.save(sess, conv_ckpt)                                                 #Saving Checkpoints

        output_place = tf.identity(conv_2d,name="CONV2D_output")                    #Naming Output
        output_place = sess.run(conv_2d, feed_dict={input_place:test_input})        #Running Session with input and obtaining Output Tensors

        clear_op_dir("CONV2D", model_saved_dir)                                     #Clears SavedModelDir -- Necessary
        tf.compat.v1.saved_model.simple_save(sess,                                  #Saving Model into SavedModelDir
                                             model_saved_dir,                                          
                                             inputs={"CONV2D_input":input_place},                      
                                             outputs={"CONV2D_op":conv_2d})                            
                                                                                                       
        converter=tf.lite.TFLiteConverter.from_saved_model(model_saved_dir)         #Creates Converter Object
        tflite_model=converter.convert()                                            #Performs tflite conversion with it
        open(op_model_filename, "wb").write(tflite_model)                           #Writes Conversion to pre-defined tflite folder


def process_MAX_POOL_2D(options, io):

    pool_dir = models_folder + "POOL2D/"
    pool_ckpt = pool_dir + 'graph.ckpt'
    model_saved_dir = pool_dir + saved_models_folder
    op_model_filename = pool_dir + tflite_models_folder + "POOL2D_model.tflite"

    batch_size=1

    pool_size = get_filter(options)                 #Gets pool size
    padding = get_padding(options)                  #Gets Padding
    strides = get_strides(options)                  #Gets Strides
    activ_func = get_activation_function(options)   #Gets Activation Function

    input_shape = get_input_tensor_shape(io)        #Gets inputs shape

    pool_2d = tf.Graph()                            #Initializes Graph

    with pool_2d.as_default(), tf.compat.v1.Session() as sess:

        #Defining test data
        test_input = np.random.rand(input_shape[0],input_shape[1],
                                    input_shape[2],input_shape[3])

        #Defining input
        input_place = tf.raw_ops.Placeholder(dtype=tf.float32, 
                                             shape=input_shape, 
                                             name="POOL2D_input")

        #Model Creation/Instantiation
        pool_2d = tf.nn.max_pool2d(input_place, 2,
                                   strides, padding,
                                   data_format='NHWC', name=None)
                                    

        init = tf.compat.v1.global_variables_initializer()                          #Initialize global variables
        #saver = tf.compat.v1.train.Saver()

        sess.run(init)                                                              #Runs Sessions initialization

        output_place = tf.identity(pool_2d ,name="POOL2D_output")                   #Naming Output
        output_place = sess.run(pool_2d, feed_dict={input_place:test_input})        #Running Session and obtaining Output Tensors

        clear_op_dir("POOL2D", model_saved_dir)                                     #Clears SavedModelDir -- Necessary
        tf.compat.v1.saved_model.simple_save(sess,                                  #Saving Model into SavedModelDir
                                             model_saved_dir, 
                                             inputs={"POOL2D_input":input_place}, 
                                             outputs={"POOL2D_op":pool_2d})

        converter=tf.lite.TFLiteConverter.from_saved_model(model_saved_dir)         #Creates Converter Object
        tflite_model=converter.convert()                                            #Performs tflite conversion with it
        open(op_model_filename, "wb").write(tflite_model)                           #Writes Conversion


def process_RESHAPE(options, io):                   #FLATTEN

    reshape_dir = models_folder + "RESHAPE/"
    model_saved_dir = reshape_dir + saved_models_folder
    op_model_filename = reshape_dir + tflite_models_folder + "RESHAPE_model.tflite"

    input_shape = get_input_tensor_shape(io)
    output_shape = get_output_tensor_shape(io)

    reshape_graph=tf.Graph()                        #Creating Flatten/Reshape Graph

    with reshape_graph.as_default(), tf.compat.v1.Session() as sess:

        test_input = np.random.rand(input_shape[0],input_shape[1],
                                    input_shape[2],input_shape[3])


        input_place = tf.raw_ops.Placeholder(dtype=tf.int32, 
                                             shape=input_shape, 
                                             name="RESHAPE_input")

        sess.run(tf.compat.v1.global_variables_initializer())

        flattened_op = tf.reshape(input_place, 
                                 (output_shape[0][0], output_shape[0][1]), 
                                 name="RESHAPE_op")

        sess.run(flattened_op, feed_dict={input_place:test_input})

        clear_op_dir("POOL2D", model_saved_dir)                                     #Clears SavedModelDir -- Necessary
        tf.compat.v1.saved_model.simple_save(sess,                                  #Saving Model into SavedModelDir
                                             model_saved_dir, 
                                             inputs={"RESHAPE_input":input_place}, 
                                             outputs={"RESHAPE_op":flattened_op})

        converter=tf.lite.TFLiteConverter.from_saved_model(model_saved_dir)         #Creates Converter Object
        tflite_model=converter.convert()                                            #Performs tflite conversion with it
        open(op_model_filename, "wb").write(tflite_model)                           #Writes Conversion



def process_FULLY_CONNECTED(options, io):

    input_shape = get_input_tensor_shape(io)
    output_shape = get_output_tensor_shape(io)

    weights_format = get_weights_format(options)
    activ_func = get_activation_function(options)

    keep_num_dim = get_num_dims(options)

    fully_connected_graph=tf.Graph()                        #Creating FCC Graph

    with fully_connected_graph.as_default(), tf.compat.v1.Session() as sess:
    
        pass
        #test_input = np.random.rand(input_shape[0],input_shape[1],
        #                            input_shape[2],input_shape[3])

        ##Defining input
        #input_place = tf.raw_ops.Placeholder(dtype=tf.float32, 
        #                                     shape=input_shape, 
        #                                     name="FCC_input")

        ##Model Creation/Instantiation
        #FCC = tf.add(tf.matmul(input_place, weights_format['wd1']), biases['bd1'])                               

        #init = tf.compat.v1.global_variables_initializer()                          #Initialize global variables
        ##saver = tf.compat.v1.train.Saver()

        #sess.run(init)                                                              #Runs Sessions initialization

        #output_place = tf.identity(pool_2d ,name="POOL2D_output")                   #Naming Output
        #output_place = sess.run(pool_2d, feed_dict={input_place:test_input})        #Running Session and obtaining Output Tensors

def process_SOFTMAX(options, io):
    pass


def process_operation(model, graph, op):

    opcode_builtin = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() #Necessary OP Code Intex to retrieve OP_NAME
    op_name = class_code_to_name(sys.modules["tflite"].BuiltinOperator.BuiltinOperator, opcode_builtin) #Operation Name
    op_opts = process_options(op, op.BuiltinOptions()) #Operaton Options
    op_opts_name = class_code_to_name(sys.modules["tflite"].BuiltinOptions.BuiltinOptions,op.BuiltinOptionsType()) #Name of Operation Options

    io_lengths = process_io_lengths(op) #io_lengths [0]/[1] - Number of Input/Output Tensors
    io = process_io(op) #io indexes needed to retrieve the actual i/o tensors

    input_tensors = []
    for i, index in enumerate(io[0]): #Looping through input tensor range
        tensor = graph.Tensors(index)
        input_tensors.append(
            (tensor.ShapeAsNumpy(), class_code_to_name(sys.modules["tflite"].TensorType.TensorType, tensor.Type())))

    output_tensors = []
    for i, index in enumerate(io[1]): #Looping through output tensor range
        tensor = graph.Tensors(index)
        output_tensors.append(
            (tensor.ShapeAsNumpy(), class_code_to_name(sys.modules["tflite"].TensorType.TensorType, tensor.Type())))


    OP_ARRAY.append(op_name)
    eval("process_" + op_name)(op_opts, (input_tensors, output_tensors)) #Calls the respective operation


def main():
    with open(model_filename, "rb") as f:
        model = sys.modules["tflite"].Model.Model.GetRootAsModel(f.read(), 0) #Gets Model
        graph = model.Subgraphs(0) #Retrieves Subgraphs

        for i in range(graph.OperatorsLength()): #Loops over Operations/Nodes?
            process_operation(model, graph, graph.Operators(i))

        for OP in OP_ARRAY:
            print(OP)


if __name__ == '__main__':
    import os, sys
    import tensorflow as tf
    from tensorflow.python.tools import freeze_graph
    from tensorflow.compat.v1.train import Saver as saver
    import numpy as np

    path = os.path.join(os.path.dirname(__file__), "tflite")

    for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
        mod_name = '.'.join(["tflite", py])
        mod = __import__(mod_name, fromlist=[py])
        classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
        for cls in classes:
            setattr(sys.modules[__name__], cls.__name__, cls)

    main()
