# tflite folder generated using tflite schema and the flattbuffer compiler
# See  https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html

from deploy import *
from utils import *
from gets import *

model_filename = "source_models/MNIST_model.tflite"
model_example="source_models/efficientnet-edgetpu-S_quant_edgetpu.tflite"
models_folder = "single_layer_models/"
compiled_models_folder = "compiled_edge"

op_array=[]

# Functions to process each operation take the form of "process_" + the builtin opcode name that can
# be found in the TFLite schema under `BuiltinOperator`. This way the functions can be resolved using `eval` and
# the resolved builtin operator name.

def process_CONV_2D(options, io):

    #TODO DILLATION
    """ Conv2D layers have three inputs: 
        Input
        Weights
        Bias
    """

    #Relevant Folder Names
    op_name="CONV_2D"
    conv_dir = models_folder + op_name + "/"

    #Retrieving operation relevant variables.
    filter_count = 32

    input_shape = get_input_tensor_shape(io) 
    test_input = np.random.rand(input_shape[0],input_shape[1],                  #Defining test data
                                input_shape[2],input_shape[3])

    kernel_shape = get_kernel_shape(io) 

    padding = get_padding(options) 
    strides = get_strides(options)   
    activ_func = get_activation_function(options) 

    conv_graph=tf.Graph() 
    with conv_graph.as_default(), tf.compat.v1.Session() as sess:
        
        kernel_place = tf.Variable(tf.random.normal([3,3,1,filter_count],       #Defining Kernel
                                   dtype="float32"), 
                                   dtype=tf.float32) 
        
        input_place = tf.raw_ops.Placeholder(dtype=tf.float32,                  #Defining input
                                             shape=input_shape, 
                                             name=op_name+"_input") 
        
        conv_2d = tf.nn.conv2d(input_place, filters=kernel_place,               #Model Creation/Instantiation               
                               strides=strides, padding=padding, 
                               name=op_name+"_op")

        init = tf.compat.v1.global_variables_initializer()                      #Initialize global variables
        sess.run(init)                                                          #Runs Sessions initialization

        output_place = tf.identity(conv_2d,name=op_name+"_output")              #Naming Output
        output_place = sess.run(conv_2d, feed_dict={input_place:test_input})    #Running Session with input and obtaining Output Tensors
        
        tmp_model_saved_dir=save_session(sess, op_name, conv_2d, conv_dir, input_place)

    tflite_conversion(conv_dir, tmp_model_saved_dir, op_name, conv_2d, input_place)


def process_MAX_POOL_2D(options, io):

    op_name="MAX_POOL_2D"
    pool_dir = models_folder + op_name + "/"

    pool_size = get_filter(options)                 
    padding = get_padding(options)                  
    strides = get_strides(options)                  
    activ_func = get_activation_function(options)   

    input_shape = get_input_tensor_shape(io)        
    test_input = np.random.rand(input_shape[0],input_shape[1],                  #Defining test data
                                input_shape[2],input_shape[3])

    pool_2d = tf.Graph()                            
    with pool_2d.as_default(), tf.compat.v1.Session() as sess:
        
        input_place = tf.raw_ops.Placeholder(dtype=tf.float32,                  #Defining input
                                             shape=input_shape, 
                                             name=op_name+"_input")

        pool_2d = tf.nn.max_pool2d(input_place, 2,                              #Model Creation/Instantiation
                                   strides, padding,
                                   data_format='NHWC', 
                                   name=op_name+"_op")
                                    

        init = tf.compat.v1.global_variables_initializer()                      #Initialize global variables
        sess.run(init)                                                          #Runs Sessions initialization

        output_place = tf.identity(pool_2d ,name=op_name+"_output")              #Naming Output
        output_place = sess.run(pool_2d, feed_dict={input_place:test_input})    #Running Session and obtaining Output Tensors

        tmp_model_saved_dir=save_session(sess, op_name, pool_2d, pool_dir, input_place)

    tflite_conversion(pool_dir, tmp_model_saved_dir, op_name, pool_2d, input_place)


def process_RESHAPE(options, io):

    op_name="RESHAPE"
    reshape_dir = models_folder + op_name + "/"

    input_shape = get_input_tensor_shape(io)
    test_input = np.random.rand(input_shape[0],input_shape[1],
                                input_shape[2],input_shape[3])

    output_shape = get_output_tensor_shape(io)

    reshape_graph=tf.Graph()
    with reshape_graph.as_default(), tf.compat.v1.Session() as sess:


        input_place = tf.raw_ops.Placeholder(dtype=tf.int32, 
                                             shape=input_shape, 
                                             name=op_name+"_input")

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        reshape = tf.reshape(input_place, 
                            (output_shape[0][0], output_shape[0][1]), 
                            name=op_name+"_op")

        output_place = tf.identity(reshape ,name=op_name+"_output")
        output_place = sess.run(reshape, feed_dict={input_place:test_input})

        tmp_model_saved_dir=save_session(sess, op_name, reshape, reshape_dir, input_place)

    #tflite_conversion(reshape_dir, tmp_model_saved_dir, op_name, reshape, input_place)



def process_FULLY_CONNECTED(options, io):

    op_name="FULLY_CONNECTED"
    fcl_dir = models_folder + op_name + "/"

    input_shape = get_input_tensor_shape(io)
    test_input = np.random.rand(input_shape[0],input_shape[1])

    output_shape = get_output_tensor_shape(io)
    units=output_shape[0][1]                                                    #Not sure about this

    weights_format = get_weights_format(options)

    activ_func = get_activation_function(options)
    activ_func = activation_id(activ_func)

    keep_num_dim = get_num_dims(options)                                        # Is this bias?

    fully_connected_graph=tf.Graph()
    with fully_connected_graph.as_default(), tf.compat.v1.Session() as sess:
        
        input_place = tf.raw_ops.Placeholder(dtype=tf.float32, 
                                             shape=input_shape, 
                                             name=op_name+"_input")
        
        FCL = tf.compat.v1.layers.dense(input_place,
                                        units,
                                        activ_func,
                                        use_bias=keep_num_dim,
                                        name=op_name+"_op")

        init = tf.compat.v1.global_variables_initializer()                          
        sess.run(init)                                                              

        output_place = tf.identity(FCL ,name=op_name+"_output")                          
        output_place = sess.run(FCL, feed_dict={input_place:test_input})            

        tmp_model_saved_dir=save_session(sess, op_name, FCL, fcl_dir, input_place)

    #tflite_conversion(fcl_dir, tmp_model_saved_dir, op_name, FCL, input_place)

def process_SOFTMAX(options, io):
    op_name="SOFTMAX"
    softmx_dir = models_folder + op_name + "/"

    input_shape = get_input_tensor_shape(io)
    test_input = np.random.rand(input_shape[0],input_shape[1])

    output_shape = get_output_tensor_shape(io)
    units = output_shape[0][1]

    softmx_graph=tf.Graph()
    with softmx_graph.as_default(), tf.compat.v1.Session() as sess:
        
        input_place = tf.raw_ops.Placeholder(dtype=tf.float32, 
                                             shape=input_shape, 
                                             name=op_name+"_input")

        Soft = tf.nn.softmax(input_place,None, op_name+"_op")

        init = tf.compat.v1.global_variables_initializer()                          
        sess.run(init)                                                              

        output_place = tf.identity(Soft ,name=op_name+"_output")                          
        output_place = sess.run(Soft, feed_dict={input_place:test_input})            

        tmp_model_saved_dir=save_session(sess, op_name, Soft, softmx_dir, input_place)

    tflite_conversion(softmx_dir, tmp_model_saved_dir, op_name, Soft, input_place)


def process_operation(model, graph, op):

    opcode_builtin = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() #Necessary OP Code Intex to retrieve OP_NAME
    op_name = class_code_to_name(sys.modules["tflite"].BuiltinOperator.BuiltinOperator, opcode_builtin) #Operation Name
    op_array.append(op_name)
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

    eval("process_" + op_name)(op_opts, (input_tensors, output_tensors)) #Calls the respective operation

def source_tflite_conversion():
    with open(model_filename, "rb") as f:
        model = sys.modules["tflite"].Model.Model.GetRootAsModel(f.read(), 0) #Gets Model
        graph = model.Subgraphs(0) #Retrieves Subgraphs

        for i in range(graph.OperatorsLength()): #Loops over Operations/Nodes?
            process_operation(model, graph, graph.Operators(i))


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

    source_tflite_conversion()

    #edge_tflite_compilation()

    edge_group_tflite_deployment(compiled_models_folder)
    #gpu_group_tflite_deployment()
    #cpu_group_tflite_deployment(models_folder, op_array)
