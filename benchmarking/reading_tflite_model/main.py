# tflite folder generated using tflite schema and the flattbuffer compiler
# See  https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html

from utils import *

model_filename = "MNIST_model.tflite"
models_folder = "models/"

def class_code_to_name(cls, code):
    for name, value in cls.__dict__.items():
        if value == code:
            return name
    return None


def process_io_lengths(op):
    return (op.InputsLength(), op.OutputsLength())


def process_io(op):
    (input, output) = process_io_lengths(op)
    inputs = []
    outputs = []
    for i in range(input):
        inputs.append(op.Inputs(i))
    for i in range(output):
        outputs.append(op.Outputs(i))

    return (inputs, outputs)


def process_io_numpy(op):
    return (op.InputsAsNumpy(), op.OutputsAsNumpy())

def print_options(options):
    if options:
        for key, item in options.items():
            print("{} -> {}".format(key, item))


def process_options(op, options):
    op_type = class_code_to_name(sys.modules['tflite'].BuiltinOptions.BuiltinOptions, op.BuiltinOptionsType())
    if op_type != "NONE":
        import re

        opt = eval("sys.modules['tflite'].{}.{}()".format(op_type, op_type))
        opt.Init(options.Bytes, options.Pos)

        methods = [func for func in dir(opt) if
                   callable(getattr(opt, func)) and re.search(r'^((?!Init)(?!__)(?!{}).)*$'.format(op_type), func)]

        opts = {}
        for method in methods:
            opts[method] = eval("opt.{}()".format(method))

        return opts
    return None

def get_strides(options):
    return [1, options['StrideH'], options['StrideW'], 1]

def get_padding(options):
    return class_code_to_name(sys.modules['tflite'].Padding.Padding, options['Padding'])

# Pool size
def get_filter(options):
    return [options['StrideH'], options['StrideW']]

def get_input_tensor_shape(io):
    return io[0][0][0]

def get_kernel_shape(io):
    return io[0][1][0][1:]

def get_output_tensor_shape(io):
    return io[1][0]

def get_activation_function(options):
    return class_code_to_name(sys.modules['tflite'].ActivationFunctionType.ActivationFunctionType, options['FusedActivationFunction'])

def get_num_dims(options):
    return options['KeepNumDims']

def get_weights_format(options):
    return options['WeightsFormat']

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
    conv_dir = models_folder + "CONV_2D/"
    conv_ckpt = conv_dir + 'graph.ckpt'
    conv_saved_pb = conv_dir + 'graph_saved.pb'
    conv_frozen_pb = conv_dir + 'graph_frozen.pb'
    
    batch_size = 1
    filter_count = 32

    input_shape = get_input_tensor_shape(io) #Gets inputs shape
    kernel_shape = get_kernel_shape(io) #Gets kernel shape

    reshaped_data_test = np.reshape(data_test[0], input_shape)

    padding = get_padding(options) #Gets Padding
    strides = get_strides(options) #Gets Strides
    activ_func = get_activation_function(options) #Gets Activation Function

    conv_graph = tf.Graph() #Initializes Graph

    with conv_graph.as_default(), tf.compat.v1.Session() as sess:

        kernel_place = tf.Variable(tf.random.normal([3,3,1,filter_count], dtype="float32"), dtype=tf.float32) #Defining Kernel
        input_place = tf.raw_ops.Placeholder(dtype=tf.float32, shape=input_shape, name="CONV2D_input") #Defining input
        conv_2d = tf.nn.conv2d(input_place, filters=kernel_place, strides=strides, padding=padding, name="CONV2D_op") #Model Creation

        init = tf.compat.v1.global_variables_initializer() #Initialize global variables
        sess.run(init) #Runs Sessions initialization

        output_place = tf.identity(conv_2d,name="CONV2D_output") 
        output_place = sess.run(conv_2d, feed_dict={input_place:reshaped_data_test}) #Defining/Obtaining Output Tensors

        tf.io.write_graph(conv_graph, conv_dir, "graph_saved.pb", as_text=False)
        saver=tf.compat.v1.train.Saver()
        saver.save(sess,conv_ckpt)

        #tflite_class = tf.function(func=tf.compat.v1.lite.TFLiteConverter.from_session(sess,input_place,output_place)) #Converting to tflite model from session

        #freeze_graph.freeze_graph(conv_saved_pb,
        #                          "",
        #                          True,
        #                          conv_ckpt,
        #                          "OUTPUT_nodes",
        #                          "",
        #                          "",
        #                          conv_frozen_pb,
        #                          True,
        #                          None)

        #tflite_class = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(conv_pb,input_place,output_place,input_shape) #Converting to tflite from frozen graph


def process_MAX_POOL_2D(options, io):
    pool_size = get_filter(options)
    padding = get_padding(options)
    strides = get_strides(options)
    activ_func = get_activation_function(options)
    pass


def process_RESHAPE(options, io):
    output_shape = get_output_tensor_shape(io)
    pass


def process_FULLY_CONNECTED(options, io):
    activ_func = get_activation_function(options)
    keep_num_dim = get_num_dims(options)
    weights_format = get_weights_format(options)
    output_shape = get_output_tensor_shape(io)
    pass


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


    eval("process_" + op_name)(op_opts, (input_tensors, output_tensors)) #Calls the respective operation


def main():
    with open(model_filename, "rb") as f:
        model = sys.modules["tflite"].Model.Model.GetRootAsModel(f.read(), 0) #Gets Model

        graph = model.Subgraphs(0) #Retrieves Subgraphs

        for i in range(graph.OperatorsLength()): #Loops over Operations/Nodes?
            process_operation(model, graph, graph.Operators(i))


if __name__ == '__main__':
    import os, sys
    import tensorflow as tf
    from tensorflow.python.tools import freeze_graph

    path = os.path.join(os.path.dirname(__file__), "tflite")

    data_test = import_data()

    for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
        mod_name = '.'.join(["tflite", py])
        mod = __import__(mod_name, fromlist=[py])
        classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
        for cls in classes:
            setattr(sys.modules[__name__], cls.__name__, cls)

    main()
