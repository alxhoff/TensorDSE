from analyze import *
from docker import *
from deploy import *
from utils import *
from gets import *

COMPILED_MODELS_FOLDER = "models/tpu_compiled_models/"
MODELS_FOLDER = "models/single_layer_models/"

source_model_filename = ""

# Functions to process each operation take the form of "process_" + the builtin opcode name that can
# be found in the TFLite schema under `BuiltinOperator`. This way the functions can be resolved using `eval` and
# the resolved builtin operator name.


def process_CONV_2D(options, io):
    """
    Processes necessary information to create a 2D convolution model from
    the given arguments.
    """
    import numpy as np

    op_name = "CONV_2D"
    conv_dir = f"{MODELS_FOLDER}{op_name}/"

    # Retrieving operation relevant variables.
    filter_count = 28

    input_shape = get_input_tensor_shape(io)
    test_input = np.array(np.random.random_sample(input_shape))

    kernel_shape = get_kernel_shape(io)

    padding = get_padding(options)
    strides = get_strides(options)
    activ_func = get_activation_function(options)

    conv_graph = tf.Graph()
    with conv_graph.as_default(), tf.compat.v1.Session() as sess:

        kernel_place = tf.Variable(tf.random.normal([3, 3, 1, filter_count],
                                                    dtype="float32"),
                                   dtype=tf.float32)

        input_place = tf.raw_ops.Placeholder(dtype=tf.float32,
                                             shape=input_shape,
                                             name=op_name+"_input")

        conv_2d = tf.nn.conv2d(input_place, filters=kernel_place,
                               strides=strides, padding=padding,
                               name=op_name+"_op")

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        output_place = tf.identity(conv_2d, name=op_name+"_output")
        output_place = sess.run(conv_2d, feed_dict={input_place: test_input})

        tmp_model_saved_dir = save_session(
            sess, op_name, conv_2d, conv_dir, input_place)

    tflite_conversion(conv_dir, tmp_model_saved_dir,
                      op_name, conv_2d, input_place)


def process_MAX_POOL_2D(options, io):
    """
    Processes necessary information to create a 2D pooling model from
    the given arguments.
    """
    import numpy as np

    op_name = "MAX_POOL_2D"
    pool_dir = f"{MODELS_FOLDER}{op_name}/"

    pool_size = get_filter(options)
    padding = get_padding(options)
    strides = get_strides(options)
    activ_func = get_activation_function(options)

    input_shape = get_input_tensor_shape(io)
    test_input = np.array(np.random.random_sample(input_shape))

    pool_2d = tf.Graph()

    with pool_2d.as_default(), tf.compat.v1.Session() as sess:

        input_place = tf.raw_ops.Placeholder(dtype=tf.float32,
                                             shape=input_shape,
                                             name=op_name+"_input")

        pool_2d = tf.nn.max_pool2d(input_place, 2,
                                   strides, padding,
                                   data_format='NHWC',
                                   name=op_name+"_op")

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        output_place = tf.identity(pool_2d, name=op_name+"_output")
        output_place = sess.run(pool_2d, feed_dict={input_place: test_input})

        tmp_model_saved_dir = save_session(
            sess, op_name, pool_2d, pool_dir, input_place)

    tflite_conversion(pool_dir, tmp_model_saved_dir,
                      op_name, pool_2d, input_place)


def process_RESHAPE(options, io):
    """
    Processes necessary information to create a reshaping operation from
    the given arguments.
    """
    import numpy as np

    op_name = "RESHAPE"
    reshape_dir = f"{MODELS_FOLDER}{op_name}/"

    input_shape = get_input_tensor_shape(io)
    test_input = np.array(np.random.random_sample(input_shape), dtype=np.int32)

    output_shape = get_output_tensor_shape(io)

    reshape_graph = tf.Graph()
    with reshape_graph.as_default(), tf.compat.v1.Session() as sess:

        input_place = tf.raw_ops.Placeholder(dtype=tf.int32,
                                             shape=input_shape,
                                             name=op_name+"_input")

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        reshape = tf.reshape(input_place,
                             (output_shape[0][0], output_shape[0][1]),
                             name=op_name+"_op")

        output_place = tf.identity(reshape, name=op_name+"_output")
        output_place = sess.run(reshape, feed_dict={input_place: test_input})

        tmp_model_saved_dir = save_session(
            sess, op_name, reshape, reshape_dir, input_place)

    tflite_conversion(reshape_dir, tmp_model_saved_dir,
                      op_name, reshape, input_place)


def process_FULLY_CONNECTED(options, io):
    """
    Processes necessary information to create a fully connected layer model from
    the given arguments.
    """
    import numpy as np

    activ_func = get_activation_function(options)
    activ_func = get_activation_id(activ_func)

    op_name = "FULLY_CONNECTED_" + activ_func if activ_func else "FULLY_CONNECTED"
    fcl_dir = f"{MODELS_FOLDER}{op_name}/"

    input_shape = get_input_tensor_shape(io)
    test_input = np.array(np.random.random_sample(input_shape))

    output_shape = get_output_tensor_shape(io)
    units = output_shape[0][1]

    weights_format = get_weights_format(options)

    keep_num_dim = get_num_dims(options)

    fully_connected_graph = tf.Graph()

    with fully_connected_graph.as_default(), tf.compat.v1.Session() as sess:

        input_place = tf.raw_ops.Placeholder(dtype=tf.float32,
                                             shape=input_shape,
                                             name=op_name+"_input")

        fcl = tf.compat.v1.layers.dense(input_place,
                                        units,
                                        activ_func,
                                        use_bias=keep_num_dim,
                                        name=op_name+"_op")

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        output_place = tf.identity(fcl, name=op_name+"_output")
        output_place = sess.run(fcl, feed_dict={input_place: test_input})

        tmp_model_saved_dir = save_session(
            sess, op_name, fcl, fcl_dir, input_place)

    tflite_conversion(fcl_dir, tmp_model_saved_dir, op_name, fcl, input_place)


def process_SOFTMAX(options, io):
    """
    Processes necessary information to create a softmax operation from
    the given arguments.
    """
    import numpy as np

    op_name = "SOFTMAX"
    softmx_dir = f"{MODELS_FOLDER}{op_name}/"

    input_shape = get_input_tensor_shape(io)
    test_input = np.array(np.random.random_sample(input_shape))

    output_shape = get_output_tensor_shape(io)
    units = output_shape[0][1]

    softmx_graph = tf.Graph()
    with softmx_graph.as_default(), tf.compat.v1.Session() as sess:

        input_place = tf.raw_ops.Placeholder(dtype=tf.float32,
                                             shape=input_shape,
                                             name=op_name+"_input")

        Soft = tf.nn.softmax(input_place, None, op_name+"_op")

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        output_place = tf.identity(Soft, name=op_name+"_output")
        output_place = sess.run(Soft, feed_dict={input_place: test_input})

        tmp_model_saved_dir = save_session(
            sess, op_name, Soft, softmx_dir, input_place)

    tflite_conversion(softmx_dir, tmp_model_saved_dir,
                      op_name, Soft, input_place)


def process_operation(model, graph, op):
    """
    Processes necessary information to create a softmax operation from
    the given arguments.
    """

    opcode_builtin = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()

    op_name = class_code_to_name(sys.modules["tflite"].BuiltinOperator.BuiltinOperator,
                                 opcode_builtin)
    op_opts = process_options(op, op.BuiltinOptions())

    op_opts_name = class_code_to_name(sys.modules["tflite"].BuiltinOptions.BuiltinOptions,
                                      op.BuiltinOptionsType())

    io_lengths = process_io_lengths(op)
    io = process_io(op)  # IO indexes are retreived.

    input_tensors = []
    for i, index in enumerate(io[0]):
        tensor = graph.Tensors(index)
        input_tensors.append(
            (tensor.ShapeAsNumpy(), class_code_to_name(sys.modules["tflite"].TensorType.TensorType,
                                                       tensor.Type())))

    output_tensors = []
    for i, index in enumerate(io[1]):
        tensor = graph.Tensors(index)
        output_tensors.append(
            (tensor.ShapeAsNumpy(), class_code_to_name(sys.modules["tflite"].TensorType.TensorType,
                                                       tensor.Type())))

    # Calls respective operation function.
    eval("process_" + op_name)(op_opts, (input_tensors, output_tensors))


def source_tflite_conversion():
    """
    Individually calls onto the process_operation function to prepare the corresponding
    function call that recreates the chosen operation.
    """
    with open(source_model_filename, "rb") as f:
        model = sys.modules["tflite"].Model.Model.GetRootAsModel(f.read(), 0)
        graph = model.Subgraphs(0)

        for i in range(graph.OperatorsLength()):
            process_operation(model, graph, graph.Operators(i))


if __name__ == '__main__':
    import os
    import sys
    import argparse
    import tensorflow as tf

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model',
                        default="models/source_models/MNIST_model.tflite",
                        help='File path to the SOURCE .tflite file.')

    parser.add_argument('-c', '--count',
                        type=int, default=1000,
                        help='Number of times to run inference.')

    args = parser.parse_args()
    source_model_filename = args.model

    path = os.path.join(os.path.dirname(__file__), "tflite")

    for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
        mod_name = '.'.join(["tflite", py])
        mod = __import__(mod_name, fromlist=[py])
        classes = [getattr(mod, x) for x in dir(
            mod) if isinstance(getattr(mod, x), type)]
        for cls in classes:
            setattr(sys.modules[__name__], cls.__name__, cls)

    source_tflite_conversion()
    edge_tflite_compilation()
    full_tflite_deployment(count=args.count)
    full_tflite_analysis()
